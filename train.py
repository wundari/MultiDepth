# %%
import torch
import torch.nn as nn

from pathlib import Path
from time import perf_counter

from models.stereo.raft_stereo_mono_beta_vlm_flux_core import (
    RAFTStereoMonoBetaVLMFluxConfig,
    RAFTStereoMonoBetaVLMFluxCore,
)
from models.vlm.qwen_encoder import QwenVisionConfig
from models.vlm.flux_confidence import FluxConfidenceConfig
from models.fusion.refine_vlm import StereoMonoVLMRefinementConfig
from training.dataloader import build_training_dataloader
from training.stages import (
    StagePreset,
    apply_training_stage,
    summarize_trainable_parameters,
)
from training.optim import OptimizerConfig, build_optimizer, build_scheduler
from training.checkpoint import save_checkpoint

from losses.confidence import sequence_l1_with_confidence_loss

from config.cfg_train import TrainConfig

# %%
train_config = TrainConfig()

loader = build_training_dataloader(
    root=train_config.root,
    batch_size=train_config.batch_size,
    num_workers=train_config.num_workers,
    crop_size=train_config.crop_size,
    use_mask_as_valid=train_config.use_mask_as_valid,
)


def build_model(backend: str) -> RAFTStereoMonoBetaVLMFluxCore:
    qwen_cfg = QwenVisionConfig(
        backend=backend, freeze_visual_backbone=(backend == "real_qwen2vl")
    )
    flux_cfg = FluxConfidenceConfig(qwen=qwen_cfg)
    refine_cfg = StereoMonoVLMRefinementConfig(flux_confidence=flux_cfg)
    model_cfg = RAFTStereoMonoBetaVLMFluxConfig(vlm_refinement=refine_cfg)
    return RAFTStereoMonoBetaVLMFluxCore(model_cfg)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
backend = "mock"
model = build_model(backend).to(device)
stage = StagePreset.by_name("vlm_adapters")
stage_report = apply_training_stage(model, stage)
summary = summarize_trainable_parameters(model)
print(f"Stage: {stage.name}")
print(f"LoRA targets: {len(stage_report['lora_targets'])}")
print(f"Trainable params: {summary['trainable']} / {summary['total']}")

optim_cfg = OptimizerConfig(
    lr=stage.base_lr,
    weight_decay=stage.weight_decay,
    warmup_steps=train_config.warmup_steps,
    total_steps=train_config.max_steps,
)

optimizer = build_optimizer(
    model,
    lr=optim_cfg.lr,
    weight_decay=optim_cfg.weight_decay,
    lr_multipliers=stage.lr_multipliers,
    betas=optim_cfg.betas,
    eps=optim_cfg.eps,
)

scheduler = build_scheduler(optimizer, optim_cfg)
scaler = torch.amp.GradScaler(device.type)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: TrainConfig,
    epoch: int = 0,
    start_step: int = 0,
    stage_name: str = "",
) -> tuple[int, dict[str, float]]:

    device = torch.device(config.device)

    model.train()

    metrics: dict[str, float] = {}
    step = start_step
    last_log = perf_counter()

    for batch in loader:

        if step >= config.max_steps:
            break

        # batch = next(iter(loader))
        left = batch["left"].to(device)
        right = batch["right"].to(device)
        target = batch["target_flow"].to(device)
        valid = batch["valid"].to(device)

        optimizer.zero_grad(set_to_none=True)
        # autocast_enabled = config.amp and device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
        ):
            outputs = model(left, right)
            loss, batch_metrics = sequence_l1_with_confidence_loss(
                outputs["disp_predictions"], target, valid, outputs["conf"]
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1
        metrics = {**batch_metrics}
        metrics["lr"] = optimizer.param_groups[0]["lr"]

        if step % config.log_every == 0:
            now = perf_counter()
            metrics["iter_time_sec"] = (now - last_log) / float(config.log_every)
            last_log = now

            print(
                f"[epoch {epoch:03d} step {step:06d}] "
                f"loss={metrics['loss']:.4f} flow={metrics['flow_loss']:.4f} "
                f"conf={metrics['confidence_loss']:.4f} "
                f"epe={metrics['epe']:.4f} lr={metrics['lr']:.2e}"
            )

        if step % config.save_every == 0:
            ckpt_path = Path(config.output_dir) / f"checkpoint_step_{step:06d}.pt"
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=step,
                stage_name=stage_name,
                metrics=metrics,
            )

    return step, metrics


step, metrics = train_one_epoch(
    model, loader, optimizer, scheduler, scaler, train_config
)
# %%
from utils.geometry import normalize_uint8_image, make_coords_grid
from utils.schedules import modulation_weight
from models.stereo.corr import CorrelationPyramid1D

from models.priors.beta_modulator import (
    BetaModulator,
    BetaModulatorConfig,
    BetaModulationOutput,
)
from models.stereo.upsample import upsample_flow

batch = next(iter(loader))
left = batch["left"].to(device)
right = batch["right"].to(device)
target = batch["target_flow"].to(device)
valid = batch["valid"].to(device)
left_norm = normalize_uint8_image(left)  # convert pix val from [0.0, 255.0] to [-1, 1]
right_norm = normalize_uint8_image(right)
fmap_left, fmap_right = model.feature_encoder([left_norm, right_norm])
model._validate_downsample_ratio(left, fmap_left)

mono = model.mono(left)
depth_lowres = mono.inverse_depth
mono_features = mono.penultimate
if depth_lowres.shape[-2:] != fmap_left.shape[-2:]:
    depth_lowres = F.interpolate(
        depth_lowres,
        size=fmap_left.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
if mono_features.shape[-2:] != fmap_left.shape[-2:]:
    mono_features = F.interpolate(
        mono_features,
        size=fmap_left.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

context_outputs = model.context_adapter(mono_features)
hidden_states, contexts = model._split_context_outputs(context_outputs)
context_gates = model.update_block.prepare_contexts(contexts)

batch, _, h_low, w_low = fmap_left.shape
coords0 = make_coords_grid(
    batch, h_low, w_low, device=fmap_left.device, dtype=fmap_left.dtype
)
coords1 = coords0.clone()

corr_pyramid = CorrelationPyramid1D(
    fmap_left,
    fmap_right,
    num_levels=model.config.corr_levels,
    radius=model.config.corr_radius,
)

disp_predictions: list[torch.Tensor] = []
modulation_predictions: list[torch.Tensor] = []
beta_out: BetaModulationOutput | None = None
up_mask = None

num_iters = model.config.iters
for iteration in range(num_iters):
    # iteration = 0
    coords1 = coords1.detach()
    corr = corr_pyramid.sample(coords1)
    flow = coords1 - coords0

    disp_lbp = model.lbp(flow[:, :1])
    depth_lbp = model.lbp(depth_lowres)
    beta_out = model.beta_modulator(disp_lbp, depth_lbp, return_distribution=True)
    modulation_predictions.append(beta_out.modulation)

    update_32 = (iteration % model.config.update_32_every) == 0
    update_16 = (iteration % model.config.update_16_every) == 0 or update_32
    hidden_states, up_mask, delta_flow = model.update_block(
        hidden_states,
        context_gates,
        corr,
        flow,
        update_8=True,
        update_16=update_16,
        update_32=update_32,
    )

    weight = modulation_weight(
        iteration,
        num_iters,
        mode=model.config.modulation_schedule,
        ratio=model.config.modulation_ratio,
    )
    delta_flow_x = delta_flow[:, :1] * (1.0 + beta_out.modulation * weight)
    delta_flow = torch.cat([delta_flow_x, torch.zeros_like(delta_flow[:, 1:2])], dim=1)
    coords1 = coords1 + delta_flow

    flow_lowres = coords1 - coords0
    fullres_flow = upsample_flow(
        flow_lowres,
        factor=model.config.downsample_factor,
        mask_logits=up_mask if model.config.use_convex_upsampling else None,
    )
    disp_predictions.append(fullres_flow[:, :1])

corr = corr_pyramid.sample(coords1.detach())
disp_positive = -(coords1 - coords0)[:, :1]
prompt = None
refine_out = model.refinement(
    left_norm,
    disp_positive,
    depth_lowres,
    hidden_states[0],
    corr,
    beta_distribution=beta_out.distribution,
    prompt=prompt,
)

# %% debugging model.refinement
prompt = prompt if prompt is not None else self.prompt_builder.build().full_text

refine_out = model.refinement(
    left_rgb=left_norm,
    disp=disp_positive,
    depth=depth_lowres,
    hidden=hidden_states[0],
    cost_volume=corr,
    beta_distribution=beta_out.distribution,
    prompt=prompt,
)
# model.refinement
stereo_disp = disp_positive
cost_volume = corr
mono_disp = depth_lowres
hidden = hidden_states[0]
beta_distribution = beta_out.distribution
confidence_logits, confidence, vlm_aux = model.refinement.confidence_head(
    left_rgb=left_norm,
    cost_volume=corr,
    stereo_disp=disp_positive,
    mono_disp=depth_lowres,
    hidden=hidden_states[0],
    beta_distribution=beta_out.distribution,
    prompt=prompt,
)

# model.refinement.confidence_head
qwen_out = model.refinement.confidence_head.qwen(
    left_norm,
    prompt=prompt,
)

qwen_spatial = qwen_out.spatial
if qwen_spatial.shape[-2:] != stereo_disp.shape[-2:]:
    qwen_spatial = F.interpolate(
        qwen_spatial,
        size=stereo_disp.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

latent_inputs = [cost_volume, stereo_disp, mono_disp]
if model.refinement.config.use_hidden_features and hidden is not None:
    latent_inputs.append(hidden)

if model.refinement.config.use_beta_statistics and beta_distribution is not None:
    latent_inputs.extend([beta_distribution.mean, beta_distribution.variance])

latent_inputs = torch.cat(latent_inputs, dim=1)
logits, confidence = model.refinement.confidence_head.decoder(
    latent_inputs, qwen_out.tokens, qwen_out.prompt_embedding
)

# model.refinement.confidence_head.decoder
qwen_tokens = qwen_out.tokens
qwen_prompt_embedding = qwen_out.prompt_embedding
latent = (
    model.refinement.confidence_head.decoder.latent_proj(latent_inputs)
    .flatten(2)
    .transpose(1, 2)
)  # [B, HW, D]

cond_prompt = qwen_prompt_embedding.unsqueeze(1)
cond = torch.cat(
    [
        model.refinement.confidence_head.decoder.cond_proj(qwen_tokens),
        model.refinement.confidence_head.decoder.cond_proj(cond_prompt),
    ],
    dim=1,
)

for block in model.refinement.confidence_head.decoder.blocks:
    latent = block(latent, cond)


latent = self.out_norm(latent)
latent_map = latent.transpose(1, 2).reshape(B, -1, H, W)
logits = self.out_head(latent_map)
confidence = torch.sigmoid(logits)


# %%

depth_registered_neg_up = upsample_flow(
    -refine_out.depth_registered,
    factor=self.config.downsample_factor,
    mask_logits=(refine_out.up_mask if self.config.use_convex_upsampling else None),
)
fused_neg_up = upsample_flow(
    -refine_out.fused_disp,
    factor=self.config.downsample_factor,
    mask_logits=(refine_out.up_mask if self.config.use_convex_upsampling else None),
)
disp_predictions.append(depth_registered_neg_up)
disp_predictions.append(fused_neg_up)
