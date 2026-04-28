# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float

from config.cfg import UpdateBlockConfig, MultiScaleUpdateConfig


# %%
class ConvGRUCell(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int = 3) -> None:
        """
        A regular GRU maintains a hidden state h and updates it given new input x using two gates:

        Update gate z — how much of the old hidden state to keep vs replace
        Reset gate r — how much of the old hidden state to expose when computing the candidate

        z = σ(W_z · [h, x])          # how much to update
        r = σ(W_r · [h, x])          # how much to reset
        q = tanh(W_q · [r*h, x])     # candidate new state
        h_new = (1 - z) * h + z * q  # blend old and candidate

        This class does exactly the same thing,
        but W_z, W_r, W_q are Conv2d layers instead of linear layers,
        so the hidden state is a spatial feature map (B, C, H, W) rather than a vector.

        Args:
            hidden_dim (int): dimension of the GRU hidden state
            input_dim (int): dimension of the input feature map
            kernel_size (int, optional): size of the convolutional kernels. Defaults to 3.
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv_z = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )  # how much to update
        self.conv_r = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )  # how much to reset
        self.conv_q = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )  # new state candidate

        # update weights
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, hidden: Float[Tensor, "B C H W"], x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C H W"]:
        hx = torch.cat([hidden, x], dim=1)
        z = torch.sigmoid(self.conv_z(hx))  # how much to update
        r = torch.sigmoid(self.conv_r(hx))  # how much to reset
        q = torch.tanh(
            self.conv_q(torch.cat([r * hidden, x], dim=1))
        )  # new state candidate
        return (1.0 - z) * hidden + z * q


class ContextGateProjector(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        """
        This class takes a context feature map and projects it into three
        separate tensors that will be injected into a GRU as biases — one for each gate.

        What it does
        A single Conv2d projects context from context_dim channels to 3 * hidden_dim channels,
        then torch.chunk(..., 3, dim=1) splits that along the channel dimension into
        three equal pieces.

        context (B, context_dim, H, W)
            └─► proj Conv2d ──► (B, 3*hidden_dim, H, W)
                                    │
                            chunk into 3
                            ┌─────────────────┐
                            ▼        ▼        ▼
                    (B,H,H,W) (B,H,H,W) (B,H,H,W)
                        inp_z    inp_r    inp_q

        Why this exists
        In standard GRU refinement, the hidden state and input drive the gates.
        But you also have a context feature map (from the left image encoder)
        that contains scene structure — edges, textures, surface normals implicitly — that
        shouldn't be updated each iteration, it's static per-image information.

        Args:
            context_dim (int): dimension of the context feature map coming from the encoder
            hidden_dim (int): dimension of the GRU hidden state
        """

        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=context_dim,
            out_channels=3 * hidden_dim,
            kernel_size=3,
            padding=1,
        )

        # initialize weight
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, context: Float[Tensor, "B C H W"]) -> tuple[
        Float[Tensor, "B hidden_dim H W"],
        Float[Tensor, "B hidden_dim H W"],
        Float[Tensor, "B hidden_dim H W"],
    ]:

        return torch.chunk(self.proj(context), 3, dim=1)


class ContextGRUCell(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int = 3) -> None:
        """
        This is the complete picture — ContextGRUCell is ConvGRUCell from before,
        but with the context gate biases (gz, gr, gq) injected from
        ContextGateProjector.

        The three classes form a single system:
        ContextGateProjector          ContextGRUCell
        ─────────────────────         ──────────────────────────────────────
        context_feat                  hidden, x, (gz, gr, gq)
            │                              │         │
            └──► [gz, gr, gq]  ────────────┘         │
                (computed once,      added as bias   │
                before the loop)    to each gate ───┘

        Why additive bias injection works well here
        The gates are linear transforms followed by a nonlinearity.
        Adding gz before the sigmoid is mathematically equivalent to having a
        spatially-varying bias that shifts the gate's operating point:

        sigmoid(conv_z(hx) + gz)

        If gz is large and positive at some pixel,
        the update gate is pushed toward 1 → "replace hidden state here".
        If large and negative, pushed toward 0 → "preserve hidden state here".
        The context feature map is essentially telling the GRU where to be
        aggressive vs conservative about updating, based on
        static image structure like edges or illusion regions —
        without consuming any compute inside the loop.

        Args:
            hidden_dim (int): _description_
            input_dim (int): _description_
            kernel_size (int, optional): _description_. Defaults to 3.
        """

        super().__init__()
        padding = kernel_size // 2
        self.conv_z = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )  # how much to update
        self.conv_r = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )  # how much to reset
        self.conv_q = nn.Conv2d(
            in_channels=hidden_dim + input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )  # new state candidate

        # initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden: Float[Tensor, "B C H W"],
        x: Float[Tensor, "B C H W"],
        gates: tuple[
            Float[Tensor, "B hidden_dim H W"],
            Float[Tensor, "B hidden_dim H W"],
            Float[Tensor, "B hidden_dim H W"],
        ],
    ):

        gz, gr, gq = gates
        hx = torch.cat([hidden, x], dim=1)
        z = torch.sigmoid(self.conv_z(hx) + gz)
        r = torch.sigmoid(self.conv_r(hx) + gr)
        q = torch.tanh(self.conv_q(torch.cat([r * hidden, x], dim=1)) + gq)

        return (1.0 - z) * hidden + z * q


class StereoMotionEncoder(nn.Module):
    """
    StereoMotionEncoder compresses two sources of information —
    the correlation/cost volume lookup and the current disparity estimate —
    into a single compact motion feature vector that feeds into the GRU as the input x.

    corr (B, corr_channels, H, W)  ──► corr_tower (64ch) ──┐
                                                            ├──► cat (96ch) ──► fuse [B, motion_dim, H, W]
    flow (B, 2, H, W)              ──► flow_tower (32ch) ──┘

    It sits between the cost volume and the GRU cell — every iteration, you look up the cost volume around the current disparity estimate, encode both with this module, and pass the result as x to ContextGRUCell.

    The two towers:
    corr_tower — compresses the cost volume lookup.
                The 1×1 conv first does a cheap channel reduction (like a learned PCA),
                then the 3×3 conv mixes spatial neighbourhood information.
                Starting with 1×1 is intentional: the correlation channels encode
                similarity at different disparity offsets, and the 1×1 learns which
                offsets matter before doing any spatial mixing.

    flow_tower — encodes the current disparity. The input has 2 channels:
                typically [disp, disp / max_disp] — the raw value and a normalised version,
                giving the network both absolute and relative scale. The 7×7 kernel on
                the first conv gives a wide receptive field, which helps because disparity
                is spatially smooth and context from neighbouring pixels is informative.

    fuse — concatenates both (64 + 32 = 96 channels) and projects to motion_dim,
            the exact size expected by the GRU input.

    Args:
        config (UpdateBlockConfig | MultiScaleUpdateConfig): _description_
    """

    def __init__(self, config: UpdateBlockConfig | MultiScaleUpdateConfig) -> None:

        super().__init__()
        corr_out = 64
        flow_out = 32
        self.corr_tower = nn.Sequential(
            nn.Conv2d(
                in_channels=config.corr_channels, out_channels=corr_out, kernel_size=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=corr_out, out_channels=corr_out, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.flow_tower = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=flow_out, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=flow_out, out_channels=flow_out, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=corr_out + flow_out,
                out_channels=config.motion_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, flow: Float[Tensor, "B C H W"], corr: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C H W"]:
        return self.fuse(
            torch.cat([self.corr_tower(corr), self.flow_tower(flow)], dim=1)
        )


class DeltaFlowHead(nn.Module):
    """
    a small two-layer conv network that reads the GRU hidden state and
    predicts how much to update the current disparity estimate at each pixel.

    hidden (B, hidden_dim, H, W)
        └──► Conv2d 3×3 ──► ReLU ──► Conv2d 3×3 ──► delta (B, 2, H, W)
            (→ 256ch)                (→ 2ch)

    The output is a residual — it gets added to the current disparity each iteration,
    not used as an absolute prediction.
    Over 12 iterations these small deltas accumulate into the final estimate.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 256, output_dim: int = 2
    ) -> None:

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=output_dim,
                kernel_size=3,
                padding=1,
            ),
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        modules = list(self.modules())
        for i, module in enumerate(modules):
            if isinstance(module, nn.Conv2d):
                is_last = i == len(modules) - 1
                if is_last:
                    # Small init keeps early delta updates near zero,
                    # letting the GRU stabilise before making large moves
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.kaiming_normal_(
                        module.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:

        return self.net(x)


class StereoUpdateBlock(nn.Module):
    """
    StereoUpdateBlock is the complete GRU iteration step —
    the thing you call 12 times in the refinement loop.
    Each call takes the current state of the world (hidden state, context,
    cost lookup, current disparity) and returns an updated hidden state,
    a disparity delta, and an upsampling mask.

    corr, flow ──► motion_encoder ──► motion (B, motion_dim, H, W)
                                            │
    context ────────────────────────────────┤
                                            ▼
    hidden ─────────────────────────► ConvGRUCell ──► new hidden
                                                         │
                                ┌────────────────────────┤
                                ▼                        ▼
                            delta_head               mask_head
                                │                        │
                                ▼                        ▼
                            delta_flow              upsampling mask
                        (zeroed y-channel)

    motion_encoder: compresses the cost volume lookup + current disparity into motion_dim channels,
                    as you already know from the previous class.
    gru: updates the hidden state given [context, motion] as input.
         Context carries static image structure (edges, surfaces),
         motion carries the dynamic iteration signal
         (where we are in disparity space, what the cost volume says).
    delta_head: predicts the disparity update to add to the current estimate.
                Outputs 2 channels even though stereo only moves along the x-axis — see the bug below.
    mask_head: predicts a soft upsampling mask for converting the low-resolution (H/4, W/4)
                disparity back to full resolution.
                The 9 * upsample_factor * 2 output encodes a 3×3 neighbourhood weight for
                each full-resolution pixel, learned rather than bilinear.
                The 0.25 scaling keeps logits in a reasonable range before softmax.

    """

    def __init__(self, config: UpdateBlockConfig) -> None:

        super().__init__()
        self.config = config
        self.motion_encoder = StereoMotionEncoder(config)
        self.gru = ConvGRUCell(
            config.hidden_dim, config.context_dim + config.motion_dim
        )
        self.delta_head = DeltaFlowHead(config.hidden_dim, hidden_dim=256, output_dim=2)
        self.mask_head = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_dim,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=9 * (config.upsample_factor**2),
                kernel_size=1,
            ),
        )

    def forward(
        self,
        hidden: Float[Tensor, "B C H W"],
        context: Float[Tensor, "B C H W"],
        corr: Float[Tensor, "B C H W"],
        flow: Float[Tensor, "B C H W"],
    ) -> tuple[
        Float[Tensor, "B C H W"], Float[Tensor, "B C H W"], Float[Tensor, "B C H W"]
    ]:

        motion = self.motion_encoder(flow, corr)
        hidden = self.gru(hidden, torch.cat([context, motion], dim=1))
        delta_flow = self.delta_head(hidden)

        # Zero y-component: rectified stereo only has horizontal disparity.
        # Multiplicative mask avoids in-place autograd issues.
        # delta_flow[:, 1] = 0.0
        delta_flow = delta_flow * torch.tensor(
            [1.0, 0.0], device=delta_flow.device
        ).view(1, 2, 1, 1)
        mask = 0.25 * self.mask_head(hidden)

        return hidden, mask, delta_flow


class MultiScaleStereoUpdateBlock(nn.Module):
    """
     This is the multi-scale upgrade of StereoUpdateBlock.
     Instead of one GRU operating at 1/8 resolution,
     it runs three GRUs at three scales simultaneously (1/8, 1/16, 1/32),
     passing information both bottom-up and top-down each iteration.
     The intuition is that the coarser scales see larger context and can resolve
     ambiguities (like illusion regions) that the fine scale alone cannot.

             motion (from corr+flow, at 1/8)
                          │
         ┌────────────────┼─────────────────────┐
         ▼                ▼                     ▼
    scale 1/32       scale 1/16            scale 1/8
    (h32, 64ch)     (h16, 96ch)           (h8, 128ch)
         │                │                     │
    gru32 ──────►    gru16 ──────────────► gru8
    (updated        (gets h32 up +         (gets motion +
     from h16 dn)    h8 dn)                 h16 up)
                                                │
                                          delta_head / mask_head

    The three GRUs update in a specific sequence: 32 → 16 → 8, coarse to fine.
    This is deliberate — each scale conditions on the freshly updated coarser scale above it.

    gru32 gets in32(avg_pool(h16)) — a downsampled summary of the current fine-scale state.
        It operates at the coarsest level with the widest receptive field.

    gru16 gets in16(cat(upsample(h32), avg_pool(h8))) — the freshly updated coarse state coming down,
        plus a downsampled view of the fine state coming up. It bridges both directions.

    gru8 gets in8(cat(motion, upsample(h16))) — the motion encoding (cost volume + disparity) plus
        guidance from the mid-scale. This is the only scale that directly sees the cost volume signal.
        The disparity delta and upsampling mask are read from h8 only.

    """

    def __init__(self, config: MultiScaleUpdateConfig) -> None:

        super().__init__()

        self.config = config
        h8, h16, h32 = config.hidden_dims  # (128, 96, 64)
        c8, c16, c32 = config.context_dims  # (128, 96, 64)
        self.motion_encoder = StereoMotionEncoder(config)
        self.context_pre8 = ContextGateProjector(c8, h8)
        self.context_pre16 = ContextGateProjector(c16, h16)
        self.context_pre32 = ContextGateProjector(c32, h32)
        self.in8 = nn.Sequential(
            nn.Conv2d(
                in_channels=config.motion_dim + h16,
                out_channels=h8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.in16 = nn.Sequential(
            nn.Conv2d(
                in_channels=h32 + h8,
                out_channels=h16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        self.in32 = nn.Sequential(
            nn.Conv2d(
                in_channels=h16,
                out_channels=h32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.gru8 = ContextGRUCell(hidden_dim=h8, input_dim=h8)
        self.gru16 = ContextGRUCell(hidden_dim=h16, input_dim=h16)
        self.gru32 = ContextGRUCell(hidden_dim=h32, input_dim=h32)
        self.delta_head = DeltaFlowHead(input_dim=h8, hidden_dim=256, output_dim=2)
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels=h8, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=9 * (config.upsample_factor**2),
                kernel_size=1,
            ),
        )

    def prepare_contexts(
        self,
        contexts: tuple[
            Float[Tensor, "B C H W"],
            Float[Tensor, "B C H W"],
            Float[Tensor, "B C H W"],
        ],
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor],  # gates for gru8  (gz, gr, gq)
        tuple[Tensor, Tensor, Tensor],  # gates for gru16
        tuple[Tensor, Tensor, Tensor],  # gates for gru32
    ]:

        c8, c16, c32 = contexts

        return (self.context_pre8(c8), self.context_pre16(c16), self.context_pre32(c32))

    def forward(
        self,
        hidden_states: tuple[
            Float[Tensor, "B C H W"],
            Float[Tensor, "B C H W"],
            Float[Tensor, "B C H W"],
        ],
        context_gates: tuple[
            Float[Tensor, "B C H W"],
            Float[Tensor, "B C H W"],
            Float[Tensor, "B C H W"],
        ],
        corr: Float[Tensor, "B C H W"],
        flow: Float[Tensor, "B C H W"],
        update_8: bool = True,
        update_16: bool = True,
        update_32: bool = True,
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor],
        Float[Tensor, "B C H W"],
        Float[Tensor, "B C H W"],
    ]:

        h8, h16, h32 = hidden_states
        g8, g16, g32 = context_gates

        motion = self.motion_encoder(flow, corr)
        if update_32:
            h32 = self.gru32(
                h32, self.in32(F.avg_pool2d(h16, kernel_size=2, stride=2)), g32
            )

        if update_16:
            x16 = torch.cat(
                [
                    F.interpolate(
                        h32, size=h16.shape[-2:], mode="bilinear", align_corners=True
                    ),
                    F.avg_pool2d(h8, kernel_size=2, stride=2),
                ],
                dim=1,
            )
            h16 = self.gru16(h16, self.in16(x16), g16)
        if update_8:
            x8 = torch.cat(
                [
                    motion,
                    F.interpolate(
                        h16, size=h8.shape[-2:], mode="bilinear", align_corners=True
                    ),
                ],
                dim=1,
            )
            h8 = self.gru8(h8, self.in8(x8), g8)

        delta_flow = self.delta_head(h8)
        delta_flow[:, 1] = 0.0
        mask = 0.25 * self.mask_head(h8)

        return (h8, h16, h32), mask, delta_flow
