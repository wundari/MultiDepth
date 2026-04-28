# %%
from __future__ import annotations

from config.cfg import PromptBundle


# %%


class ReflectiveObjectPromptBuilder:
    """
    Build the default prompt used by the Qwen2VL * flux confidence branch
    """

    def __init__(self) -> None:

        self._system = (
            "You are a vision-language assistant for stereo depth fusion. "
            "Your job is to identify regions where stereo matching may be "
            "unreliable because of transparency, reflectance, specular "
            "highlights, or other illusion-like cues."
        )

        self._user = (
            "Using the input image, identify transparent or reflective objects "
            "such as mirrors, glass, windows, showcases, glossy surfaces, and "
            "similar regions. Return a confidence-oriented binary mask where "
            "suspicious regions are white (255) and all other regions are black (0). "
            "If your reason about corner coordinates internally, use them only "
            "to improve the mask."
        )

    def build(self, extra_context: str | None = None) -> PromptBundle:
        user = (
            self._user
            if not extra_context
            else f"{self._user}\nAdditional context: {extra_context}"
        )

        return PromptBundle(system=self._system, user=user)
