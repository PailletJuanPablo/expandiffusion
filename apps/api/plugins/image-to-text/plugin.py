"""Image-to-text plugin action."""

from __future__ import annotations

import threading
from typing import Any

from expandiffusion import constants
from expandiffusion.errors import AppError
from expandiffusion.plugin_actions import PluginAction, PluginActionContext, PluginTool
from expandiffusion.schemas import PluginActionResult

ACTION_ID = "image-to-text"
TOOL_ID = "image-to-text"
CAPTION_MODEL_NAME = "blip-large"
CLIP_MODEL_NAME = "ViT-L-14/openai"
CAPTION_MAX_LENGTH = 64
MIN_FLAVORS = 8
MAX_FLAVORS = 32

_INTERROGATOR_LOCK = threading.Lock()
_INTERROGATOR: Any | None = None


class ImageToTextAction(PluginAction):
    """Generate a Stable Diffusion prompt from the selected image block."""

    id = ACTION_ID
    label = "Describe selection"
    description = "Runs CLIP Interrogator over the selected block."

    def run(self, context: PluginActionContext) -> PluginActionResult:
        interrogator = _load_interrogator()
        image = context.image.convert("RGB")
        text = interrogator.interrogate(
            image,
            min_flavors=MIN_FLAVORS,
            max_flavors=MAX_FLAVORS,
        ).strip()
        if not text:
            raise AppError(
                constants.ERROR_PLUGIN_ACTION_FAILED,
                "CLIP Interrogator did not return any text.",
                status_code=500,
            )
        return PluginActionResult(
            action_id=self.id,
            text=text,
            data={
                "engine": "clip-interrogator",
                "caption_model_name": CAPTION_MODEL_NAME,
                "clip_model_name": CLIP_MODEL_NAME,
                "device": getattr(interrogator, "device", "unknown"),
                "target": context.target,
            },
        )


def register(context) -> None:
    action = ImageToTextAction()
    context.register_action(action)
    context.register_tool(
        PluginTool(
            id=TOOL_ID,
            label="Image to Text",
            description="Select an image block and generate a Stable Diffusion prompt.",
            action_id=action.id,
            icon="captions",
            icon_color="#4f46e5",
            accent_color="#4f46e5",
            result_label="Image description",
            controls=action.controls(),
            default_values=action.defaults(),
        )
    )


def _load_interrogator() -> Any:
    try:
        from clip_interrogator import Config, Interrogator
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            (
                "Image to Text requires CLIP Interrogator. "
                "Install the API diffusers extra."
            ),
            status_code=500,
        ) from exc

    global _INTERROGATOR
    with _INTERROGATOR_LOCK:
        if _INTERROGATOR is None:
            _INTERROGATOR = Interrogator(
                Config(
                    caption_model_name=CAPTION_MODEL_NAME,
                    caption_max_length=CAPTION_MAX_LENGTH,
                    clip_model_name=CLIP_MODEL_NAME,
                    cache_path=str(constants.DEFAULT_DATA_DIR / "clip-interrogator"),
                    download_cache=True,
                    quiet=True,
                )
            )
        return _INTERROGATOR
