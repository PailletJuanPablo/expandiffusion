"""Semantic object selection plugin action."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from expandiffusion import constants
from expandiffusion.errors import AppError
from expandiffusion.image_utils import decode_data_url, encode_png_data_url, normalize_mask
from expandiffusion.plugin_actions import PluginAction, PluginActionContext, PluginTool
from expandiffusion.schemas import ControlSchema, PluginActionResult

ACTION_ID = "object-selector"
TOOL_ID = "object-selector"

PARAM_PROMPT = "object_selector_prompt"
PARAM_MASK_EXPAND = "object_selector_mask_expand"
PARAM_MASK_BLUR = "object_selector_mask_blur"

FLORENCE_MODEL_ID = "florence-community/Florence-2-base-ft"
FLORENCE_TASK = "<OPEN_VOCABULARY_DETECTION>"
SAM_MODEL_ID = "facebook/sam-vit-base"
WHITE_THRESHOLD = 128
DEFAULT_MASK_EXPAND = 4
DEFAULT_MASK_BLUR = 2

_FLORENCE_LOCK = threading.Lock()
_FLORENCE: tuple[Any, Any] | None = None
_SAM_LOCK = threading.Lock()
_SAM: tuple[Any, Any] | None = None


class ObjectSelectorAction(PluginAction):
    """Detect an object and return an editable inpaint mask."""

    id = ACTION_ID
    label = "Select object"
    description = "Builds an object mask from a click, a prompt, or both."

    def controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_PROMPT,
                label="Object prompt",
                kind=constants.CONTROL_TEXT,
                section=constants.CONTROL_SECTION_BASIC,
                default_value="",
                placeholder="person, chair, logo...",
            ),
            ControlSchema(
                id=PARAM_MASK_EXPAND,
                label="Mask expand",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=DEFAULT_MASK_EXPAND,
                min=0,
                max=64,
                step=1,
            ),
            ControlSchema(
                id=PARAM_MASK_BLUR,
                label="Mask blur",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=DEFAULT_MASK_BLUR,
                min=0,
                max=32,
                step=1,
            ),
        ]

    def run(self, context: PluginActionContext) -> PluginActionResult:
        image = context.image.convert("RGB")
        prompt = str(context.control(PARAM_PROMPT, "") or "").strip()
        point = _target_point(context.target, image.size)
        if not prompt and point is None:
            raise AppError(
                constants.ERROR_PLUGIN_ACTION_FAILED,
                "Click the object or enter an object prompt before selecting.",
                status_code=400,
            )

        boxes = _detect_prompt_boxes(image, prompt) if prompt else []
        if prompt and not boxes and point is None:
            raise AppError(
                constants.ERROR_PLUGIN_ACTION_FAILED,
                f"Object selector did not detect '{prompt}'.",
                status_code=404,
            )

        selected_box = _select_box(boxes, point)
        mask, confidence = _segment_mask(image, selected_box, point)
        mask = _postprocess_mask(
            mask,
            _visible_mask(context.target, image.size),
            _int_control(context, PARAM_MASK_EXPAND, DEFAULT_MASK_EXPAND, 0, 64),
            _int_control(context, PARAM_MASK_BLUR, DEFAULT_MASK_BLUR, 0, 32),
        )
        if mask.getbbox() is None:
            raise AppError(
                constants.ERROR_PLUGIN_ACTION_FAILED,
                "Object selector mask is outside visible canvas content.",
                status_code=404,
            )

        ordered_boxes = _selected_box_first(boxes, selected_box)
        return PluginActionResult(
            action_id=self.id,
            text="Object mask ready.",
            mask=encode_png_data_url(mask),
            data={
                "boxes": [_box_report(box) for box in ordered_boxes],
                "confidence": confidence,
                "source": _selection_source(prompt, point),
                "target": context.target,
            },
        )


def register(context) -> None:
    action = ObjectSelectorAction()
    context.register_action(action)
    context.register_tool(
        PluginTool(
            id=TOOL_ID,
            label="Object Selector",
            description="Click visible canvas content or enter a prompt to create an object mask.",
            action_id=action.id,
            icon="wand-sparkles",
            icon_color="#0f766e",
            accent_color="#0f766e",
            result_label="Object mask",
            target=constants.PLUGIN_TOOL_TARGET_CANVAS,
            controls=action.controls(),
            default_values=action.defaults(),
        )
    )


def _detect_prompt_boxes(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            "Object selector requires PyTorch.",
            status_code=500,
        ) from exc

    processor, detector = _load_florence_detector()
    inputs = processor(text=FLORENCE_TASK + prompt, images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = detector.generate(**inputs, max_new_tokens=512, num_beams=3)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=FLORENCE_TASK,
        image_size=image.size,
    )[FLORENCE_TASK]
    labels = parsed.get("labels", [])
    boxes = []
    for index, box in enumerate(parsed.get("bboxes", [])):
        boxes.append(
            {
                "box": _clamp_box(tuple(int(round(value)) for value in box), image.size),
                "label": labels[index] if index < len(labels) else prompt,
                "confidence": None,
            }
        )
    return [box for box in boxes if _box_area(_box_tuple(box)) > 0]


def _segment_mask(
    image: Image.Image,
    box: tuple[int, int, int, int] | None,
    point: tuple[int, int] | None,
) -> tuple[Image.Image, float | None]:
    try:
        import torch
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            "Object selector requires PyTorch.",
            status_code=500,
        ) from exc

    processor, model = _load_sam()
    kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if point is not None:
        kwargs["input_points"] = [[[[point[0], point[1]]]]]
        kwargs["input_labels"] = [[[1]]]
    if box is not None:
        kwargs["input_boxes"] = [[list(box)]]
    inputs = processor(image, **kwargs)
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores.detach().cpu()
    if scores.ndim > 2:
        scores = scores[0, 0]
    elif scores.ndim > 1:
        scores = scores[0]
    index = int(scores.argmax().item()) if scores.numel() else 0
    selected = masks
    while selected.ndim > 3:
        selected = selected[0]
    if selected.ndim == 3:
        selected = selected[min(index, selected.shape[0] - 1)]
    array = selected.numpy().astype(np.uint8) * 255
    return Image.fromarray(array, mode="L"), float(scores[index].item()) if scores.numel() else None


def _load_florence_detector() -> tuple[Any, Any]:
    try:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            (
                "Object selector requires transformers with Florence-2 support. "
                "Install the API diffusers extra."
            ),
            status_code=500,
        ) from exc

    global _FLORENCE
    with _FLORENCE_LOCK:
        if _FLORENCE is None:
            processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID)
            detector = Florence2ForConditionalGeneration.from_pretrained(FLORENCE_MODEL_ID)
            detector.to("cpu")
            detector.eval()
            _FLORENCE = (processor, detector)
        return _FLORENCE


def _load_sam() -> tuple[Any, Any]:
    try:
        from transformers import SamModel, SamProcessor
    except ImportError as exc:
        raise AppError(
            constants.ERROR_UNSUPPORTED_OPERATION,
            "Object selector requires transformers with SAM support.",
            status_code=500,
        ) from exc

    global _SAM
    with _SAM_LOCK:
        if _SAM is None:
            processor = SamProcessor.from_pretrained(SAM_MODEL_ID)
            model = SamModel.from_pretrained(SAM_MODEL_ID)
            model.to("cpu")
            model.eval()
            _SAM = (processor, model)
        return _SAM


def _postprocess_mask(
    mask: Image.Image,
    visible_mask: Image.Image,
    expand: int,
    blur: int,
) -> Image.Image:
    processed = normalize_mask(mask, visible_mask.size).point(
        lambda pixel: 255 if pixel >= WHITE_THRESHOLD else 0
    )
    if expand > 0:
        kernel = max(3, expand * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        processed = processed.filter(ImageFilter.MaxFilter(kernel))
    if blur > 0:
        processed = processed.filter(ImageFilter.GaussianBlur(radius=blur))
    return ImageChops.multiply(processed, visible_mask)


def _visible_mask(target: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    visible_mask = target.get("visible_mask")
    if isinstance(visible_mask, str) and visible_mask:
        return normalize_mask(decode_data_url(visible_mask), size).point(
            lambda pixel: 255 if pixel >= WHITE_THRESHOLD else 0
        )
    return Image.new("L", size, 255)


def _target_point(
    target: dict[str, Any],
    size: tuple[int, int],
) -> tuple[int, int] | None:
    point = target.get("point")
    if not isinstance(point, dict):
        return None
    x = _number(point.get("x"))
    y = _number(point.get("y"))
    if x is None or y is None:
        return None
    return (
        max(0, min(size[0] - 1, int(round(x)))),
        max(0, min(size[1] - 1, int(round(y)))),
    )


def _select_box(
    boxes: list[dict[str, Any]],
    point: tuple[int, int] | None,
) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    if point is None:
        return _box_tuple(boxes[0])
    return _box_tuple(
        min(
            boxes,
            key=lambda box: (
                0 if _box_contains(_box_tuple(box), point) else 1,
                _box_point_distance(_box_tuple(box), point),
                -_confidence(box),
            ),
        )
    )


def _selected_box_first(
    boxes: list[dict[str, Any]],
    selected_box: tuple[int, int, int, int] | None,
) -> list[dict[str, Any]]:
    if selected_box is None:
        return boxes
    return sorted(boxes, key=lambda box: 0 if _box_tuple(box) == selected_box else 1)


def _box_report(box: dict[str, Any]) -> dict[str, Any]:
    return {
        "box": list(_box_tuple(box)),
        "label": str(box.get("label", "")),
        "confidence": box.get("confidence"),
    }


def _selection_source(prompt: str, point: tuple[int, int] | None) -> str:
    if prompt and point is not None:
        return "prompt_and_click"
    if prompt:
        return "prompt"
    return "click"


def _box_tuple(box: dict[str, Any]) -> tuple[int, int, int, int]:
    raw_box = box.get("box") or box.get("bbox") or (0, 0, 0, 0)
    return tuple(int(round(value)) for value in raw_box[:4])


def _clamp_box(
    box: tuple[int, int, int, int],
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        max(0, min(size[0], left)),
        max(0, min(size[1], top)),
        max(0, min(size[0], right)),
        max(0, min(size[1], bottom)),
    )


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _box_contains(box: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _box_point_distance(box: tuple[int, int, int, int], point: tuple[int, int]) -> float:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return ((center_x - point[0]) ** 2 + (center_y - point[1]) ** 2) ** 0.5


def _confidence(box: dict[str, Any]) -> float:
    value = box.get("confidence")
    return value if isinstance(value, int | float) else 0.0


def _int_control(
    context: PluginActionContext,
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = _number(context.control(key, default))
    if value is None:
        return default
    return max(minimum, min(maximum, int(round(value))))


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and np.isfinite(value):
        return float(value)
    return None
