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
        points = _target_points(context.target, image.size)
        if not prompt and not points:
            raise AppError(
                constants.ERROR_PLUGIN_ACTION_FAILED,
                "Click the object or enter an object prompt before selecting.",
                status_code=400,
            )

        boxes = _detect_prompt_boxes(image, prompt) if prompt else []
        if prompt and not boxes and not points:
            raise AppError(
                constants.ERROR_PLUGIN_ACTION_FAILED,
                f"Object selector did not detect '{prompt}'.",
                status_code=404,
            )

        visible_mask = _visible_mask(context.target, image.size)
        expand = _int_control(context, PARAM_MASK_EXPAND, DEFAULT_MASK_EXPAND, 0, 64)
        blur = _int_control(context, PARAM_MASK_BLUR, DEFAULT_MASK_BLUR, 0, 32)
        selections = (
            _segment_all_boxes(image, boxes, visible_mask, expand, blur)
            if boxes and not points
            else [_segment_selected_object(image, boxes, points, visible_mask, expand, blur)]
        )
        selections = [
            selection for selection in selections if selection["mask"].getbbox() is not None
        ]
        if selections:
            mask = _merge_masks([selection["mask"] for selection in selections])
        else:
            mask = Image.new("L", image.size, 0)
        confidence = _selection_confidence(selections)
        if mask.getbbox() is None:
            raise AppError(
                constants.ERROR_PLUGIN_ACTION_FAILED,
                "Object selector mask is outside visible canvas content.",
                status_code=404,
            )

        selected_box = selections[0]["box"] if len(selections) == 1 else None
        ordered_boxes = _selected_box_first(boxes, selected_box)
        return PluginActionResult(
            action_id=self.id,
            text="Object mask ready.",
            mask=encode_png_data_url(mask),
            data={
                "boxes": [_box_report(box) for box in ordered_boxes],
                "confidence": confidence,
                "source": _selection_source(prompt, points),
                "selections": [_selection_report(selection) for selection in selections],
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
    points: list[tuple[int, int]],
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
    if points:
        kwargs["input_points"] = [[[list(point) for point in points]]]
        kwargs["input_labels"] = [[[1 for _point in points]]]
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


def _segment_all_boxes(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    visible_mask: Image.Image,
    expand: int,
    blur: int,
) -> list[dict[str, Any]]:
    selections = []
    for index, box in enumerate(boxes):
        box_tuple = _box_tuple(box)
        mask, confidence = _segment_mask(image, box_tuple, [])
        selections.append(
            {
                "id": f"object-{index + 1}",
                "box": box_tuple,
                "label": str(box.get("label", "")),
                "confidence": confidence,
                "mask": _postprocess_mask(mask, visible_mask, expand, blur),
            }
        )
    return selections


def _segment_selected_object(
    image: Image.Image,
    boxes: list[dict[str, Any]],
    points: list[tuple[int, int]],
    visible_mask: Image.Image,
    expand: int,
    blur: int,
) -> dict[str, Any]:
    selected_box = _select_box(boxes, points)
    mask, confidence = _segment_mask(image, selected_box, points)
    source_box = _box_for_tuple(boxes, selected_box)
    return {
        "id": f"object-{_box_index(boxes, selected_box) + 1}",
        "box": selected_box,
        "label": str(source_box.get("label", "")) if source_box else "",
        "confidence": confidence,
        "mask": _postprocess_mask(mask, visible_mask, expand, blur),
    }


def _merge_masks(masks: list[Image.Image]) -> Image.Image:
    merged = Image.new("L", masks[0].size, 0)
    for mask in masks:
        merged = ImageChops.lighter(merged, mask)
    return merged


def _selection_confidence(selections: list[dict[str, Any]]) -> float | None:
    confidences = [
        selection["confidence"]
        for selection in selections
        if isinstance(selection.get("confidence"), int | float)
    ]
    return max(confidences) if confidences else None


def _selection_report(selection: dict[str, Any]) -> dict[str, Any]:
    box = selection.get("box")
    return {
        "id": selection["id"],
        "box": list(box) if box else None,
        "label": selection["label"],
        "confidence": selection["confidence"],
        "mask": encode_png_data_url(selection["mask"]),
    }


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


def _target_points(
    target: dict[str, Any],
    size: tuple[int, int],
) -> list[tuple[int, int]]:
    points = target.get("points")
    if isinstance(points, list):
        parsed_points = [
            point for point in (_parse_target_point(item, size) for item in points) if point
        ]
        if parsed_points:
            return parsed_points
    point = target.get("point")
    parsed_point = _parse_target_point(point, size)
    return [parsed_point] if parsed_point else []


def _parse_target_point(
    point: Any,
    size: tuple[int, int],
) -> tuple[int, int] | None:
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
    points: list[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    if not points:
        return _box_tuple(boxes[0])
    return _box_tuple(
        min(
            boxes,
            key=lambda box: (
                0 if _box_contains_any_point(_box_tuple(box), points) else 1,
                _box_points_distance(_box_tuple(box), points),
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


def _box_for_tuple(
    boxes: list[dict[str, Any]],
    selected_box: tuple[int, int, int, int] | None,
) -> dict[str, Any] | None:
    if selected_box is None:
        return None
    for box in boxes:
        if _box_tuple(box) == selected_box:
            return box
    return None


def _box_index(
    boxes: list[dict[str, Any]],
    selected_box: tuple[int, int, int, int] | None,
) -> int:
    if selected_box is None:
        return 0
    for index, box in enumerate(boxes):
        if _box_tuple(box) == selected_box:
            return index
    return 0


def _box_report(box: dict[str, Any]) -> dict[str, Any]:
    return {
        "box": list(_box_tuple(box)),
        "label": str(box.get("label", "")),
        "confidence": box.get("confidence"),
    }


def _selection_source(prompt: str, points: list[tuple[int, int]]) -> str:
    if prompt and points:
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


def _box_contains_any_point(
    box: tuple[int, int, int, int],
    points: list[tuple[int, int]],
) -> bool:
    return any(_box_contains(box, point) for point in points)


def _box_point_distance(box: tuple[int, int, int, int], point: tuple[int, int]) -> float:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return ((center_x - point[0]) ** 2 + (center_y - point[1]) ** 2) ** 0.5


def _box_points_distance(box: tuple[int, int, int, int], points: list[tuple[int, int]]) -> float:
    return min(_box_point_distance(box, point) for point in points)


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
