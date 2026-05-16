"""Color and tone adjustment plugin."""

from __future__ import annotations

from PIL import Image, ImageEnhance, ImageStat

from expandiffusion import constants
from expandiffusion.image_utils import encode_png_data_url
from expandiffusion.plugin_actions import PluginAction, PluginActionContext, PluginTool
from expandiffusion.schemas import ControlSchema, PluginActionResult

ACTION_ID = "image-adjustments"
TOOL_ID = "image-adjustments"

PARAM_BRIGHTNESS = "image_adjustments_brightness"
PARAM_CONTRAST = "image_adjustments_contrast"
PARAM_SATURATION = "image_adjustments_saturation"
PARAM_VIBRANCE = "image_adjustments_vibrance"
PARAM_EXPOSURE = "image_adjustments_exposure"
PARAM_GAMMA = "image_adjustments_gamma"
PARAM_SHADOWS = "image_adjustments_shadows"
PARAM_HIGHLIGHTS = "image_adjustments_highlights"
PARAM_HUE = "image_adjustments_hue"
PARAM_WARMTH = "image_adjustments_warmth"
PARAM_TINT = "image_adjustments_tint"
PARAM_COLOR_MATCH = "image_adjustments_color_match"
PARAM_SHARPNESS = "image_adjustments_sharpness"


class ImageAdjustmentsAction(PluginAction):
    """Apply color and tone edits to an uploaded image."""

    id = ACTION_ID
    label = "Image adjustments"
    description = "Applies live color, tone, channel balance, and sharpness edits."

    def controls(self) -> list[ControlSchema]:
        return [
            ControlSchema(
                id=PARAM_BRIGHTNESS,
                label="Brightness",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=1.0,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
            ControlSchema(
                id=PARAM_CONTRAST,
                label="Contrast",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=1.0,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
            ControlSchema(
                id=PARAM_SATURATION,
                label="Saturation",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=1.0,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
            ControlSchema(
                id=PARAM_VIBRANCE,
                label="Vibrance",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=0.0,
                min=-100.0,
                max=100.0,
                step=1.0,
            ),
            ControlSchema(
                id=PARAM_EXPOSURE,
                label="Exposure",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=0.0,
                min=-2.0,
                max=2.0,
                step=0.05,
            ),
            ControlSchema(
                id=PARAM_GAMMA,
                label="Gamma",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=1.0,
                min=0.25,
                max=3.0,
                step=0.05,
            ),
            ControlSchema(
                id=PARAM_SHADOWS,
                label="Shadows",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=0.0,
                min=-100.0,
                max=100.0,
                step=1.0,
            ),
            ControlSchema(
                id=PARAM_HIGHLIGHTS,
                label="Highlights",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=0.0,
                min=-100.0,
                max=100.0,
                step=1.0,
            ),
            ControlSchema(
                id=PARAM_HUE,
                label="Hue",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=0.0,
                min=-180.0,
                max=180.0,
                step=1.0,
            ),
            ControlSchema(
                id=PARAM_WARMTH,
                label="Warmth",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=0.0,
                min=-100.0,
                max=100.0,
                step=1.0,
            ),
            ControlSchema(
                id=PARAM_TINT,
                label="Tint",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=0.0,
                min=-100.0,
                max=100.0,
                step=1.0,
            ),
            ControlSchema(
                id=PARAM_COLOR_MATCH,
                label="Auto color match",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=0.0,
                min=0.0,
                max=100.0,
                step=1.0,
            ),
            ControlSchema(
                id=PARAM_SHARPNESS,
                label="Sharpness",
                kind=constants.CONTROL_SLIDER,
                section=constants.CONTROL_SECTION_BASIC,
                default_value=1.0,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
        ]

    def run(self, context: PluginActionContext) -> PluginActionResult:
        image = context.image.convert("RGBA")
        image = _adjust_exposure(image, _float_control(context, PARAM_EXPOSURE, 0.0, -2.0, 2.0))
        image = _adjust_gamma(image, _float_control(context, PARAM_GAMMA, 1.0, 0.25, 3.0))
        image = _adjust_shadows_highlights(
            image,
            _float_control(context, PARAM_SHADOWS, 0.0, -100.0, 100.0),
            _float_control(context, PARAM_HIGHLIGHTS, 0.0, -100.0, 100.0),
        )
        image = _shift_hue(image, _float_control(context, PARAM_HUE, 0.0, -180.0, 180.0))
        image = _adjust_warmth(image, _float_control(context, PARAM_WARMTH, 0.0, -100.0, 100.0))
        image = _adjust_tint(image, _float_control(context, PARAM_TINT, 0.0, -100.0, 100.0))
        image = _auto_color_match(
            image,
            _float_control(context, PARAM_COLOR_MATCH, 0.0, 0.0, 100.0),
        )
        image = ImageEnhance.Brightness(image).enhance(
            _float_control(context, PARAM_BRIGHTNESS, 1.0, 0.0, 2.0)
        )
        image = ImageEnhance.Contrast(image).enhance(
            _float_control(context, PARAM_CONTRAST, 1.0, 0.0, 2.0)
        )
        image = ImageEnhance.Color(image).enhance(
            _float_control(context, PARAM_SATURATION, 1.0, 0.0, 2.0)
        )
        image = _adjust_vibrance(image, _float_control(context, PARAM_VIBRANCE, 0.0, -100.0, 100.0))
        image = ImageEnhance.Sharpness(image).enhance(
            _float_control(context, PARAM_SHARPNESS, 1.0, 0.0, 2.0)
        )
        return PluginActionResult(
            action_id=self.id,
            text="Image adjustments applied.",
            image=encode_png_data_url(image),
            data={"target": context.target},
        )


def register(context) -> None:
    action = ImageAdjustmentsAction()
    context.register_action(action)
    context.register_tool(
        PluginTool(
            id=TOOL_ID,
            label="Image Adjustments",
            description="Select an uploaded image, adjust color and tone, then apply.",
            action_id=action.id,
            icon="palette",
            icon_color="#f97316",
            accent_color="#f97316",
            target=constants.PLUGIN_TOOL_TARGET_IMAGE,
            live_preview=True,
            controls=action.controls(),
            default_values=action.defaults(),
        )
    )


def _shift_hue(image: Image.Image, degrees: float) -> Image.Image:
    if abs(degrees) < 0.01:
        return image
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    hue, saturation, value = rgba.convert("RGB").convert("HSV").split()
    offset = int(round((degrees / 360.0) * 255))
    shifted_hue = hue.point(lambda pixel: (pixel + offset) % 256)
    shifted = Image.merge("HSV", (shifted_hue, saturation, value)).convert("RGB")
    shifted.putalpha(alpha)
    return shifted


def _adjust_warmth(image: Image.Image, warmth: float) -> Image.Image:
    if abs(warmth) < 0.01:
        return image
    rgba = image.convert("RGBA")
    red, green, blue, alpha = rgba.split()
    normalized = warmth / 100.0
    red_factor = 1.0 + 0.28 * normalized
    blue_factor = 1.0 - 0.28 * normalized
    adjusted = Image.merge(
        "RGBA",
        (
            red.point(lambda pixel: _clamp_channel(pixel * red_factor)),
            green,
            blue.point(lambda pixel: _clamp_channel(pixel * blue_factor)),
            alpha,
        ),
    )
    return adjusted


def _adjust_tint(image: Image.Image, tint: float) -> Image.Image:
    if abs(tint) < 0.01:
        return image
    rgba = image.convert("RGBA")
    red, green, blue, alpha = rgba.split()
    normalized = tint / 100.0
    red_factor = 1.0 + 0.14 * normalized
    green_factor = 1.0 - 0.22 * normalized
    blue_factor = 1.0 + 0.14 * normalized
    return Image.merge(
        "RGBA",
        (
            red.point(lambda pixel: _clamp_channel(pixel * red_factor)),
            green.point(lambda pixel: _clamp_channel(pixel * green_factor)),
            blue.point(lambda pixel: _clamp_channel(pixel * blue_factor)),
            alpha,
        ),
    )


def _adjust_exposure(image: Image.Image, exposure: float) -> Image.Image:
    if abs(exposure) < 0.01:
        return image
    return _map_rgb_channels(image, lambda pixel: _clamp_channel(pixel * (2.0**exposure)))


def _adjust_gamma(image: Image.Image, gamma: float) -> Image.Image:
    if abs(gamma - 1.0) < 0.01:
        return image
    inverse = 1.0 / gamma
    return _map_rgb_channels(
        image,
        lambda pixel: _clamp_channel(((pixel / 255.0) ** inverse) * 255.0),
    )


def _adjust_shadows_highlights(
    image: Image.Image,
    shadows: float,
    highlights: float,
) -> Image.Image:
    if abs(shadows) < 0.01 and abs(highlights) < 0.01:
        return image
    shadow_strength = shadows / 100.0
    highlight_strength = highlights / 100.0

    def transform(pixel: int) -> int:
        normalized = pixel / 255.0
        shadow_delta = shadow_strength * ((1.0 - normalized) ** 2) * 78.0
        highlight_delta = highlight_strength * (normalized**2) * 78.0
        return _clamp_channel(pixel + shadow_delta + highlight_delta)

    return _map_rgb_channels(image, transform)


def _adjust_vibrance(image: Image.Image, vibrance: float) -> Image.Image:
    if abs(vibrance) < 0.01:
        return image
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    hue, saturation, value = rgba.convert("RGB").convert("HSV").split()
    normalized = vibrance / 100.0
    if normalized > 0:
        adjusted_saturation = saturation.point(
            lambda pixel: _clamp_channel(pixel + (255 - pixel) * normalized * 0.82)
        )
    else:
        adjusted_saturation = saturation.point(
            lambda pixel: _clamp_channel(pixel * (1.0 + normalized * 0.82))
        )
    adjusted = Image.merge("HSV", (hue, adjusted_saturation, value)).convert("RGB")
    adjusted.putalpha(alpha)
    return adjusted


def _auto_color_match(image: Image.Image, strength: float) -> Image.Image:
    if strength < 0.01:
        return image
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    stat = ImageStat.Stat(rgba.convert("RGB"), mask=alpha)
    means = stat.mean
    if len(means) < 3 or min(means[:3]) <= 0.01:
        return rgba
    target_mean = sum(means[:3]) / 3.0
    normalized = strength / 100.0
    factors = [1.0 + ((target_mean / mean) - 1.0) * normalized for mean in means[:3]]
    red, green, blue, alpha = rgba.split()
    return Image.merge(
        "RGBA",
        (
            red.point(lambda pixel: _clamp_channel(pixel * factors[0])),
            green.point(lambda pixel: _clamp_channel(pixel * factors[1])),
            blue.point(lambda pixel: _clamp_channel(pixel * factors[2])),
            alpha,
        ),
    )


def _map_rgb_channels(image: Image.Image, transform) -> Image.Image:
    rgba = image.convert("RGBA")
    red, green, blue, alpha = rgba.split()
    return Image.merge(
        "RGBA",
        (
            red.point(transform),
            green.point(transform),
            blue.point(transform),
            alpha,
        ),
    )


def _float_control(
    context: PluginActionContext,
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = context.control(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return max(minimum, min(maximum, float(value)))


def _clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))
