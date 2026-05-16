# Expandiffusion Plugins

Plugins extend the backend by registering model adapters, generation postprocessors, generic actions, or editor tools. Metadata is exposed to the frontend so the UI can render model sources, toolbar tools, processing actions, and controls dynamically.

## Layout

```text
plugins/
  my-plugin/
    plugin.json
    plugin.py
```

## Manifest

`plugin.json`:

```json
{
  "id": "my-plugin",
  "label": "My Plugin",
  "version": "0.1.0",
  "description": "Optional description."
}
```

`id`, `label`, and `version` are required. Invalid manifests are reported by `GET /api/plugins` and the plugin is not loaded.

## Registration Hook

Preferred hook:

```python
def register(context):
    context.register_model_adapter(MyModelAdapter())
```

Legacy hook, still supported:

```python
def register_model_adapters(registry):
    registry.register(MyModelAdapter())
```

Adapter ids must be unique. If a plugin registers an id that already exists, loading that plugin fails with `PLUGIN_LOAD_FAILED`; the existing adapter is not replaced.

Plugins can also register postprocessors:

```python
from expandiffusion.postprocessors import GenerationPostprocessor


class MyPostprocessor(GenerationPostprocessor):
    id = "my-postprocessor"
    label = "My Postprocessor"

    def process(self, context):
        return context.generated


def register(context):
    context.register_generation_postprocessor(MyPostprocessor())
```

Postprocessors own their controls and generation defaults. Any plugin-specific
generation parameter should be read through `context.parameter("my_key")`, not
added to the core generation schema.

Correction plugins are postprocessors with `category = "correction"`. They are
not executed automatically; the frontend sends the active ordered ids in
`parameters.correction_pipeline`.

Plugins can register selected-block actions:

```python
from expandiffusion.plugin_actions import PluginAction
from expandiffusion.schemas import PluginActionResult


class MyAction(PluginAction):
    id = "my-action"
    label = "My Action"

    def run(self, context):
        return PluginActionResult(action_id=self.id, text="Action output")


def register(context):
    context.register_action(MyAction())
```

Actions own their controls and receive an image data URL decoded by the core API.
Any action-specific option should be read through `context.control("my_key")`,
not added to the core generation schema.

Plugins can also expose an editor tool that appears in the left toolbar and
runs a registered action from its custom inspector controls:

```python
from expandiffusion.plugin_actions import PluginAction, PluginTool


def register(context):
    action = MyAction()
    context.register_action(action)
    context.register_tool(
        PluginTool(
            id="my-tool",
            label="My Tool",
            action_id=action.id,
            icon="text-search",
            icon_color="#4f46e5",
            accent_color="#4f46e5",
            result_label="Action result",
            target="frame",
            controls=action.controls(),
            default_values=action.defaults(),
        )
    )
```

Tools own their inspector controls. The core editor only exposes generic
toolbar/control/action interfaces; plugin-specific behavior remains inside the
plugin action. Use `target="frame"` for tools that process the active canvas
rectangle, or `target="image"` for tools that process a selected uploaded base
or reference image. Toolbar visuals can be customized with `icon`,
`icon_color`, and `accent_color`; text output headings can be customized with
`result_label`. Image tools that set `live_preview=True` are re-run from their
inspector controls and preview their image output directly on the selected
canvas layer until the user applies or resets it.

## Included Postprocessors

- `auto-detailer`: localized second-pass inpainting for detected faces/bodies.
  Face targets use RetinaFace through the optional `face-restoration` extra;
  body targets use OpenCV.
- `art-face-repair`: Florence-2 guided local repaint for malformed faces in
  paintings. It expands the face context, upscales the detail crop, and skips
  outputs that introduce a dark halo.
- `image-adjustments`: live exposure, gamma, shadows, highlights, saturation,
  vibrance, hue, warmth, tint, auto color-match, contrast, and sharpness
  adjustments for uploaded raster/reference images.
- `image-to-text`: CLIP Interrogator tool/action that converts the selected
  image block into a Stable Diffusion prompt. It requires the API diffusers
  extra.
- `gfpgan-face-restore`: dedicated GFPGAN/RestoreFormer face restoration. It
  requires the optional API extra `face-restoration`.
- `correction-mask-feather`: soft mask-edge lighting correction.
- `correction-border-blend`: OpenCV gradient-domain boundary blending.
- `correction-color-match`: LAB color statistic transfer.
- `correction-histogram-match`: generated-region histogram matching.
- `correction-multiband-blend`: multiscale Laplacian boundary blending.

## Adapter Contract

Adapters implement `expandiffusion.adapters.base.ModelAdapter`:

```python
from expandiffusion.adapters.base import ModelAdapter
from expandiffusion.schemas import AdapterCapabilities


class MyModelAdapter(ModelAdapter):
    id = "my-model"
    label = "My Model"
    family = "custom"
    description = "Custom generation adapter."
    default_model_id = None
    capabilities = AdapterCapabilities(inpaint=True, outpaint=True)

    def load(self, config):
        self.loaded_config = config

    def unload(self):
        self.loaded_config = None

    def generate(self, context):
        return [context.source]
```

The base adapter supplies default model source schemas and generation controls. Override `model_sources()`, `load_controls()`, `generation_controls()`, or `generation_defaults()` when the adapter needs a different frontend surface.

## Runtime API

- `GET /api/plugins`: plugin load status, registered contribution ids, and load errors.
- `GET /api/plugins/actions`: enabled plugin action metadata and controls.
- `GET /api/plugins/tools`: enabled plugin editor tool metadata and controls.
- `POST /api/plugins/actions/{action_id}/run`: execute an action for the selected image block.
- `POST /api/plugins/{plugin_id}/enable`: enable and load a plugin.
- `POST /api/plugins/{plugin_id}/disable`: disable a plugin and unregister its adapters/postprocessors/actions/tools.
- `GET /api/adapters`: adapter metadata, plugin owner, model source schemas, load controls, generation controls, and generation defaults.
