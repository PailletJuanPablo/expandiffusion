"""Local plugin discovery."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

from . import constants
from .adapters.base import ModelAdapter
from .adapters.registry import AdapterRegistry
from .errors import AppError
from .image_utils import decode_data_url
from .persistence import PersistenceStore
from .plugin_actions import (
    PluginAction,
    PluginActionContext,
    PluginActionRegistry,
    PluginTool,
    PluginToolRegistry,
)
from .postprocessors import GenerationPostprocessor, GenerationPostprocessorRegistry
from .schemas import (
    AdapterInfo,
    PluginActionInfo,
    PluginActionResult,
    PluginActionRunRequest,
    PluginInfo,
    PluginManifest,
    PluginToolInfo,
)


class PluginRegistrationContext:
    """Scoped plugin registration API."""

    def __init__(
        self,
        registry: AdapterRegistry,
        postprocessors: GenerationPostprocessorRegistry,
        actions: PluginActionRegistry,
        tools: PluginToolRegistry,
        plugin_id: str,
    ) -> None:
        self.registry = registry
        self.postprocessors = postprocessors
        self.actions = actions
        self.tools = tools
        self.plugin_id = plugin_id
        self.adapter_ids: list[str] = []
        self.postprocessor_ids: list[str] = []
        self.action_ids: list[str] = []
        self.tool_ids: list[str] = []

    def register_model_adapter(self, adapter: ModelAdapter) -> None:
        """Register a model adapter owned by the plugin."""
        self.registry.register(adapter, plugin_id=self.plugin_id)
        self.adapter_ids.append(adapter.id)

    def register_generation_postprocessor(self, processor: GenerationPostprocessor) -> None:
        """Register a generation postprocessor owned by the plugin."""
        self.postprocessors.register(processor, plugin_id=self.plugin_id)
        self.postprocessor_ids.append(processor.id)

    def register_action(self, action: PluginAction) -> None:
        """Register an action owned by the plugin."""
        self.actions.register(action, plugin_id=self.plugin_id)
        self.action_ids.append(action.id)

    def register_tool(self, tool: PluginTool) -> None:
        """Register an editor tool owned by the plugin."""
        self.tools.register(tool, plugin_id=self.plugin_id)
        self.tool_ids.append(tool.id)

    def register(self, adapter: ModelAdapter) -> None:
        """Legacy-compatible adapter registration alias."""
        self.register_model_adapter(adapter)


def load_local_plugins(
    registry: AdapterRegistry,
    directory: Path | None = None,
    postprocessors: GenerationPostprocessorRegistry | None = None,
    actions: PluginActionRegistry | None = None,
    tools: PluginToolRegistry | None = None,
) -> list[PluginInfo]:
    """Load plugin folders from the configured plugin directory."""
    plugin_directory = directory or Path(
        os.getenv("EXPANDIFFUSION_PLUGIN_DIR", constants.DEFAULT_PLUGIN_DIR)
    )
    if not plugin_directory.exists():
        return []
    postprocessor_registry = postprocessors or GenerationPostprocessorRegistry()
    action_registry = actions or PluginActionRegistry()
    tool_registry = tools or PluginToolRegistry()
    loaded: list[PluginInfo] = []
    for plugin_dir in sorted(path for path in plugin_directory.iterdir() if path.is_dir()):
        manifest_path = plugin_dir / "plugin.json"
        module_path = plugin_dir / "plugin.py"
        if not manifest_path.exists() or not module_path.exists():
            continue
        loaded.append(
            _load_plugin(
                registry,
                postprocessor_registry,
                action_registry,
                tool_registry,
                plugin_dir,
                manifest_path,
                module_path,
            )
        )
    return loaded


class PluginManager:
    """Load, list and toggle local plugins."""

    def __init__(
        self,
        registry: AdapterRegistry,
        postprocessors: GenerationPostprocessorRegistry,
        persistence: PersistenceStore,
        directory: Path | None = None,
        actions: PluginActionRegistry | None = None,
        tools: PluginToolRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.postprocessors = postprocessors
        self.actions = actions or PluginActionRegistry()
        self.tools = tools or PluginToolRegistry()
        self.persistence = persistence
        self.directory = directory or Path(
            os.getenv("EXPANDIFFUSION_PLUGIN_DIR", constants.DEFAULT_PLUGIN_DIR)
        )
        self._plugins: dict[str, PluginInfo] = {}

    def load_all(self) -> list[PluginInfo]:
        """Scan the plugin directory and load enabled plugins."""
        self._plugins = {}
        if not self.directory.exists():
            return []
        for plugin_dir in sorted(path for path in self.directory.iterdir() if path.is_dir()):
            manifest_path = plugin_dir / "plugin.json"
            module_path = plugin_dir / "plugin.py"
            if not manifest_path.exists() or not module_path.exists():
                continue
            manifest = _read_manifest_fallback(plugin_dir, manifest_path)
            if not self.persistence.is_plugin_enabled(manifest.id):
                self._plugins[manifest.id] = PluginInfo(
                    id=manifest.id,
                    label=manifest.label,
                    version=manifest.version,
                    description=manifest.description,
                    path=str(plugin_dir),
                    enabled=False,
                    loaded=False,
                )
                continue
            self._plugins[manifest.id] = _load_plugin(
                self.registry,
                self.postprocessors,
                self.actions,
                self.tools,
                plugin_dir,
                manifest_path,
                module_path,
            )
        return self.list()

    def list(self) -> list[PluginInfo]:
        """Return current plugin status."""
        return list(self._plugins.values())

    def list_adapters(self) -> list[AdapterInfo]:
        """Return adapter metadata plus enabled postprocessor controls."""
        controls = []
        defaults = {}
        postprocessor_infos = self.postprocessors.processor_infos()
        for processor in self.postprocessors.processors():
            plugin_id = self.postprocessors.plugin_id_for(processor.id)
            controls.extend(
                control.model_copy(update={"plugin_id": plugin_id})
                for control in processor.generation_controls()
            )
            defaults.update(processor.generation_defaults())
        if not controls and not defaults and not postprocessor_infos:
            return self.registry.list()
        return [
            adapter.model_copy(
                update={
                    "generation_controls": [*adapter.generation_controls, *controls],
                    "generation_defaults": {
                        **adapter.generation_defaults,
                        **defaults,
                    },
                    "postprocessors": postprocessor_infos,
                }
            )
            for adapter in self.registry.list()
        ]

    def list_actions(self) -> list[PluginActionInfo]:
        """Return enabled plugin actions."""
        return self.actions.action_infos()

    def list_tools(self) -> list[PluginToolInfo]:
        """Return enabled plugin editor tools."""
        return self.tools.tool_infos()

    def run_action(
        self,
        action_id: str,
        request: PluginActionRunRequest,
    ) -> PluginActionResult:
        """Run a registered plugin action on a selected image."""
        action = self.actions.get(action_id)
        try:
            result = action.run(
                PluginActionContext(
                    image=decode_data_url(request.image).convert("RGB"),
                    controls={**action.defaults(), **request.controls},
                    target=request.target,
                    metadata=request.metadata,
                )
            )
        except AppError:
            raise
        except Exception as exc:
            raise AppError(
                constants.ERROR_PLUGIN_ACTION_FAILED,
                f"Plugin action '{action_id}' failed: {exc}",
                status_code=500,
                details={"action_id": action_id},
            ) from exc
        return result.model_copy(update={"action_id": action.id})

    def enable(self, plugin_id: str) -> PluginInfo:
        """Enable and load a plugin by id."""
        existing = self._plugins.get(plugin_id)
        if existing and existing.loaded:
            self.persistence.set_plugin_enabled(plugin_id, True)
            return existing.model_copy(update={"enabled": True})
        plugin_paths = self._plugin_paths(plugin_id)
        self.persistence.set_plugin_enabled(plugin_id, True)
        plugin = _load_plugin(
            self.registry,
            self.postprocessors,
            self.actions,
            self.tools,
            plugin_paths[0],
            plugin_paths[1],
            plugin_paths[2],
        )
        self._plugins[plugin.id] = plugin
        return plugin

    def disable(self, plugin_id: str) -> PluginInfo:
        """Disable a plugin and unregister its runtime contributions."""
        plugin = self._plugins.get(plugin_id)
        if plugin is None:
            plugin_paths = self._plugin_paths(plugin_id)
            manifest = _read_manifest_fallback(plugin_paths[0], plugin_paths[1])
            plugin = PluginInfo(
                id=manifest.id,
                label=manifest.label,
                version=manifest.version,
                description=manifest.description,
                path=str(plugin_paths[0]),
                enabled=True,
                loaded=False,
            )
        for adapter_id in plugin.adapter_ids:
            self.registry.unregister(adapter_id)
        for processor_id in plugin.postprocessor_ids:
            self.postprocessors.unregister(processor_id)
        for action_id in plugin.action_ids:
            self.actions.unregister(action_id)
        for tool_id in plugin.tool_ids:
            self.tools.unregister(tool_id)
        self.persistence.set_plugin_enabled(plugin_id, False)
        disabled = plugin.model_copy(
            update={
                "adapter_ids": [],
                "postprocessor_ids": [],
                "action_ids": [],
                "tool_ids": [],
                "enabled": False,
                "loaded": False,
                "error_code": None,
                "error": None,
            }
        )
        self._plugins[plugin_id] = disabled
        return disabled

    def plugin_adapter_ids(self, plugin_id: str) -> list[str]:
        """Return registered adapter ids owned by a plugin."""
        plugin = self._plugins.get(plugin_id)
        return plugin.adapter_ids if plugin else []

    def _plugin_paths(self, plugin_id: str) -> tuple[Path, Path, Path]:
        if not self.directory.exists():
            raise _plugin_not_found(plugin_id)
        for plugin_dir in sorted(path for path in self.directory.iterdir() if path.is_dir()):
            manifest_path = plugin_dir / "plugin.json"
            module_path = plugin_dir / "plugin.py"
            if not manifest_path.exists() or not module_path.exists():
                continue
            manifest = _read_manifest_fallback(plugin_dir, manifest_path)
            if manifest.id == plugin_id:
                return plugin_dir, manifest_path, module_path
        raise _plugin_not_found(plugin_id)


def _load_plugin(
    registry: AdapterRegistry,
    postprocessors: GenerationPostprocessorRegistry,
    actions: PluginActionRegistry,
    tools: PluginToolRegistry,
    plugin_dir: Path,
    manifest_path: Path,
    module_path: Path,
) -> PluginInfo:
    try:
        manifest = PluginManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        context = PluginRegistrationContext(
            registry,
            postprocessors,
            actions,
            tools,
            manifest.id,
        )
        module = _load_module(manifest.id, module_path)
        _call_registration_hook(module, context)
        return PluginInfo(
            id=manifest.id,
            label=manifest.label,
            version=manifest.version,
            description=manifest.description,
            path=str(plugin_dir),
            adapter_ids=context.adapter_ids,
            postprocessor_ids=context.postprocessor_ids,
            action_ids=context.action_ids,
            tool_ids=context.tool_ids,
            enabled=True,
            loaded=True,
        )
    except Exception as exc:
        for adapter_id in context.adapter_ids if "context" in locals() else []:
            registry.unregister(adapter_id)
        for processor_id in context.postprocessor_ids if "context" in locals() else []:
            postprocessors.unregister(processor_id)
        for action_id in context.action_ids if "context" in locals() else []:
            actions.unregister(action_id)
        for tool_id in context.tool_ids if "context" in locals() else []:
            tools.unregister(tool_id)
        manifest = _read_manifest_fallback(plugin_dir, manifest_path)
        return PluginInfo(
            id=manifest.id,
            label=manifest.label,
            version=manifest.version,
            description=manifest.description,
            path=str(plugin_dir),
            enabled=True,
            loaded=False,
            error_code=constants.ERROR_PLUGIN_LOAD_FAILED,
            error=str(exc),
        )


def _call_registration_hook(module: ModuleType, context: PluginRegistrationContext) -> None:
    hook = getattr(module, "register", None)
    if callable(hook):
        hook(context)
        return
    legacy_hook = getattr(module, "register_model_adapters", None)
    if callable(legacy_hook):
        legacy_hook(context)


def _read_manifest_fallback(plugin_dir: Path, manifest_path: Path) -> PluginManifest:
    try:
        return PluginManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return PluginManifest(id=plugin_dir.name, label=plugin_dir.name, version="unknown")


def _load_module(plugin_id: str, module_path: Path) -> ModuleType:
    module_name = f"expandiffusion_plugin_{plugin_id.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Plugin module cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _plugin_not_found(plugin_id: str) -> AppError:
    return AppError(
        constants.ERROR_PLUGIN_NOT_FOUND,
        f"Plugin '{plugin_id}' was not found.",
        status_code=404,
    )
