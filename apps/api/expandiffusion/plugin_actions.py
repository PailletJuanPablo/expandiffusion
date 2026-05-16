"""Generic plugin action contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from . import constants
from .errors import AppError
from .schemas import ControlSchema, PluginActionInfo, PluginActionResult, PluginToolInfo


@dataclass(slots=True)
class PluginActionContext:
    """Runtime inputs passed into plugin actions."""

    image: Image.Image
    controls: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def control(self, key: str, default: Any = None) -> Any:
        """Return an action control value."""
        return self.controls.get(key, default)


class PluginAction(ABC):
    """Base interface for plugins that process a selected image block."""

    id: str
    label: str
    description: str = ""
    menu: str = constants.PLUGIN_ACTION_MENU_SELECTION

    def controls(self) -> list[ControlSchema]:
        """Return frontend controls contributed by this action."""
        return []

    def defaults(self) -> dict[str, Any]:
        """Return default control values contributed by this action."""
        return {
            control.id: control.default_value
            for control in self.controls()
            if control.default_value is not None
        }

    @abstractmethod
    def run(self, context: PluginActionContext) -> PluginActionResult:
        """Run the action and return a generic result."""

    def info(self, plugin_id: str | None = None) -> PluginActionInfo:
        """Return serializable action metadata."""
        return PluginActionInfo(
            id=self.id,
            label=self.label,
            description=self.description,
            plugin_id=plugin_id,
            menu=self.menu,
            controls=[
                control.model_copy(update={"plugin_id": plugin_id})
                for control in self.controls()
            ],
            default_values=self.defaults(),
        )


@dataclass(frozen=True, slots=True)
class PluginTool:
    """Editor toolbar contribution provided by a plugin."""

    id: str
    label: str
    action_id: str
    description: str = ""
    icon: str = "puzzle"
    icon_color: str | None = None
    accent_color: str | None = None
    result_label: str | None = None
    target: str = constants.PLUGIN_TOOL_TARGET_FRAME
    live_preview: bool = False
    controls: list[ControlSchema] = field(default_factory=list)
    default_values: dict[str, Any] = field(default_factory=dict)

    def info(self, plugin_id: str | None = None) -> PluginToolInfo:
        """Return serializable tool metadata."""
        defaults = dict(self.default_values)
        for control in self.controls:
            if control.default_value is not None and control.id not in defaults:
                defaults[control.id] = control.default_value
        return PluginToolInfo(
            id=self.id,
            label=self.label,
            description=self.description,
            plugin_id=plugin_id,
            action_id=self.action_id,
            icon=self.icon,
            icon_color=self.icon_color,
            accent_color=self.accent_color,
            result_label=self.result_label,
            target=self.target,
            live_preview=self.live_preview,
            controls=[
                control.model_copy(update={"plugin_id": plugin_id})
                for control in self.controls
            ],
            default_values=defaults,
        )


class PluginActionRegistry:
    """Mutable registry of plugin-provided actions."""

    def __init__(self) -> None:
        self._actions: dict[str, PluginAction] = {}
        self._action_plugin_ids: dict[str, str] = {}

    def register(self, action: PluginAction, plugin_id: str) -> None:
        """Register an action by id."""
        if action.id in self._actions:
            raise AppError(
                constants.ERROR_PLUGIN_LOAD_FAILED,
                f"Plugin action '{action.id}' is already registered.",
                status_code=500,
                details={"action_id": action.id, "plugin_id": plugin_id},
            )
        self._actions[action.id] = action
        self._action_plugin_ids[action.id] = plugin_id

    def actions(self) -> Iterable[PluginAction]:
        """Yield registered plugin actions."""
        return self._actions.values()

    def get(self, action_id: str) -> PluginAction:
        """Return a registered plugin action."""
        action = self._actions.get(action_id)
        if action is None:
            raise AppError(
                constants.ERROR_PLUGIN_ACTION_NOT_FOUND,
                f"Plugin action '{action_id}' was not found.",
                status_code=404,
                details={"action_id": action_id},
            )
        return action

    def action_infos(self) -> list[PluginActionInfo]:
        """Return serializable metadata for all registered actions."""
        return [
            action.info(self.plugin_id_for(action.id))
            for action in sorted(self._actions.values(), key=lambda item: item.id)
        ]

    def plugin_id_for(self, action_id: str) -> str | None:
        """Return the plugin owner for an action id."""
        return self._action_plugin_ids.get(action_id)

    def unregister(self, action_id: str) -> None:
        """Remove an action from the registry."""
        self._actions.pop(action_id, None)
        self._action_plugin_ids.pop(action_id, None)

    def unregister_plugin(self, plugin_id: str) -> None:
        """Remove all actions owned by a plugin."""
        for action_id, owner_id in list(self._action_plugin_ids.items()):
            if owner_id == plugin_id:
                self.unregister(action_id)


class PluginToolRegistry:
    """Mutable registry of plugin-provided editor tools."""

    def __init__(self) -> None:
        self._tools: dict[str, PluginTool] = {}
        self._tool_plugin_ids: dict[str, str] = {}

    def register(self, tool: PluginTool, plugin_id: str) -> None:
        """Register an editor tool by id."""
        if tool.id in self._tools:
            raise AppError(
                constants.ERROR_PLUGIN_LOAD_FAILED,
                f"Plugin tool '{tool.id}' is already registered.",
                status_code=500,
                details={"tool_id": tool.id, "plugin_id": plugin_id},
            )
        self._tools[tool.id] = tool
        self._tool_plugin_ids[tool.id] = plugin_id

    def tool_infos(self) -> list[PluginToolInfo]:
        """Return serializable metadata for all registered tools."""
        return [
            tool.info(self.plugin_id_for(tool.id))
            for tool in sorted(self._tools.values(), key=lambda item: item.id)
        ]

    def plugin_id_for(self, tool_id: str) -> str | None:
        """Return the plugin owner for a tool id."""
        return self._tool_plugin_ids.get(tool_id)

    def unregister(self, tool_id: str) -> None:
        """Remove an editor tool from the registry."""
        self._tools.pop(tool_id, None)
        self._tool_plugin_ids.pop(tool_id, None)

    def unregister_plugin(self, plugin_id: str) -> None:
        """Remove all editor tools owned by a plugin."""
        for tool_id, owner_id in list(self._tool_plugin_ids.items()):
            if owner_id == plugin_id:
                self.unregister(tool_id)
