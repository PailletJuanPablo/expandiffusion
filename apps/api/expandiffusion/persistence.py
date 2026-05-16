"""JSON persistence for local application history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from . import constants
from .schemas import JobResult, ModelInfo, ModelLoadRequest, PersistentState


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class PersistenceStore:
    """Persist model, project and generation summaries to one JSON file."""

    def __init__(self, path: Path = constants.DEFAULT_STATE_PATH) -> None:
        self.path = path
        self._lock = RLock()
        self._state = self._read_state()

    def get_state(self) -> PersistentState:
        """Return current persisted state."""
        with self._lock:
            return PersistentState.model_validate(self._state)

    def record_model_loaded(self, model: ModelInfo, request: ModelLoadRequest) -> None:
        """Persist a successful model load."""
        loaded_at = utc_now()
        record = {
            "adapter_id": model.adapter_id,
            "adapter_label": model.adapter_label,
            "model_id": model.model_id,
            "local_path": model.local_path,
            "single_file_path": model.single_file_path,
            "model_url": request.model_url,
            "device": model.device,
            "dtype": model.dtype,
            "safety_checker": request.safety_checker,
            "loaded_at": loaded_at,
            "unloaded_at": None,
        }

        def mutate(state: dict[str, Any]) -> None:
            self._mark_unloaded(state, model.adapter_id, loaded_at)
            state["current_model"] = record
            state["model_loads"] = [record, *state["model_loads"]][
                : constants.PERSISTENCE_MAX_MODEL_LOADS
            ]

        self._mutate(mutate)

    def record_model_unloaded(self, adapter_id: str) -> None:
        """Persist a model unload."""
        unloaded_at = utc_now()

        def mutate(state: dict[str, Any]) -> None:
            current_model = state.get("current_model")
            if current_model and current_model.get("adapter_id") == adapter_id:
                state["current_model"] = None
            self._mark_unloaded(state, adapter_id, unloaded_at)

        self._mutate(mutate)

    def record_generation_created(self, job: Any) -> None:
        """Persist a queued generation job without image payloads."""
        request = job.request
        parameters = request.parameters
        record = {
            "job_id": job.id,
            "project_id": request.project_id,
            "adapter_id": request.adapter_id,
            "prompt": parameters.prompt,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "width": parameters.width,
            "height": parameters.height,
            "sample_count": parameters.sample_count,
            "result_count": 0,
            "error": None,
        }

        def mutate(state: dict[str, Any]) -> None:
            state["generations"] = [record, *state["generations"]][
                : constants.PERSISTENCE_MAX_GENERATIONS
            ]
            self._upsert_project(state, record, increment_generation_count=True)

        self._mutate(mutate)

    def record_generation_updated(self, job: Any) -> None:
        """Persist terminal or running state for an existing generation."""
        result: JobResult | None = job.result
        updates = {
            "status": job.status,
            "updated_at": job.updated_at,
            "result_count": len(result.images) if result else 0,
            "error": job.error,
        }

        def mutate(state: dict[str, Any]) -> None:
            generation = self._find_generation(state, job.id)
            if generation is None:
                return
            generation.update(updates)
            self._upsert_project(state, generation, increment_generation_count=False)

        self._mutate(mutate)

    def is_plugin_enabled(self, plugin_id: str) -> bool:
        """Return persisted plugin enablement, defaulting to enabled."""
        with self._lock:
            for item in self._state.get("plugin_states", []):
                if item["plugin_id"] == plugin_id:
                    return bool(item["enabled"])
        return True

    def set_plugin_enabled(self, plugin_id: str, enabled: bool) -> None:
        """Persist plugin enablement state."""
        updated_at = utc_now()

        def mutate(state: dict[str, Any]) -> None:
            for item in state["plugin_states"]:
                if item["plugin_id"] == plugin_id:
                    item["enabled"] = enabled
                    item["updated_at"] = updated_at
                    return
            state["plugin_states"].append(
                {
                    "plugin_id": plugin_id,
                    "enabled": enabled,
                    "updated_at": updated_at,
                }
            )

        self._mutate(mutate)

    def _read_state(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_state()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return PersistentState.model_validate(payload).model_dump(mode="json")

    def _default_state(self) -> dict[str, Any]:
        return PersistentState(updated_at=utc_now()).model_dump(mode="json")

    def _mutate(self, callback: Any) -> None:
        with self._lock:
            callback(self._state)
            self._state["updated_at"] = utc_now()
            self._write_state()

    def _write_state(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        payload = json.dumps(self._state, indent=2, ensure_ascii=False)
        temp_path.write_text(f"{payload}\n", encoding="utf-8")
        temp_path.replace(self.path)

    def _mark_unloaded(self, state: dict[str, Any], adapter_id: str, timestamp: str) -> None:
        for item in state["model_loads"]:
            if item["adapter_id"] == adapter_id and item.get("unloaded_at") is None:
                item["unloaded_at"] = timestamp
                return

    def _find_generation(self, state: dict[str, Any], job_id: str) -> dict[str, Any] | None:
        for item in state["generations"]:
            if item["job_id"] == job_id:
                return item
        return None

    def _upsert_project(
        self,
        state: dict[str, Any],
        generation: dict[str, Any],
        increment_generation_count: bool,
    ) -> None:
        project_id = generation.get("project_id")
        if not project_id:
            return
        project = self._find_project(state, project_id)
        if project is None:
            project = {
                "project_id": project_id,
                "created_at": generation["created_at"],
                "updated_at": generation["updated_at"],
                "generation_count": 0,
                "last_job_id": None,
                "last_prompt": "",
                "last_status": "",
                "width": None,
                "height": None,
            }
            state["projects"] = [project, *state["projects"]][
                : constants.PERSISTENCE_MAX_PROJECTS
            ]
        if increment_generation_count:
            project["generation_count"] += 1
        project["updated_at"] = generation["updated_at"]
        project["last_job_id"] = generation["job_id"]
        project["last_prompt"] = generation["prompt"]
        project["last_status"] = generation["status"]
        project["width"] = generation["width"]
        project["height"] = generation["height"]

    def _find_project(self, state: dict[str, Any], project_id: str) -> dict[str, Any] | None:
        for item in state["projects"]:
            if item["project_id"] == project_id:
                return item
        return None
