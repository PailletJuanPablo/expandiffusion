"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import os

from fastapi import BackgroundTasks, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__, constants
from .adapters.registry import create_default_registry
from .errors import AppError
from .jobs import JobStore
from .persistence import PersistenceStore
from .plugin_actions import PluginActionRegistry, PluginToolRegistry
from .plugins import PluginManager
from .postprocessors import GenerationPostprocessorRegistry
from .runtime import inspect_runtime
from .schemas import (
    HealthResponse,
    ModelLoadProgress,
    ModelLoadRequest,
    OutpaintRequest,
    PersistentState,
    PluginActionInfo,
    PluginActionResult,
    PluginActionRunRequest,
    PluginInfo,
    PluginToolInfo,
    RuntimeInfo,
)
from .services import GenerationService, ModelService

registry = create_default_registry()
persistence = PersistenceStore()
postprocessors = GenerationPostprocessorRegistry()
actions = PluginActionRegistry()
tools = PluginToolRegistry()
plugin_manager = PluginManager(
    registry,
    postprocessors,
    persistence,
    actions=actions,
    tools=tools,
)
plugin_manager.load_all()
jobs = JobStore(persistence)
models = ModelService(registry, persistence)
generations = GenerationService(registry, models, jobs, postprocessors)

app = FastAPI(title="Expandiffusion API", version=__version__)
web_port = os.getenv("EXPANDIFFUSION_WEB_PORT", "5180")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://127.0.0.1:{web_port}",
        f"http://localhost:{web_port}",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    """Serialize application errors consistently."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )


@app.get(f"{constants.API_PREFIX}/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return API health and active model state."""
    return HealthResponse(
        ok=True,
        version=__version__,
        loaded_adapter_id=models.loaded_adapter_id,
        runtime=inspect_runtime(),
    )


@app.get(f"{constants.API_PREFIX}/runtime", response_model=RuntimeInfo)
async def runtime() -> RuntimeInfo:
    """Return PyTorch and CUDA visibility."""
    return inspect_runtime()


@app.get(f"{constants.API_PREFIX}/adapters")
async def list_adapters():
    """Return registered adapters."""
    return plugin_manager.list_adapters()


@app.get(f"{constants.API_PREFIX}/plugins", response_model=list[PluginInfo])
async def list_plugins() -> list[PluginInfo]:
    """Return local plugin load status."""
    return plugin_manager.list()


@app.get(f"{constants.API_PREFIX}/plugins/actions", response_model=list[PluginActionInfo])
async def list_plugin_actions() -> list[PluginActionInfo]:
    """Return plugin-provided actions."""
    return plugin_manager.list_actions()


@app.get(f"{constants.API_PREFIX}/plugins/tools", response_model=list[PluginToolInfo])
async def list_plugin_tools() -> list[PluginToolInfo]:
    """Return plugin-provided editor tools."""
    return plugin_manager.list_tools()


@app.post(
    f"{constants.API_PREFIX}/plugins/actions/{{action_id}}/run",
    response_model=PluginActionResult,
)
async def run_plugin_action(
    action_id: str,
    request: PluginActionRunRequest,
) -> PluginActionResult:
    """Run a plugin action on a selected image."""
    return await asyncio.to_thread(plugin_manager.run_action, action_id, request)


@app.post(f"{constants.API_PREFIX}/plugins/{{plugin_id}}/enable", response_model=PluginInfo)
async def enable_plugin(plugin_id: str) -> PluginInfo:
    """Enable and load a local plugin."""
    return plugin_manager.enable(plugin_id)


@app.post(f"{constants.API_PREFIX}/plugins/{{plugin_id}}/disable", response_model=PluginInfo)
async def disable_plugin(plugin_id: str) -> PluginInfo:
    """Disable a local plugin and unregister its runtime contributions."""
    if models.loaded_adapter_id in plugin_manager.plugin_adapter_ids(plugin_id):
        models.unload(models.loaded_adapter_id)
    return plugin_manager.disable(plugin_id)


@app.get(f"{constants.API_PREFIX}/models")
async def list_models():
    """Return load state for available adapters."""
    return models.list_models()


@app.get(f"{constants.API_PREFIX}/models/load/progress", response_model=ModelLoadProgress)
async def model_load_progress() -> ModelLoadProgress:
    """Return current model load progress."""
    return models.load_progress()


@app.post(f"{constants.API_PREFIX}/models/load/cancel", response_model=ModelLoadProgress)
async def cancel_model_load() -> ModelLoadProgress:
    """Cancel the active model load when the adapter supports interruption."""
    return models.cancel_load()


@app.get(f"{constants.API_PREFIX}/state", response_model=PersistentState)
async def persistent_state() -> PersistentState:
    """Return persisted local history."""
    return persistence.get_state()


@app.post(f"{constants.API_PREFIX}/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model adapter."""
    return await asyncio.to_thread(models.load, request)


@app.post(f"{constants.API_PREFIX}/models/unload")
async def unload_model(request: ModelLoadRequest):
    """Unload a model adapter."""
    return await asyncio.to_thread(models.unload, request.adapter_id)


@app.post(f"{constants.API_PREFIX}/generations/outpaint")
async def outpaint(request: OutpaintRequest, background_tasks: BackgroundTasks):
    """Start an outpaint job."""
    response = generations.start_outpaint(request)
    background_tasks.add_task(generations.run_outpaint, response.job_id)
    return response


@app.post(f"{constants.API_PREFIX}/generations/inpaint")
async def inpaint(request: OutpaintRequest, background_tasks: BackgroundTasks):
    """Start an inpaint job."""
    response = generations.start_inpaint(request)
    background_tasks.add_task(generations.run_inpaint, response.job_id)
    return response


@app.post(f"{constants.API_PREFIX}/jobs/{{job_id}}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued or running job."""
    return await jobs.cancel(job_id)


@app.get(f"{constants.API_PREFIX}/jobs/{{job_id}}")
async def get_job(job_id: str):
    """Return job status."""
    return jobs.info(job_id)


@app.get(f"{constants.API_PREFIX}/jobs/{{job_id}}/result")
async def get_job_result(job_id: str):
    """Return completed job result."""
    return jobs.get_result(job_id)


@app.websocket(f"{constants.API_PREFIX}/jobs/{{job_id}}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    """Stream job events to the frontend."""
    await websocket.accept()
    queue = await jobs.subscribe(job_id)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        jobs.unsubscribe(job_id, queue)
