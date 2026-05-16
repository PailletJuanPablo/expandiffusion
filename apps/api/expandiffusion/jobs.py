"""In-memory generation job store."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from . import constants
from .errors import AppError
from .persistence import PersistenceStore
from .schemas import JobInfo, JobResult, OutpaintRequest


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class JobRecord:
    """Mutable internal job record."""

    id: str
    request: OutpaintRequest
    status: str = constants.JOB_QUEUED
    progress: float = 0.0
    message: str = "Queued"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    error: str | None = None
    result: JobResult | None = None
    cancel_requested: bool = False
    subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)


class JobStore:
    """Small in-memory job store with websocket subscriptions."""

    def __init__(self, persistence: PersistenceStore) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self.persistence = persistence

    def create(self, request: OutpaintRequest) -> JobRecord:
        """Create and store a queued job."""
        job = JobRecord(id=uuid4().hex, request=request)
        self._jobs[job.id] = job
        self.persistence.record_generation_created(job)
        return job

    def get(self, job_id: str) -> JobRecord:
        """Return a job or raise a typed not-found error."""
        job = self._jobs.get(job_id)
        if job is None:
            raise AppError(
                constants.ERROR_JOB_NOT_FOUND,
                f"Job '{job_id}' was not found.",
                status_code=404,
            )
        return job

    def info(self, job_id: str) -> JobInfo:
        """Return public job info."""
        return self.to_info(self.get(job_id))

    def to_info(self, job: JobRecord) -> JobInfo:
        """Convert a job record to a public model."""
        return JobInfo(
            id=job.id,
            status=job.status,
            progress=job.progress,
            message=job.message,
            adapter_id=job.request.adapter_id,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=job.error,
            result_count=len(job.result.images) if job.result else 0,
        )

    async def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to job events."""
        job = self.get(job_id)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        job.subscribers.add(queue)
        await queue.put(self._event(job, constants.EVENT_JOB_UPDATE))
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a websocket subscriber."""
        job = self._jobs.get(job_id)
        if job is not None:
            job.subscribers.discard(queue)

    async def mark_running(self, job: JobRecord) -> None:
        """Mark a job as running and notify subscribers."""
        job.status = constants.JOB_RUNNING
        job.message = "Running"
        job.progress = constants.PROGRESS_STARTED
        job.updated_at = utc_now()
        self.persistence.record_generation_updated(job)
        await self.publish(job, constants.EVENT_JOB_UPDATE)

    async def update(self, job: JobRecord, progress: float, message: str) -> None:
        """Update job progress."""
        job.progress = max(0.0, min(constants.PROGRESS_FINISHED, progress))
        job.message = message
        job.updated_at = utc_now()
        await self.publish(job, constants.EVENT_JOB_UPDATE)

    async def complete(self, job: JobRecord, result: JobResult) -> None:
        """Complete a job with generated images."""
        job.status = constants.JOB_SUCCEEDED
        job.progress = constants.PROGRESS_FINISHED
        job.message = "Completed"
        job.result = result
        job.updated_at = utc_now()
        self.persistence.record_generation_updated(job)
        await self.publish(job, constants.EVENT_JOB_DONE)

    async def fail(self, job: JobRecord, message: str) -> None:
        """Fail a job."""
        job.status = constants.JOB_FAILED
        job.error = message
        job.message = message
        job.updated_at = utc_now()
        self.persistence.record_generation_updated(job)
        await self.publish(job, constants.EVENT_JOB_ERROR)

    async def cancel(self, job_id: str) -> JobInfo:
        """Request job cancellation."""
        job = self.get(job_id)
        job.cancel_requested = True
        if job.status in {constants.JOB_QUEUED, constants.JOB_RUNNING}:
            job.status = constants.JOB_CANCELLED
            job.message = "Cancellation requested"
            job.updated_at = utc_now()
            self.persistence.record_generation_updated(job)
            await self.publish(job, constants.EVENT_JOB_CANCELLED)
        return self.to_info(job)

    async def publish(self, job: JobRecord, event_type: str) -> None:
        """Publish an event to all subscribers."""
        payload = self._event(job, event_type)
        for queue in list(job.subscribers):
            await queue.put(payload)

    def get_result(self, job_id: str) -> JobResult:
        """Return a completed job result."""
        job = self.get(job_id)
        if job.result is None:
            raise AppError(
                constants.ERROR_JOB_NOT_FINISHED,
                "The job has no result yet.",
                status_code=409,
            )
        return job.result

    def _event(self, job: JobRecord, event_type: str) -> dict[str, Any]:
        return {"type": event_type, "job": self.to_info(job).model_dump()}
