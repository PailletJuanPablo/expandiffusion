"""Typed application errors."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AppError(Exception):
    """Exception that carries a stable API error code."""

    code: str
    message: str
    status_code: int = 400
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class GenerationCancelled(Exception):
    """Raised by adapters when a running generation is cancelled."""

