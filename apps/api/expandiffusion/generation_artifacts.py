"""Debug artifact capture for generation jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from . import constants


class GenerationArtifactRecorder:
    """Persist the exact intermediate images and metadata for a generation job."""

    def __init__(self, root: Path = constants.DEFAULT_GENERATION_ARTIFACT_DIR) -> None:
        self.root = root

    def job_dir(self, job_id: str) -> Path:
        """Return and create the artifact folder for a job."""
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_image(self, job_id: str, filename: str, image: Image.Image) -> None:
        """Save a PIL image artifact for a job."""
        image.save(self.job_dir(job_id) / filename)

    def save_json(self, job_id: str, filename: str, payload: dict[str, Any]) -> None:
        """Save a JSON artifact for a job."""
        path = self.job_dir(job_id) / filename
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
