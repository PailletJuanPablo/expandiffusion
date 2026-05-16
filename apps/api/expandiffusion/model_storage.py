"""Local model download storage."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import urlopen

from . import constants
from .errors import AppError

HTTP_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
FILENAME_PATTERN = re.compile(r'filename="?([^";]+)"?', re.IGNORECASE)
CIVITAI_HOSTS = {"civitai.com", "www.civitai.com"}
DownloadProgressCallback = Callable[[int, int | None, str], None]
DownloadCancelCheck = Callable[[], bool]


class ModelStorage:
    """Download direct model URLs into the local model storage directory."""

    def __init__(self, directory: Path = constants.DEFAULT_MODEL_STORAGE_DIR) -> None:
        self.directory = directory

    def resolve_url(
        self,
        url: str,
        progress: DownloadProgressCallback | None = None,
        is_cancelled: DownloadCancelCheck | None = None,
    ) -> Path:
        """Return a local file path for a direct model URL, downloading it if needed."""
        download_url = _model_download_url(url)
        parsed = urlparse(download_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise AppError(
                constants.ERROR_MODEL_DOWNLOAD_FAILED,
                "Model URL must be an http or https direct file URL.",
                status_code=422,
            )

        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / self._filename_for_url(download_url)
        if target.exists() and target.stat().st_size > 0:
            _report_download_progress(
                progress,
                target.stat().st_size,
                target.stat().st_size,
                target,
            )
            return target

        temp_target = target.with_suffix(f"{target.suffix}.tmp")
        try:
            _raise_if_cancelled(is_cancelled)
            with urlopen(download_url, timeout=HTTP_TIMEOUT_SECONDS) as response:
                header_filename = self._filename_from_headers(
                    response.headers.get("Content-Disposition")
                )
                if header_filename:
                    target = self.directory / self._filename_for_url(
                        download_url,
                        header_filename,
                    )
                    if target.exists() and target.stat().st_size > 0:
                        _report_download_progress(
                            progress,
                            target.stat().st_size,
                            target.stat().st_size,
                            target,
                        )
                        return target
                    temp_target = target.with_suffix(f"{target.suffix}.tmp")
                total_bytes = _content_length(response.headers.get("Content-Length"))
                downloaded_bytes = 0
                _report_download_progress(progress, downloaded_bytes, total_bytes, target)
                with temp_target.open("wb") as output:
                    while True:
                        _raise_if_cancelled(is_cancelled)
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded_bytes += len(chunk)
                        _report_download_progress(progress, downloaded_bytes, total_bytes, target)
                        _raise_if_cancelled(is_cancelled)
            if temp_target.stat().st_size == 0:
                temp_target.unlink(missing_ok=True)
                raise AppError(
                    constants.ERROR_MODEL_DOWNLOAD_FAILED,
                    "Downloaded model file is empty.",
                    status_code=502,
                )
            temp_target.replace(target)
            return target
        except AppError:
            temp_target.unlink(missing_ok=True)
            raise
        except (OSError, URLError) as exc:
            temp_target.unlink(missing_ok=True)
            raise AppError(
                constants.ERROR_MODEL_DOWNLOAD_FAILED,
                "Failed to download model URL.",
                status_code=502,
                details={"reason": str(exc)},
            ) from exc

    def _filename_for_url(self, url: str, filename: str | None = None) -> str:
        source_name = filename or Path(unquote(urlparse(url).path)).name or "model.safetensors"
        source_path = Path(source_name)
        stem = self._sanitize_filename(source_path.stem or "model")
        suffix = self._sanitize_suffix(source_path.suffix)
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        return f"{stem}-{digest}{suffix}"

    def _sanitize_filename(self, value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
        return sanitized[:80] or "model"

    def _sanitize_suffix(self, value: str) -> str:
        suffix = re.sub(r"[^A-Za-z0-9.]+", "", value)
        return suffix[:20] if suffix.startswith(".") else ".safetensors"

    def _filename_from_headers(self, content_disposition: str | None) -> str | None:
        if not content_disposition:
            return None
        match = FILENAME_PATTERN.search(content_disposition)
        return match.group(1) if match else None


def _content_length(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _model_download_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":", maxsplit=1)[0]
    if host not in CIVITAI_HOSTS or not parsed.path.startswith("/models/"):
        return url

    model_version_id = parse_qs(parsed.query).get("modelVersionId", [None])[0]
    if model_version_id is None or not model_version_id.isdigit():
        return url
    return f"{parsed.scheme}://{parsed.netloc}/api/download/models/{model_version_id}"


def _report_download_progress(
    progress: DownloadProgressCallback | None,
    bytes_done: int,
    bytes_total: int | None,
    target: Path,
) -> None:
    if progress is not None:
        progress(bytes_done, bytes_total, target.name)


def _raise_if_cancelled(is_cancelled: DownloadCancelCheck | None) -> None:
    if is_cancelled is not None and is_cancelled():
        raise AppError(
            constants.ERROR_MODEL_LOAD_CANCELLED,
            "Model load cancelled.",
            status_code=409,
        )
