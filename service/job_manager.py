"""
In-memory job manager for the FastSAM3D service.

One GPU worker thread processes jobs sequentially (GPU is exclusive).
Jobs are kept in memory for JOB_TTL_HOURS hours then auto-deleted.
"""

import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# How long to keep completed/failed job outputs on disk (default 24 h)
JOB_TTL_SECONDS = float(os.getenv("JOB_TTL_HOURS", "24")) * 3600

# Root directory for per-job output folders
JOBS_ROOT = Path(os.getenv("JOBS_OUTPUT_DIR", "/app/job_outputs"))


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobState:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    output_dir: Optional[Path] = None
    progress: int = 0          # 0-100
    message: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return round(self.finished_at - self.started_at, 1)
        return None


class JobManager:
    """Thread-safe job queue with a single GPU worker."""

    def __init__(self, estimator=None):
        self._jobs: Dict[str, JobState] = {}
        self._queue: List[str] = []
        self._lock = threading.Lock()
        self._estimator = estimator   # set after model warm-up
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        # Background cleanup
        self._cleaner = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleaner.start()

    def set_estimator(self, estimator):
        """Called after model warm-up to inject the loaded estimator."""
        self._estimator = estimator

    # ── Public API ─────────────────────────────────────────────────────────────

    def submit(self, video_bytes: bytes, filename: str, params: dict) -> str:
        """
        Queue a new inference job.

        Args:
            video_bytes: Raw video file bytes
            filename: Original video filename (used to name outputs)
            params: Dict with keys: person_height, inference_type, fx,
                    no_mvnx, no_ipsmvnx, no_mesh_glb

        Returns:
            job_id string
        """
        job_id = str(uuid.uuid4())
        output_dir = JOBS_ROOT / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write video to job directory
        video_path = output_dir / filename
        video_path.write_bytes(video_bytes)

        job = JobState(job_id=job_id, output_dir=output_dir)
        job.message = "Waiting for GPU worker"

        # Store params alongside the job
        with open(output_dir / "_params.json", "w") as f:
            import json
            json.dump({"video_path": str(video_path), **params}, f)

        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job_id)

        logger.info("Job %s queued (queue length: %d)", job_id, len(self._queue))
        return job_id

    def get_status(self, job_id: str) -> Optional[JobState]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_output_files(self, job_id: str) -> Optional[List[str]]:
        job = self.get_status(job_id)
        if job is None or job.output_dir is None:
            return None
        if job.status != JobStatus.DONE:
            return None
        # List all output files (exclude internal files starting with _)
        return [
            f.name for f in job.output_dir.iterdir()
            if f.is_file() and not f.name.startswith("_") and not f.suffix == ".mp4"
            or f.suffix in {".trc", ".mot", ".osim", ".glb", ".mvnx", ".ipsmvnx", ".json", ".sto", ".mp4"}
        ]

    def get_output_path(self, job_id: str, filename: str) -> Optional[Path]:
        job = self.get_status(job_id)
        if job is None or job.output_dir is None:
            return None
        path = job.output_dir / filename
        return path if path.exists() else None

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job_id in self._queue:
                self._queue.remove(job_id)
        if job and job.output_dir and job.output_dir.exists():
            shutil.rmtree(job.output_dir, ignore_errors=True)
        return job is not None

    def queue_length(self) -> int:
        with self._lock:
            return sum(
                1 for jid in self._queue
                if self._jobs.get(jid, JobState("")).status == JobStatus.QUEUED
            )

    # ── Worker loop ────────────────────────────────────────────────────────────

    def _worker_loop(self):
        while True:
            job_id = self._pop_next()
            if job_id is None:
                time.sleep(0.5)
                continue
            self._process(job_id)

    def _pop_next(self) -> Optional[str]:
        with self._lock:
            for jid in self._queue:
                job = self._jobs.get(jid)
                if job and job.status == JobStatus.QUEUED:
                    return jid
        return None

    def _process(self, job_id: str):
        import json
        import subprocess
        import sys

        job = self._jobs.get(job_id)
        if job is None:
            return

        with self._lock:
            job.status = JobStatus.PROCESSING
            job.started_at = time.time()
            job.message = "Running inference…"
            job.progress = 5

        try:
            params_path = job.output_dir / "_params.json"
            with open(params_path) as f:
                params = json.load(f)

            video_path = params["video_path"]
            output_dir = str(job.output_dir)

            cmd = [
                sys.executable,
                "/app/demo_video_opensim.py",
                "--input", video_path,
                "--output_dir", output_dir,
                "--person_height", str(params.get("person_height", 1.75)),
                "--inference_type", str(params.get("inference_type", "body")),
            ]
            if params.get("fx"):
                cmd += ["--fx", str(params["fx"])]
            if params.get("no_mesh_glb"):
                cmd.append("--no_mesh_glb")
            if params.get("no_mvnx"):
                cmd.append("--no_mvnx")
            if params.get("no_ipsmvnx"):
                cmd.append("--no_ipsmvnx")

            # Env: forward TRT / compile flags from process environment
            env = os.environ.copy()

            logger.info("Job %s starting: %s", job_id, " ".join(cmd))

            with self._lock:
                job.progress = 10
                job.message = "Inference running…"

            result = subprocess.run(
                cmd,
                capture_output=False,
                env=env,
                timeout=3600,   # 1-hour hard cap
            )

            if result.returncode != 0:
                raise RuntimeError(f"demo_video_opensim.py exited with code {result.returncode}")

            with self._lock:
                job.status = JobStatus.DONE
                job.progress = 100
                job.finished_at = time.time()
                job.message = "Done"

            logger.info("Job %s completed in %.1f s", job_id, job.duration_s or 0)

        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            with self._lock:
                job.status = JobStatus.FAILED
                job.finished_at = time.time()
                job.error = str(exc)
                job.message = f"Failed: {exc}"

    # ── Cleanup loop ───────────────────────────────────────────────────────────

    def _cleanup_loop(self):
        while True:
            time.sleep(3600)   # check hourly
            now = time.time()
            expired = []
            with self._lock:
                for jid, job in self._jobs.items():
                    if job.status in (JobStatus.DONE, JobStatus.FAILED):
                        age = now - (job.finished_at or job.created_at)
                        if age > JOB_TTL_SECONDS:
                            expired.append(jid)
            for jid in expired:
                logger.info("Cleaning up expired job %s", jid)
                self.delete(jid)
