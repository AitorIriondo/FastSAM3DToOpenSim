"""
FastSAM3D Service — HTTP API

Wraps demo_video_opensim.py as an async REST service.
The SAM3D model is loaded once at startup; inference jobs are queued
and executed sequentially by a single GPU worker thread.

Usage (inside container):
    uvicorn service.main:app --host 0.0.0.0 --port 8000 --workers 1

Endpoints:
    POST   /api/v1/process          — submit a video for inference
    GET    /api/v1/status/{job_id}  — check job progress
    GET    /api/v1/results/{job_id} — list output files
    GET    /api/v1/download/{job_id}/{filename} — download a file
    DELETE /api/v1/results/{job_id} — delete job and files
    GET    /api/v1/health           — service health check
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from service.job_manager import JobManager, JobStatus

logger = logging.getLogger(__name__)

# ── Startup / shutdown ─────────────────────────────────────────────────────────

_manager: Optional[JobManager] = None
_model_loaded: bool = False
_startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager, _model_loaded
    logger.info("FastSAM3D Service starting up…")
    _manager = JobManager()
    # Model warm-up is done inside the worker process (demo_video_opensim.py
    # loads the model on first call).  We mark the service ready immediately.
    _model_loaded = True
    logger.info("Service ready — GPU worker started")
    yield
    logger.info("Service shutting down")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FastSAM3D Service",
    description=(
        "REST API for the FastSAM3DToOpenSim inference pipeline. "
        "Accepts video uploads and returns biomechanics outputs: "
        "TRC, IK MOT, GLB, MVNX (Xsens v4), and IPS MVNX."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow EasyErgoDashboard and any other caller
_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_manager() -> JobManager:
    if _manager is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return _manager


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health():
    """Service health check. Returns GPU availability and queue depth."""
    import torch
    gpu_ok = torch.cuda.is_available()
    mgr = _get_manager()
    return {
        "status": "ok",
        "model_loaded": _model_loaded,
        "gpu_available": gpu_ok,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_ok else None,
        "jobs_queued": mgr.queue_length(),
        "uptime_s": round(time.time() - _startup_time, 1),
    }


@app.post("/api/v1/process", status_code=202)
async def process(
    video: UploadFile = File(..., description="Video file to analyse"),
    person_height: float = Form(1.75, description="Subject height in metres"),
    inference_type: str = Form("body", description="'body' (fast) or 'full' (with hands)"),
    fx: Optional[float] = Form(None, description="Camera focal length in pixels (optional)"),
    no_mesh_glb: bool = Form(False, description="Skip mesh GLB export"),
    no_mvnx: bool = Form(False, description="Skip Xsens MVNX export"),
    no_ipsmvnx: bool = Form(False, description="Skip IPS MVNX export"),
):
    """
    Submit a video for inference.

    Returns immediately with a job_id. Poll /api/v1/status/{job_id} for progress.
    """
    if inference_type not in ("body", "full"):
        raise HTTPException(status_code=422, detail="inference_type must be 'body' or 'full'")

    content = await video.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty video file")

    mgr = _get_manager()
    job_id = mgr.submit(
        video_bytes=content,
        filename=video.filename or "input.mp4",
        params={
            "person_height": person_height,
            "inference_type": inference_type,
            "fx": fx,
            "no_mesh_glb": no_mesh_glb,
            "no_mvnx": no_mvnx,
            "no_ipsmvnx": no_ipsmvnx,
        },
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/v1/status/{job_id}")
async def status(job_id: str):
    """Check the processing status of a job."""
    mgr = _get_manager()
    job = mgr.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "error": job.error or None,
        "created_at": job.created_at,
        "duration_s": job.duration_s,
    }


@app.get("/api/v1/results/{job_id}")
async def results(job_id: str):
    """List output files available for a completed job."""
    mgr = _get_manager()
    job = mgr.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=409,
            detail=f"Job not complete yet (status: {job.status.value})",
        )
    files = mgr.get_output_files(job_id) or []
    return {
        "job_id": job_id,
        "status": job.status.value,
        "files": sorted(files),
        "duration_s": job.duration_s,
        "created_at": job.created_at,
    }


@app.get("/api/v1/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
    """Download a specific output file from a completed job."""
    mgr = _get_manager()
    job = mgr.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(status_code=409, detail="Job not complete yet")

    file_path = mgr.get_output_path(job_id, filename)
    if file_path is None:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )


@app.delete("/api/v1/results/{job_id}", status_code=204)
async def delete_job(job_id: str):
    """Delete a job and all its output files."""
    mgr = _get_manager()
    deleted = mgr.delete(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
