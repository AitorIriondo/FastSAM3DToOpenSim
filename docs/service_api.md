# FastSAM3D Service — API Reference

The service exposes a REST API at `http://<server>:8000`. Interactive docs are available at `http://<server>:8000/docs` (Swagger UI) once the container is running.

---

## Base URL

```
http://<server-ip>:8000/api/v1
```

---

## Endpoints

### `GET /api/v1/health`

Health check. Returns immediately.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
  "jobs_queued": 0,
  "uptime_s": 42.3
}
```

---

### `POST /api/v1/process`

Submit a video for inference. Returns immediately with a `job_id`.

**Request** (`multipart/form-data`):

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `video` | file | yes | — | Video file (mp4, avi, mov, etc.) |
| `person_height` | float | no | `1.75` | Subject height in metres |
| `inference_type` | string | no | `"body"` | `"body"` (fast, ~14 fps) or `"full"` (with hands, ~5 fps) |
| `fx` | float | no | `null` | Camera focal length in pixels (skips auto-estimation) |
| `no_mesh_glb` | bool | no | `false` | Skip full mesh GLB export |
| `no_mvnx` | bool | no | `false` | Skip Xsens MVNX export |
| `no_ipsmvnx` | bool | no | `false` | Skip IPS MVNX export |

**Response `202 Accepted`:**
```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "queued"
}
```

**curl example:**
```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "video=@recording.mp4" \
  -F "person_height=1.72" \
  -F "inference_type=body"
```

**Python example:**
```python
import requests

with open("recording.mp4", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/v1/process",
        files={"video": ("recording.mp4", f, "video/mp4")},
        data={"person_height": "1.72", "inference_type": "body"},
    )
job_id = resp.json()["job_id"]
```

---

### `GET /api/v1/status/{job_id}`

Poll job progress.

**Response:**
```json
{
  "job_id": "3fa85f64-...",
  "status": "processing",
  "progress": 45,
  "message": "Inference running…",
  "error": null,
  "created_at": 1711450000.0,
  "duration_s": null
}
```

`status` values: `queued` | `processing` | `done` | `failed`

`progress` is `0–100` (approximate; based on pipeline stages).

---

### `GET /api/v1/results/{job_id}`

List output files once a job is `done`.

**Response `200`:**
```json
{
  "job_id": "3fa85f64-...",
  "status": "done",
  "files": [
    "markers_recording.trc",
    "markers_recording_ik.mot",
    "markers_recording_model.osim",
    "markers_recording_mesh.glb",
    "markers_recording.mvnx",
    "markers_recording.ipsmvnx",
    "processing_report.json",
    "_ik_marker_errors.sto"
  ],
  "duration_s": 73.4,
  "created_at": 1711450000.0
}
```

Returns `409 Conflict` if the job is not yet complete.

---

### `GET /api/v1/download/{job_id}/{filename}`

Download a single output file.

```bash
# Download TRC
curl -O http://localhost:8000/api/v1/download/3fa85f64-.../markers_recording.trc

# Download MVNX
curl -O http://localhost:8000/api/v1/download/3fa85f64-.../markers_recording.mvnx

# Download IPS MVNX
curl -O http://localhost:8000/api/v1/download/3fa85f64-.../markers_recording.ipsmvnx
```

**Python example (download all files):**
```python
import requests, os

result = requests.get(f"http://localhost:8000/api/v1/results/{job_id}").json()
for filename in result["files"]:
    data = requests.get(
        f"http://localhost:8000/api/v1/download/{job_id}/{filename}"
    ).content
    with open(filename, "wb") as f:
        f.write(data)
```

---

### `DELETE /api/v1/results/{job_id}`

Delete a job and all its output files. Returns `204 No Content`.

```bash
curl -X DELETE http://localhost:8000/api/v1/results/3fa85f64-...
```

---

## Output file formats

| Extension | Format | Contents |
|-----------|--------|----------|
| `.trc` | OpenSim TRC | Marker trajectories (mm, Y-up) |
| `_ik.mot` | OpenSim MOT | IK joint angles (degrees, 40 DOF) |
| `_model.osim` | OpenSim model | Scaled Pose2Sim Wholebody model |
| `_mesh.glb` | glTF 2.0 | Animated full-body mesh (morph targets) |
| `.mvnx` | Xsens MVN v4 | Positions + orientations + joint angles (23 segments) |
| `.ipsmvnx` | IPS MVNX | 73-segment positions for IPS IMMA ergonomics software |
| `_ik_marker_errors.sto` | OpenSim storage | IK marker residuals per frame |
| `processing_report.json` | JSON | Timings, frame counts, output paths |

---

## Typical integration flow (EasyErgoDashboard)

```javascript
// 1. Submit
const form = new FormData();
form.append('video', videoFile);
form.append('person_height', '1.75');
const { job_id } = await fetch('/api/v1/process', { method: 'POST', body: form }).then(r => r.json());

// 2. Poll
let status;
do {
  await new Promise(r => setTimeout(r, 5000));
  status = await fetch(`/api/v1/status/${job_id}`).then(r => r.json());
} while (status.status === 'queued' || status.status === 'processing');

// 3. Download
const { files } = await fetch(`/api/v1/results/${job_id}`).then(r => r.json());
const mvnxUrl = `/api/v1/download/${job_id}/markers_${videoName}.mvnx`;
```

---

## Error codes

| Code | Meaning |
|------|---------|
| 400 | Empty or invalid video file |
| 404 | Job not found |
| 409 | Job not complete yet |
| 422 | Invalid parameter value |
| 503 | Service not initialised |

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTSAM_MODE` | `service` | `service` = HTTP API; `cli` = pass-through |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `JOB_TTL_HOURS` | `24` | Hours to keep job outputs before auto-delete |
| `JOBS_OUTPUT_DIR` | `/app/job_outputs` | Root directory for per-job output folders |
