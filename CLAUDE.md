# CLAUDE.md — EasyErgo FastSAM3D Service

Complete setup and environment notes for AI assistants working in this repo.
This file captures all the non-obvious things that were solved during development.

---

## Repository overview

This repo is a fork of **Fast-SAM-3D-Body** extended for EasyErgo:
- `demo_video_opensim.py` — main CLI entry point (batch inference)
- `service/` — FastAPI HTTP service (`service/main.py`, `service/job_manager.py`)
- `sam_3d_body/` — model, inference, post-processing, export
- `vendor/opensim_to_mvnx/` — vendored copy of private repo (OpenSim TRC+MOT → MVNX v4)
- `vendor/trc_to_ips/` — vendored copy of TRCtoIPS (73-marker TRC → IPS IMMA .ipsmvnx)
- `docker/` — Dockerfile, entrypoint.sh, requirements_docker.txt, trt.env
- `checkpoints/` — model weights (not in git, ~7.6 GB total)

**Remotes:**
- `origin` = `FastSAM3DToOpenSim` (upstream, do not push there)
- `easyergo` = `https://github.com/AitorIriondo/EasyErgo_FastSam3DService.git` (this service)

**Do not** add co-authors or collaborators to commits (user preference).

---

## Target hardware

| Machine | GPU | Arch | VRAM |
|---------|-----|------|------|
| Dev laptop | RTX 5090 (Laptop) | Blackwell GB202, sm_120 | 24 GB |
| Server | RTX 5070 Ti (Desktop) | Blackwell GB203, sm_120 | 16 GB |

Both are **sm_120 (Blackwell)**. They share the same compute capability but TRT
optimises for exact SM count and memory bandwidth — **always rebuild TRT engines
on the target machine**, even though technically 5090 engines might load on 5070 Ti.

Required NVIDIA driver: **≥ 570** (for CUDA 12.8 / sm_120 support).

---

## Two conda environments

The pipeline uses two separate conda environments:

| Env | Python | Purpose |
|-----|--------|---------|
| `fast_sam_3d_body` | 3.11 | Main inference, post-processing, FastAPI service |
| `opensim` | 3.10 | OpenSim 4.5 IK solver only |

**Critical:** `opensim_ik_runner.py` hard-codes the path:
```
/opt/conda/envs/opensim/bin/python
```
This is why Miniforge must be installed at `/opt/conda` (not `~/miniconda3` or similar).
In Docker this is guaranteed. For local installs, use Miniforge:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda
```

---

## Local (non-Docker) setup

```bash
# Activate main env
conda activate fast_sam_3d_body   # must be at /opt/conda/envs/fast_sam_3d_body

# Check PyTorch CUDA version — must be cu128 for sm_120
python -c "import torch; print(torch.version.cuda)"  # → 12.8

# OpenSim env is only invoked as a subprocess — never activate it manually
```

**If PyTorch shows cu124** (or any version < cu128), you're using the wrong build.
Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu128`

---

## Best CLI run command (local, TRT enabled)

```bash
SKIP_KEYPOINT_PROMPT=1 \
FOV_TRT=1 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0 \
USE_TRT_BACKBONE=1 USE_COMPILE=1 DECODER_COMPILE=1 \
COMPILE_MODE=reduce-overhead COMPILE_WARMUP_BATCH_SIZES=1 \
MHR_NO_CORRECTIVES=1 GPU_HAND_PREP=1 BODY_INTERM_PRED_LAYERS=0,2 \
DEBUG_NAN=0 PARALLEL_DECODERS=0 \
python demo_video_opensim.py \
  --input ./videos/my_video.mp4 \
  --detector yolo_pose \
  --detector_model checkpoints/yolo/yolo11m-pose.engine \
  --inference_type body \
  --person_height 1.75 \
  --floor_moge \
  --output_dir ./outputs/my_run
```

For PyTorch-only mode (no engines built yet), replace `.engine` with `.pt` and drop
all `TRT`/`COMPILE` env vars.

---

## Docker setup

### Image contents

Built on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`. Two conda envs baked in
(see above). Miniforge at `/opt/conda`.

**TRT engines are NOT in the image** — they are GPU-specific and must be compiled
by the user after first launch.

### Service mode vs CLI mode

`FASTSAM_MODE=service` → starts uvicorn (FastAPI on port 8000, default)
`FASTSAM_MODE=cli`     → pass-through to `exec "$@"` (batch inference)

```bash
# Start service (default)
docker compose up -d

# Run CLI batch job
docker compose run --rm -e FASTSAM_MODE=cli fastsam3d \
  python demo_video_opensim.py --input /app/videos/test.mp4 --person_height 1.75
```

### Build image

```bash
cd /opt/EasyErgo_FastSam3DService   # or wherever the repo lives on the server
docker compose build
# First build: ~30-45 min (CUDA + 2 conda envs + all ML packages)
# Code-only rebuild: ~5 min (only the COPY /app layer re-runs)
```

### OpenCV gotcha (SOLVED)

Several pip packages (`ultralytics`, etc.) pull in `opencv-python` from PyPI.
That PyPI build has a **broken FFMPEG backend** inside Docker — `VideoCapture`
silently fails. The Dockerfile works around this by running:
```
conda install -c conda-forge opencv
```
**after** all pip installs, which replaces the broken PyPI build with the
conda-forge version that has a working FFMPEG backend.

**Do not** change this order in the Dockerfile — putting the conda install
before pip will cause pip to reinstall the PyPI version over it.

### Symlink gotcha

Docker does not follow symlinks outside the mounted volume. If `./videos/`
contains symlinks (e.g. pointing to `~/Desktop/...`), the container sees empty
files. Replace symlinks with real copies or mount the real directory directly.

---

## TRT engine compilation (one-time, per GPU)

### Why bother

| Mode | FPS (5070 Ti, body-only) |
|------|:---:|
| PyTorch only | ~8-10 fps |
| + TRT backbone | ~14-16 fps |
| + TRT backbone + torch.compile | ~18-20 fps |

### Build all three engines inside the container

```bash
# Open a build shell (uses the server's GPU)
docker compose run --rm -e FASTSAM_MODE=cli fastsam3d bash

# Inside the container — run the convenience script:
bash /app/build_tensorrt.sh

exit
```

Or build individually:

```bash
# 1. YOLO pose engine (~2 min → ~40 MB)
python convert_yolo_pose_trt.py \
  --model checkpoints/yolo/yolo11m-pose.pt \
  --output checkpoints/yolo/yolo11m-pose.engine \
  --imgsz 640 --half

# 2. MoGe FOV estimator engine (~5 min → ~46 MB)
python convert_moge_encoder_trt.py \
  --output checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine

# 3. DINOv3 backbone engine (~20 min → ~1.6 GB)
python convert_backbone_tensorrt.py \
  --checkpoint checkpoints/sam-3d-body-dinov3 \
  --output checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine
```

**Total: ~30 min. Engines persist in `./checkpoints/` via Docker volume mount.**

### Enable TRT in service mode

Use the provided env file:
```bash
docker compose --env-file docker/trt.env up -d
```

`docker/trt.env` sets:
```
FASTSAM_MODE=service
FOV_TRT=1
USE_TRT_BACKBONE=1
USE_COMPILE=1
DECODER_COMPILE=1
COMPILE_MODE=reduce-overhead
COMPILE_WARMUP_BATCH_SIZES=1
```

### When to rebuild engines

Rebuild after: TensorRT version upgrade, GPU change, PyTorch major version bump,
backbone architecture change.

Do NOT rebuild after: Python code changes, API changes, doc updates.

### TRT build error: "unsupported SM"

Your TensorRT version may not support sm_120 yet.
Requires: **CUDA 12.8 + TRT 10.16** for Blackwell (sm_120).
Check inside container: `python -c "import tensorrt; print(tensorrt.__version__)"`

---

## Service startup and health check

```bash
# Start
docker compose up -d

# Watch logs — model takes ~30s to load
docker compose logs -f

# Health check
curl http://localhost:8000/api/v1/health
# → {"status": "ok", "model_loaded": true, "gpu_available": true, "jobs_queued": 0}
```

The service returns `"model_loaded": false` until the SAM3D model finishes loading.
EasyErgoDashboard should poll health until `model_loaded` is true before submitting jobs.

### Firewall

The service runs on port 8000. Open it for the dashboard server:
```bash
sudo ufw allow from <dashboard-ip> to any port 8000
```

### Auto-start on reboot

```bash
sudo systemctl enable docker
docker compose up -d   # run once; restart: unless-stopped handles reboots
```

---

## MVNX export details

### Normal MVNX (`vendor/opensim_to_mvnx`)

- Input: TRC file + IK MOT file + optional .osim model path
- Output: Xsens MVN Studio v4 MVNX with 23 Xsens segments, quaternion orientations,
  joint angles, and positions
- Requires successful OpenSim IK run (skipped if IK fails)
- The vendor code was copied from private repo `OpenSim-to-MVNX-master` and all
  imports were converted from absolute (`from src.xxx`) to relative (`from .xxx`,
  `from ..utils.xxx`)

### IPS MVNX (`vendor/trc_to_ips`)

- Input: 70 MHR70 keypoints (kpts_opensim) + 127-joint armature (jcoords_opensim)
- Output: `.ipsmvnx` (IPS IMMA format, positions only, Z-up)
- Always generated even in body-only mode — all 70 MHR70 keypoints are predicted
  (fingers have lower accuracy in body mode, but positions are present)
- SpineMid marker: derived as mean of `jcoords[35]` (c_spine1) and `jcoords[36]`
  (c_spine2) from the 127-joint armature. Fallback: mean of LElbow + RElbow.
- All vendor imports are relative

---

## Known issues / limitations

1. **Forward lean**: The spine-based lean correction (`--lean_fix`, default on)
   partially fixes forward lean caused by camera pitch. Experimental camera-pitch
   method (`--lean_cam_pitch_fix`) may over-correct. Not fully solved.

2. **Hand orientation in T-pose**: Hands appear rotated when the subject is
   in a T-pose. Doesn't affect body-only inference significantly.

3. **IK accuracy**: Typical RMS ~53 mm on aitor_garden_walk (1136 frames, outdoor).
   This is driven by the 40-DOF Pose2Sim_Simple model constraints. Not a bug.

4. **OOM on 16 GB VRAM**: If you hit GPU OOM on the 5070 Ti:
   - Reduce batch sizes
   - Add `--no_mesh_glb` to skip mesh GLB (biggest VRAM spike)
   - Disable `PARALLEL_DECODERS`

5. **Blender GLB export**: Requires `blender` installed on the host (or in Docker).
   The Docker image includes Blender. For local installs: `sudo apt install blender`
   and `pip3.12 install numpy --break-system-packages`.

---

## Checkpoints

~7.6 GB total, not in git. Location: `./checkpoints/`

```
checkpoints/
├── sam-3d-body-dinov3/           # SAM3D backbone (~1.6 GB)
│   ├── model.ckpt
│   ├── assets/mhr_model.pt
│   └── backbone_trt/
│       └── backbone_dinov3_fp16.engine  (built on-site)
├── yolo/
│   ├── yolo11m-pose.pt           (~40 MB)
│   └── yolo11m-pose.engine       (built on-site)
└── moge_trt/
    └── moge_dinov2_encoder_fp16.engine  (built on-site)
```

Transfer to a new machine:
```bash
rsync -av --exclude='*.engine' \
  user@source:/opt/EasyErgo_FastSam3DService/checkpoints/ ./checkpoints/
```

Then rebuild engines on the target GPU.

---

## Key environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `FASTSAM_MODE` | `service` | `service` = uvicorn, `cli` = pass-through |
| `FOV_TRT` | 0 | Use TRT MoGe FOV estimator engine |
| `USE_TRT_BACKBONE` | 0 | Use TRT DINOv3 backbone engine |
| `USE_COMPILE` | 0 | Enable `torch.compile` on transformer |
| `DECODER_COMPILE` | 0 | Enable `torch.compile` on decoder |
| `COMPILE_MODE` | `default` | `reduce-overhead` for best throughput |
| `COMPILE_WARMUP_BATCH_SIZES` | 1 | Batch sizes to warm up compiled model |
| `SKIP_KEYPOINT_PROMPT` | 0 | Skip SAM2 keypoint prompting (faster) |
| `FOV_FAST` | 0 | Use fast (small) MoGe model for FOV |
| `FOV_MODEL` | `l` | MoGe model size: `s`, `b`, `l` |
| `FOV_LEVEL` | 2 | MoGe inference level (0=fastest) |
| `MHR_NO_CORRECTIVES` | 0 | Skip corrective blend shapes (faster) |
| `GPU_HAND_PREP` | 0 | Run hand preprocessing on GPU |
| `BODY_INTERM_PRED_LAYERS` | all | Which transformer layers to use: `0,2` = faster |
| `PARALLEL_DECODERS` | 0 | Run body+hand decoders in parallel |
| `DEBUG_NAN` | 0 | Raise on NaN instead of silently continuing |
| `CORS_ORIGINS` | `*` | Allowed origins for FastAPI CORS |
| `JOBS_OUTPUT_DIR` | `/app/job_outputs` | Where service stores job outputs |
| `JOB_TTL_HOURS` | `24` | How long to keep job outputs before cleanup |

See [SETTINGS.md](SETTINGS.md) for the full list with measured performance impact.
