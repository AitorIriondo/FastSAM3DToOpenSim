# Docker Guide — FastSAM3DToOpenSim

## What the image contains

Two conda environments built on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`:

| Env | Python | Key packages |
|---|---|---|
| `fast_sam_3d_body` | 3.11 | PyTorch 2.10+cu128, TensorRT 10.16, MoGe2, detectron2 |
| `opensim` | 3.10 | opensim (opensim-org channel) |

Miniforge lives at `/opt/conda` — required because `opensim_ik_runner.py` hard-codes
`/opt/conda/envs/opensim/bin/python`.

The default entrypoint activates `fast_sam_3d_body` and sets `PYTHONPATH=/app`.

**TRT engines are NOT baked into the image** — they are GPU-architecture-specific
and must be built by the user after first launch (see below).

---

## Files

```
Dockerfile
docker-compose.yml
.dockerignore
docker/entrypoint.sh       ← activates env, sets PYTHONPATH=/app
docker/requirements_docker.txt  ← pip deps (opencv via conda-forge, chumpy/detectron2 special flags)
docker/trt.env             ← env-var overrides to enable TRT after engines are built
```

---

## Known gotchas

### Symlinks in `./videos/`
Docker does **not** follow symlinks out of a mounted volume.
If your `videos/` folder contains symlinks (e.g. pointing to `~/Desktop/…`), the
container will see an empty file. Either:
- Replace symlinks with real files: `cp --remove-destination $(readlink -f link) link`
- Or mount the real directory directly: `-v /real/path/to/videos:/app/videos:ro`

### opencv-python from pip vs conda
Several pip packages (`ultralytics` etc.) pull in `opencv-python` as a dependency.
That PyPI build has a broken FFMPEG backend in Docker (VideoCapture silently fails).
The Dockerfile works around this by running `conda install opencv` from conda-forge
**after** all pip installs, replacing the broken one.

### `version` warning in docker-compose.yml
Docker Compose v2+ shows a harmless warning about the `version` attribute. Ignore it.

---

## How to rebuild after code changes

```bash
cd /home/linuxaitor/FastSam3dBody/repo

# Rebuild image (pip/conda layers cached; only changed layers re-run)
sudo docker compose build

# If you changed Dockerfile or docker/requirements_docker.txt, force full rebuild:
sudo docker compose build --no-cache
```

---

## Running the pipeline (PyTorch, no TRT)

```bash
cd /home/linuxaitor/FastSam3dBody/repo

sudo docker compose \
  --project-directory /home/linuxaitor/FastSam3dBody/repo \
  run --rm \
  -v /path/to/real/videos:/app/videos:ro \
  -e SKIP_KEYPOINT_PROMPT=1 \
  -e FOV_FAST=1 -e FOV_MODEL=s -e FOV_LEVEL=0 \
  -e MHR_NO_CORRECTIVES=1 -e GPU_HAND_PREP=1 \
  -e BODY_INTERM_PRED_LAYERS=0,2 \
  -e DEBUG_NAN=0 -e PARALLEL_DECODERS=0 \
  fastsam3d \
  python demo_video_opensim.py \
    --input /app/videos/my_video.mp4 \
    --detector yolo_pose \
    --detector_model checkpoints/yolo/yolo11m-pose.pt \
    --person_height 1.70 \
    --floor_moge \
    --output_dir /outputs/my_run
```

Outputs appear in `./outputs/my_run/` on the host.

**Verified:** 1136 frames, aitor_garden_walk.mp4 → **9.38 fps**, ~2 min 21s total
(model load 15.8s + MoGe floor 4.1s + inference 2 min).

---

## Setting up on a new machine (e.g. RTX 5070 Ti desktop)

### Prerequisites

```bash
# 1. Install Docker Engine
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# 2. Install nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ct.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-ct.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. Verify GPU is visible
docker run --rm --runtime=nvidia nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

### Transfer the code

```bash
# Option A: git clone
git clone https://github.com/AitorIriondo/FastSAM3DToOpenSim.git
cd FastSAM3DToOpenSim

# Option B: rsync from the development machine
rsync -av --exclude='outputs/' --exclude='*.engine' \
  aitor@source:/home/linuxaitor/FastSam3dBody/repo/ ./
```

### Transfer checkpoints

```bash
# Checkpoints are ~7.6 GB — copy the whole folder
rsync -av aitor@source:/home/linuxaitor/FastSam3dBody/repo/checkpoints/ ./checkpoints/
```

**Do NOT copy `.engine` files** — they are compiled for the source GPU (sm_120 RTX 5090)
and while technically compatible with the 5070 Ti (same sm_120), it's better to
recompile them on the target GPU for optimal performance.

### Build the image on the 5070 Ti

```bash
cd /path/to/FastSAM3DToOpenSim
sudo docker compose build
# ~20–40 min first time (downloads PyTorch, TensorRT, conda packages)
```

### Build TRT engines inside the container

```bash
# Open an interactive shell in the container
sudo docker compose \
  --project-directory /path/to/FastSAM3DToOpenSim \
  run --rm fastsam3d bash

# Inside the container:
# 1. YOLO engine
python convert_yolo_pose_trt.py

# 2. DINOv3 backbone engine
python convert_backbone_tensorrt.py

# 3. MoGe FOV estimator engine
python convert_moge_encoder_trt.py

exit
```

Engines appear in `checkpoints/` on the host (via volume mount). They only need to
be built once per GPU.

### Run with TRT enabled

```bash
sudo docker compose \
  --project-directory /path/to/FastSAM3DToOpenSim \
  run --rm \
  --env-file docker/trt.env \
  -v /path/to/videos:/app/videos:ro \
  fastsam3d \
  python demo_video_opensim.py \
    --input /app/videos/my_video.mp4 \
    --detector yolo_pose \
    --detector_model checkpoints/yolo/yolo11m-pose.engine \
    --person_height 1.70 \
    --floor_moge \
    --output_dir /outputs/my_run
```

`docker/trt.env` sets:
```
FOV_TRT=1
USE_TRT_BACKBONE=1
USE_COMPILE=1
DECODER_COMPILE=1
COMPILE_MODE=reduce-overhead
COMPILE_WARMUP_BATCH_SIZES=1
```

### Expected performance (5070 Ti, TRT)

The 5070 Ti has 16 GB VRAM vs 24 GB on the RTX 5090 Laptop. The pipeline runs
comfortably within 16 GB at standard settings. Expected fps with TRT: ~15–20 fps
(similar to the 5090 since both are sm_120 and the bottleneck is the decoder,
not the backbone).

If you hit OOM:
- Reduce batch sizes
- Add `--no_mesh_glb` to skip mesh GLB export

---

## Verifying the setup

```bash
# Check both conda envs are present
sudo docker compose run --rm fastsam3d bash -c \
  "conda env list && /opt/conda/envs/opensim/bin/python -c 'import opensim; print(opensim.__version__)'"

# Check GPU is visible
sudo docker compose run --rm fastsam3d python -c \
  "import torch; print(torch.cuda.get_device_name(0))"

# Quick pipeline smoke test (PyTorch, no TRT)
sudo docker compose \
  --project-directory /path/to/FastSAM3DToOpenSim \
  run --rm \
  -v /path/to/videos:/app/videos:ro \
  -e SKIP_KEYPOINT_PROMPT=1 -e FOV_FAST=1 -e FOV_MODEL=s -e FOV_LEVEL=0 \
  -e MHR_NO_CORRECTIVES=1 -e GPU_HAND_PREP=1 -e BODY_INTERM_PRED_LAYERS=0,2 \
  -e DEBUG_NAN=0 -e PARALLEL_DECODERS=0 \
  fastsam3d \
  python demo_video_opensim.py \
    --input /app/videos/my_video.mp4 \
    --detector yolo_pose \
    --detector_model checkpoints/yolo/yolo11m-pose.pt \
    --person_height 1.70 \
    --output_dir /outputs/smoke_test
```
