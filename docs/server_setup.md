# Server Setup Guide — RTX 5070 Ti Desktop

This guide covers deploying the FastSAM3D service on a Linux server with an NVIDIA RTX 5070 Ti.

---

## Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| OS | Ubuntu 22.04 LTS | Other distros work; package names may differ |
| NVIDIA driver | ≥ 570 | Required for CUDA 12.8 (Blackwell sm_120) |
| Docker | ≥ 24 | |
| nvidia-container-toolkit | latest | Allows containers to access the GPU |
| Disk space | ~50 GB | ~15 GB image + ~30 GB checkpoints + outputs |

---

## Step 1 — NVIDIA driver

```bash
# Check current driver
nvidia-smi

# If missing or < 570, install:
sudo apt-get install -y nvidia-driver-570
sudo reboot

# Verify
nvidia-smi   # should show Driver Version: 570.x, CUDA Version: 12.8
```

---

## Step 2 — Docker + nvidia-container-toolkit

```bash
# Install Docker (if not present)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

---

## Step 3 — Clone the repo and set up checkpoints

```bash
cd /opt
git clone https://github.com/AitorIriondo/EasyErgo_FastSam3DService.git
cd EasyErgo_FastSam3DService

# Create checkpoints directory and download model weights
mkdir -p checkpoints/sam-3d-body-dinov3 checkpoints/yolo checkpoints/moge_trt
# Follow SETUP.md for checkpoint download instructions
```

---

## Step 4 — Build the Docker image

The image installs all Python dependencies (conda envs, PyTorch cu128, OpenSim).
This takes ~30–45 minutes on first build.

```bash
cd /opt/EasyErgo_FastSam3DService
docker compose build
```

After building, the image is ~15 GB. Only the Python layer changes when code is
updated — subsequent rebuilds are much faster (~5 min).

---

## Step 5 — Build TRT engines (one-time, GPU-specific)

TRT engines must be compiled for the specific GPU on the server.
The RTX 5070 Ti is Blackwell sm_120 — the same architecture as the RTX 5090.

> **Note:** Even if you have engine files from an RTX 5090, rebuild them on the 5070 Ti.
> While sm_120 is the same compute capability, TRT optimises for the exact SM count
> and memory bandwidth of the specific chip.

```bash
# Run the build script inside the container (uses the server's GPU)
docker compose run --rm \
  -e FASTSAM_MODE=cli \
  fastsam3d \
  bash /app/build_tensorrt.sh
```

This builds three engines (~30 min total):
- `checkpoints/yolo/yolo11m-pose.engine` (~40 MB)
- `checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine` (~46 MB)
- `checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine` (~1.6 GB)

See [trt_compilation.md](trt_compilation.md) for detailed TRT build instructions.

---

## Step 6 — Start the service

```bash
# PyTorch-only mode (no TRT, ~8 fps on 5070 Ti)
docker compose up -d

# TRT mode (~18-20 fps on 5070 Ti) — after building engines
docker compose --env-file docker/trt.env up -d
```

---

## Step 7 — Verify the service is running

```bash
# Check container logs
docker compose logs -f

# Health check
curl http://localhost:8000/api/v1/health
# Expected: {"status": "ok", "model_loaded": true, "gpu_available": true, ...}

# Test with a short video
curl -X POST http://localhost:8000/api/v1/process \
  -F "video=@/opt/EasyErgo_FastSam3DService/videos/test.mp4" \
  -F "person_height=1.75"
```

---

## Step 8 — Keep the service running across reboots

```bash
# The docker-compose.yml already has: restart: unless-stopped
# Enable Docker to start on boot:
sudo systemctl enable docker

# To start the service automatically on boot:
cd /opt/EasyErgo_FastSam3DService
docker compose up -d    # run this once; restart: unless-stopped handles reboots
```

---

## Updating the service

```bash
cd /opt/EasyErgo_FastSam3DService
git pull origin main

# Rebuild only the code layer (fast, ~5 min)
docker compose build --no-cache=false
docker compose up -d --force-recreate

# TRT engines do NOT need to be rebuilt on code updates —
# only rebuild if PyTorch, TensorRT, or the model architecture changes.
```

---

## Firewall / network access

The service runs on port 8000. To allow EasyErgoDashboard to reach it:

```bash
# If using ufw
sudo ufw allow 8000/tcp

# Or restrict to specific IP (recommended for production)
sudo ufw allow from <dashboard-server-ip> to any port 8000
```

For production, consider putting nginx or Caddy in front as a reverse proxy
with HTTPS and authentication.

---

## RTX 5070 Ti performance expectations

| Mode | FPS | Time for 30s video |
|------|-----|--------------------|
| PyTorch-only (body) | ~8-10 fps | ~70s |
| TRT (body, sm_120) | ~18-20 fps | ~35s |
| TRT (full, with hands) | ~8-10 fps | ~70s |

Total job time includes inference + OpenSim IK (~20s) + MVNX export (~5s).
