# TRT Engine Compilation Guide

TensorRT engines dramatically improve inference speed by compiling the model for
your specific GPU. This is a **one-time setup** per machine — engines cannot be
shared across different GPU architectures.

---

## Why bother?

| Mode | Body-only FPS | Notes |
|------|:---:|-------|
| PyTorch only | ~8-10 fps | Works out of the box, no setup |
| + TRT backbone | ~14-16 fps | ~2x faster backbone |
| + TRT backbone + torch.compile | ~18-20 fps | Best throughput on sm_120 |

All numbers are for RTX 5070 Ti / RTX 5090 (Blackwell, sm_120), 848×480 input.

---

## GPU compatibility

| GPU | Architecture | Compute Cap. | CUDA toolkit |
|-----|-------------|:---:|:---:|
| RTX 5090 | Blackwell GB202 | sm_120 | cu128 |
| RTX 5070 Ti | Blackwell GB203 | sm_120 | cu128 |
| RTX 4090 | Ada Lovelace | sm_89 | cu124 |
| RTX 3090 | Ampere | sm_86 | cu124 |

Engines built on RTX 5090 and RTX 5070 Ti share the same sm_120 architecture,
but TRT optimises for exact SM/memory specs — always rebuild on the target machine.

---

## Step 1 — Start a build container

```bash
cd /opt/EasyErgo_FastSam3DService

docker compose run --rm \
  -e FASTSAM_MODE=cli \
  fastsam3d \
  bash
```

All following commands run **inside this container**.

---

## Step 2 — Build the YOLO pose engine (~2 min)

```bash
python convert_yolo_pose_trt.py \
  --model checkpoints/yolo/yolo11m-pose.pt \
  --output checkpoints/yolo/yolo11m-pose.engine \
  --imgsz 640 --half
```

Output: `checkpoints/yolo/yolo11m-pose.engine` (~40 MB)

---

## Step 3 — Build the MoGe FOV estimator engine (~5 min)

```bash
python convert_moge_encoder_trt.py \
  --output checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine
```

Output: `checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine` (~46 MB)

---

## Step 4 — Build the DINOv3 backbone engine (~20 min)

This is the longest step. The backbone is 1.6 GB.

```bash
python convert_backbone_tensorrt.py \
  --checkpoint checkpoints/sam-3d-body-dinov3 \
  --output checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine
```

Output: `checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine` (~1.6 GB)

**Tip:** You can run all three steps with the convenience script:

```bash
bash /app/build_tensorrt.sh
```

---

## Step 5 — Verify engines work

Exit the build container and restart the service with TRT enabled:

```bash
exit   # leave the build container

# Restart with TRT env
docker compose --env-file docker/trt.env up -d

# Health check
curl http://localhost:8000/api/v1/health
# Should show: "gpu_available": true

# Test inference (submit a short video and check FPS in logs)
docker compose logs -f &
curl -X POST http://localhost:8000/api/v1/process \
  -F "video=@videos/test.mp4" \
  -F "person_height=1.75"
```

Watch the logs for lines like `frame_step=1 → ~18.2 fps`.

---

## Troubleshooting

**Engine file not found at startup:**
Make sure the checkpoints volume is mounted correctly:
```bash
docker compose exec fastsam3d ls /app/checkpoints/yolo/
# Should show yolo11m-pose.engine
```

**TRT build fails with "unsupported SM":**
Your TensorRT version may not support your GPU yet.
Check `tensorrt --version` inside the container and compare with NVIDIA's
TensorRT support matrix. CUDA 12.8 + TRT 10.16 is required for sm_120.

**Slow despite TRT:**
Make sure `USE_TRT_BACKBONE=1` and `USE_COMPILE=1` are set:
```bash
docker compose exec fastsam3d env | grep -E "TRT|COMPILE"
```

**Rebuild engines after:**
- Upgrading TensorRT (version in requirements_docker.txt)
- Changing GPU
- Major PyTorch version upgrade
- Modifying backbone architecture

You do NOT need to rebuild engines when:
- Updating Python post-processing code
- Adding/changing the API
- Updating documentation
