# FastSAM3DToOpenSim — Docker image
# Base: CUDA 12.8 (Blackwell sm_120 minimum requirement)
#
# Two conda envs are created:
#   fast_sam_3d_body  — Python 3.11, PyTorch 2.10+cu128, TensorRT 10.16
#   opensim           — Python 3.10, opensim-org channel
#
# Miniforge is placed at /opt/conda because opensim_ik_runner.py hard-codes
#   /opt/conda/envs/opensim/bin/python as the subprocess interpreter.
#
# TRT .engine files are NOT built here — they are GPU-architecture-specific.
# Build them inside a running container (see README / HOW_TO_RUN.md).
#
# Volumes expected at runtime:
#   ./checkpoints  →  /app/checkpoints
#   ./videos       →  /app/videos   (read-only)
#   ./outputs      →  /outputs

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# --------------------------------------------------------------------------- #
# System packages                                                               #
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates \
        ffmpeg \
        libgl1 libglib2.0-0 \
        libgomp1 libegl1 libxrender1 libxext6 \
        libsm6 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Miniforge → /opt/conda                                                        #
# --------------------------------------------------------------------------- #
RUN wget -q \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"
SHELL ["/bin/bash", "-c"]

# --------------------------------------------------------------------------- #
# Env 1: fast_sam_3d_body (Python 3.11)                                         #
# --------------------------------------------------------------------------- #
# Copy requirements before heavy installs so Docker can cache the layer
COPY docker/requirements_docker.txt /tmp/requirements_docker.txt

RUN conda create -y -n fast_sam_3d_body python=3.11 && \
    source activate fast_sam_3d_body && \
    # PyTorch cu128
    pip install --no-cache-dir \
        torch==2.10.0+cu128 \
        torchvision==0.25.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    # TensorRT
    pip install --no-cache-dir tensorrt==10.16.0.72 && \
    # chumpy has a broken setup.py that requires pip in isolated build env
    pip install --no-cache-dir --no-build-isolation chumpy==0.70 && \
    # Main requirements (torch/trt/git+ entries stripped)
    pip install --no-cache-dir -r /tmp/requirements_docker.txt && \
    # detectron2 imports torch in setup.py — needs no-build-isolation
    pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/detectron2.git@a1ce2f956a1d2212ad672e3c47d53405c2fe4312" && \
    # Remaining git-hosted packages
    pip install --no-cache-dir \
        "git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8" \
        "git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6" \
        "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" && \
    # pip packages (ultralytics etc.) pull in opencv-python which has broken
    # FFMPEG support in Docker. Uninstall it and replace with conda-forge opencv
    # which links against system libav.
    pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true && \
    conda install -y -n fast_sam_3d_body -c conda-forge "opencv>=4.8" && \
    conda clean -afy && pip cache purge

# --------------------------------------------------------------------------- #
# Env 2: opensim (Python 3.10, opensim-org channel)                            #
# opensim_ik_runner.py calls /opt/conda/envs/opensim/bin/python directly.     #
# --------------------------------------------------------------------------- #
RUN conda create -y -n opensim python=3.10 && \
    conda install -y -n opensim -c opensim-org -c conda-forge opensim && \
    conda clean -afy

# --------------------------------------------------------------------------- #
# Application code                                                              #
# --------------------------------------------------------------------------- #
WORKDIR /app
COPY . /app

# Ensure output dir exists
RUN mkdir -p /outputs

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
