#!/bin/bash
# FastSAM3D container entrypoint.
#
# FASTSAM_MODE=service  → start FastAPI HTTP server on port 8000 (default)
# FASTSAM_MODE=cli      → pass through to CLI (legacy / batch mode)

source /opt/conda/etc/profile.d/conda.sh
conda activate fast_sam_3d_body
export PYTHONPATH=/app:${PYTHONPATH}

if [ "${FASTSAM_MODE:-service}" = "service" ]; then
    echo "[entrypoint] Starting FastSAM3D service on :8000"
    exec uvicorn service.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --log-level info
else
    # CLI pass-through (original behaviour)
    exec "$@"
fi
