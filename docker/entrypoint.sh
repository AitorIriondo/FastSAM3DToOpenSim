#!/bin/bash
# Activate the main conda env and hand off to the command
source /opt/conda/etc/profile.d/conda.sh
conda activate fast_sam_3d_body
exec "$@"
