"""
TRC file loader.

Parses a .trc (Track Row Column) file and returns marker positions,
names, frame rate, units, and timestamps.
"""

from pathlib import Path
from typing import Dict

import numpy as np


def load_trc(trc_path: str) -> Dict:
    """
    Load a TRC file and return marker data.

    Args:
        trc_path: Path to the .trc file

    Returns:
        dict with keys:
            markers      : (T, M, 3) float64 — marker positions
            marker_names : List[str] of length M
            fps          : float
            units        : str  ('mm' or 'm')
            times        : (T,) float64 — time stamps
            num_frames   : int
            num_markers  : int
    """
    path = Path(trc_path)
    if not path.exists():
        raise FileNotFoundError(f"TRC file not found: {trc_path}")

    with open(path, "r") as fh:
        lines = fh.readlines()

    # ── Header (line index 2, 0-based) ──────────────────────────────────────
    # Format:  DataRate  CameraRate  NumFrames  NumMarkers  Units  ...
    header_vals = lines[2].strip().split("\t")
    fps         = float(header_vals[0])
    num_frames  = int(header_vals[2])
    num_markers = int(header_vals[3])
    units       = header_vals[4].strip()

    # ── Marker names (line index 3) ──────────────────────────────────────────
    # Format:  Frame#  Time  Name1  (empty)  (empty)  Name2  ...
    name_cols = lines[3].strip().split("\t")
    marker_names = []
    for i in range(2, len(name_cols), 3):
        name = name_cols[i].strip()
        if name:
            marker_names.append(name)

    if len(marker_names) != num_markers:
        raise ValueError(
            f"Expected {num_markers} marker names, found {len(marker_names)}"
        )

    # ── Data (lines 6+, 0-based) ─────────────────────────────────────────────
    markers = np.zeros((num_frames, num_markers, 3), dtype=np.float64)
    times   = np.zeros(num_frames, dtype=np.float64)

    row = 0
    for line in lines[6:]:
        line = line.strip()
        if not line:
            continue
        if row >= num_frames:
            break

        vals = line.split("\t")
        try:
            times[row] = float(vals[1])
        except (IndexError, ValueError):
            continue

        for j in range(num_markers):
            base = 2 + j * 3
            try:
                markers[row, j, 0] = float(vals[base])
                markers[row, j, 1] = float(vals[base + 1])
                markers[row, j, 2] = float(vals[base + 2])
            except (IndexError, ValueError):
                markers[row, j, :] = np.nan

        row += 1

    # Trim to actual rows read (guards against short files)
    markers = markers[:row]
    times   = times[:row]

    return {
        "markers":      markers,
        "marker_names": marker_names,
        "fps":          fps,
        "units":        units,
        "times":        times,
        "num_frames":   row,
        "num_markers":  num_markers,
    }
