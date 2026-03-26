"""
MVNX export wrappers for the FastSAM3D pipeline.

Two output formats:
  - Normal MVNX (.mvnx): full Xsens MVN v4 format via OpenSim-to-MVNX.
    Requires TRC + MOT (OpenSim IK).  Contains segment positions, orientations,
    and joint angles (23 Xsens segments).

  - IPS MVNX (.ipsmvnx): position-only format via TRCtoIPS.
    Requires a 73-marker TRC.  Intended for Industrial Path Solutions (IPS IMMA)
    ergonomics simulation.
"""

import logging
import os
import tempfile
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def export_normal_mvnx(
    trc_path: str,
    mot_path: str,
    output_path: str,
    fps: float = 30.0,
    subject_height: float = 1.75,
    osim_path: Optional[str] = None,
    original_filename: Optional[str] = None,
) -> Optional[str]:
    """
    Generate a full MVNX v4 file from OpenSim TRC + MOT outputs.

    Uses vendor/opensim_to_mvnx (OpenSim-to-MVNX).

    Args:
        trc_path: Path to TRC marker file (OpenSim markers, mm)
        mot_path: Path to IK MOT file (joint angles, degrees)
        output_path: Desired output .mvnx path
        fps: Frame rate in Hz
        subject_height: Subject height in metres
        osim_path: Optional path to .osim model for extra segment metadata
        original_filename: Source video filename for MVNX metadata

    Returns:
        output_path on success, None on failure
    """
    try:
        from vendor.opensim_to_mvnx import export_mvnx
        print(f"  Writing MVNX       → {output_path}")
        export_mvnx(
            trc_path=trc_path,
            mot_path=mot_path,
            output_path=output_path,
            fps=fps,
            subject_height=subject_height,
            osim_path=osim_path,
            original_filename=original_filename,
        )
        return output_path
    except Exception as exc:
        logger.warning("MVNX export failed: %s", exc, exc_info=True)
        print(f"  WARNING: MVNX export failed — {exc}")
        return None


def export_ips_mvnx(
    keypoints_opensim: np.ndarray,
    jcoords_opensim: Optional[np.ndarray],
    fps: float,
    output_path: str,
    input_name: str = "",
) -> Optional[str]:
    """
    Generate an IPS MVNX file (.ipsmvnx) from MHR70 keypoints.

    Builds a 73-marker TRC in memory, writes it to a temporary file, then
    calls vendor/trc_to_ips (TRCtoIPS) to convert it.

    All 70 MHR70 keypoints are used — including finger joints predicted in
    body-only mode (lower accuracy, same structure as full mode).

    Args:
        keypoints_opensim: (N, 70, 3) array in OpenSim Y-up mm
        jcoords_opensim: (N, 127, 3) MHR armature joints (for SpineMid); may be None
        fps: Frame rate in Hz
        output_path: Desired .ipsmvnx output path
        input_name: Source name used for the temporary TRC filename

    Returns:
        output_path on success, None on failure
    """
    try:
        from vendor.trc_to_ips import convert_trc_to_ipsmvnx
        from sam_3d_body.export.keypoint_converter import KeypointConverter
        from sam_3d_body.export.trc_exporter import TRCExporter

        # Build 73-marker array
        converter = KeypointConverter()
        markers_ips, names_ips = converter.convert_for_ips(
            keypoints_opensim, jcoords_3d=jcoords_opensim
        )

        # Write temporary TRC
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_trc = os.path.join(tmp_dir, f"{input_name or 'ips'}_73markers.trc")
            exporter = TRCExporter(fps=fps, units="mm")
            exporter.export(markers_ips, names_ips, tmp_trc)

            print(f"  Writing IPS MVNX   → {output_path}")
            convert_trc_to_ipsmvnx(
                trc_path=tmp_trc,
                output_path=output_path,
                original_filename=input_name or None,
            )

        return output_path

    except Exception as exc:
        logger.warning("IPS MVNX export failed: %s", exc, exc_info=True)
        print(f"  WARNING: IPS MVNX export failed — {exc}")
        return None
