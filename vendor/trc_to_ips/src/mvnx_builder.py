"""
MVNX v4 builder for TRC marker data.

Converts TRC marker positions (73 body landmarks) directly to an MVNX v4
file without computing orientations or joint angles. Each TRC marker becomes
an MVNX segment; the kinematic tree is defined in trc_marker_model.py.

Coordinate conversion (TRC Y-up → MVNX Z-up):
    TRC   : X = right,   Y = up,    Z = backward  (typical optical MoCap)
    MVNX  : X = forward, Y = left,  Z = up         (Xsens convention)
    Mapping: (x, y, z)_TRC → (x, -z, y)_MVNX
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..models.trc_marker_model import (
    SEGMENTS, SEGMENT_LABELS, SEGMENT_IDS, SEGMENT_COUNT,
    JOINTS, JOINT_LABELS, JOINT_COUNT,
    ROOT_SEGMENT,
)
from .trc_loader import load_trc


# ── helpers ──────────────────────────────────────────────────────────────────

def _arr_str(arr: np.ndarray, precision: int = 6) -> str:
    """Flat numpy array → space-separated string."""
    return " ".join(f"{v:.{precision}f}" for v in arr.ravel())


def _to_mvnx_coords(positions_mm: np.ndarray) -> np.ndarray:
    """
    Convert marker positions from TRC space to MVNX space.

    TRC (Y-up, mm)  →  MVNX (Z-up, metres)
    x_mvnx =  x_trc / 1000
    y_mvnx = -z_trc / 1000
    z_mvnx =  y_trc / 1000

    Args:
        positions_mm: (..., 3) array in TRC mm

    Returns:
        (..., 3) array in MVNX metres
    """
    out = np.empty_like(positions_mm, dtype=np.float64)
    out[..., 0] =  positions_mm[..., 0] / 1000.0   # X → X
    out[..., 1] = -positions_mm[..., 2] / 1000.0   # -Z → Y
    out[..., 2] =  positions_mm[..., 1] / 1000.0   # Y → Z
    return out


# ── main exporter ─────────────────────────────────────────────────────────────

class TRCToMVNX:
    """
    Convert a TRC file to MVNX v4 format.

    Only positions are written; orientations and joint angles are omitted.
    The MVNX segments follow the TRC marker order so that IPS correctly
    maps position data to body landmarks.
    """

    def export(
        self,
        trc_path: str,
        output_path: str,
        original_filename: Optional[str] = None,
    ) -> str:
        """
        Convert a TRC file to an MVNX v4 file.

        Args:
            trc_path:          Path to the input .trc file
            output_path:       Path for the output .mvnx file
            original_filename: Optional label stored in <subject> metadata

        Returns:
            Absolute path to the created .mvnx file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Load TRC ──────────────────────────────────────────────────────────
        print(f"  Loading TRC: {trc_path}")
        data = load_trc(trc_path)

        trc_markers     = data["markers"]       # (T, M, 3) mm
        trc_names       = data["marker_names"]  # List[str]
        fps             = data["fps"]
        units           = data["units"]
        num_frames      = data["num_frames"]
        num_trc_markers = data["num_markers"]

        print(f"  {num_frames} frames @ {fps:.3f} fps  |  {num_trc_markers} markers  |  units={units}")

        # If units are already metres, skip the /1000 step in coordinate transform
        if units.lower() == "m":
            trc_markers = trc_markers * 1000.0   # temporarily scale to mm for uniform path

        # ── Validate that TRC markers match the expected model ────────────────
        self._validate_markers(trc_names)

        # ── Reorder to match SEGMENT_LABELS (TRC order) ───────────────────────
        # trc_names should already be in TRC order == SEGMENT_LABELS order,
        # but we build an explicit index map for safety.
        trc_idx = {name: i for i, name in enumerate(trc_names)}
        ordered_indices = [trc_idx[label] for label in SEGMENT_LABELS]

        # positions[t, seg_id-1, :] = TRC marker for that segment
        positions_mm  = trc_markers[:, ordered_indices, :]   # (T, 73, 3) mm
        positions_mvnx = _to_mvnx_coords(positions_mm)       # (T, 73, 3) metres

        # Replace any NaN with 0.0 (occluded markers)
        nan_count = int(np.isnan(positions_mvnx).sum())
        if nan_count:
            print(f"  Warning: {nan_count} NaN values replaced with 0.0")
            positions_mvnx = np.nan_to_num(positions_mvnx, nan=0.0)

        # ── Calibration / T-pose positions (use first frame) ─────────────────
        tpose_positions = positions_mvnx[0]  # (73, 3)

        # ── Build XML ─────────────────────────────────────────────────────────
        print("  Building MVNX XML...")
        root = self._build_xml(
            positions_mvnx  = positions_mvnx,
            tpose_positions = tpose_positions,
            fps             = fps,
            num_frames      = num_frames,
            original_filename = original_filename,
        )

        # ── Write file ────────────────────────────────────────────────────────
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(output_path), encoding="utf-8", xml_declaration=True)

        print(f"  Written: {output_path}")
        return str(output_path)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_markers(self, trc_names):
        """Check that every expected segment label exists in the TRC file."""
        trc_set   = set(trc_names)
        model_set = set(SEGMENT_LABELS)
        missing   = model_set - trc_set
        extra     = trc_set - model_set
        if missing:
            print(f"  Warning: {len(missing)} model markers not in TRC: {sorted(missing)}")
        if extra:
            print(f"  Info: {len(extra)} extra TRC markers ignored: {sorted(extra)}")

    # ── XML construction ──────────────────────────────────────────────────────

    def _build_xml(
        self,
        positions_mvnx: np.ndarray,
        tpose_positions: np.ndarray,
        fps: float,
        num_frames: int,
        original_filename: Optional[str],
    ) -> ET.Element:
        """Build the full MVNX v4 element tree."""

        # Root <mvnx>
        mvnx = ET.Element("mvnx")
        mvnx.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        mvnx.set(
            "xsi:schemaLocation",
            "http://www.xsens.com/mvn/mvnx http://www.xsens.com/mvn/mvnx/schema.xsd",
        )
        mvnx.set("version", "4")

        # <mvn> version info
        mvn = ET.SubElement(mvnx, "mvn")
        mvn.set("version", "2024.0.0")
        mvn.set("build", "Version 2024.0.0. Build 11031. Date 2023-12-12. Revision 1702394261.")

        # <comment>
        comment = ET.SubElement(mvnx, "comment")
        comment.text = "Generated from TRC marker data by TRCToIPS converter"

        # <subject>
        now = datetime.now()
        subject = ET.SubElement(mvnx, "subject")
        subject.set("label", "MVN System 1")
        subject.set("torsoColor", "#ea6852")
        subject.set("frameRate", str(int(round(fps))))
        subject.set("segmentCount", str(SEGMENT_COUNT))
        subject.set("recDate", now.strftime("%a %b %d %H:%M:%S.000 %Y"))
        subject.set("recDateMSecsSinceEpoch", str(int(now.timestamp() * 1000)))
        if original_filename:
            subject.set("originalFilename", original_filename)
        subject.set("configuration", "FullBody")
        subject.set("userScenario", "singleLevel")
        subject.set("processingQuality", "HD")

        subj_comment = ET.SubElement(subject, "comment")
        subj_comment.text = (
            f"TRC marker positions only (no orientations). "
            f"Segments: {SEGMENT_COUNT}. Frames: {num_frames}. FPS: {fps:.6f}."
        )

        # <segments>
        self._add_segments(subject)

        # <joints>
        self._add_joints(subject)

        # <frames>
        self._add_frames(subject, positions_mvnx, tpose_positions, fps, num_frames)

        return mvnx

    def _add_segments(self, parent: ET.Element):
        """Add <segments> element with one <segment> per TRC marker."""
        segs_elem = ET.SubElement(parent, "segments")
        for seg_id, seg_label in SEGMENTS:
            seg = ET.SubElement(segs_elem, "segment")
            seg.set("label", seg_label)
            seg.set("id", str(seg_id))
            # Each segment has a single origin point so connectors can reference it
            points_elem = ET.SubElement(seg, "points")
            pt = ET.SubElement(points_elem, "point")
            pt.set("label", f"origin_{seg_label}")
            pos_b = ET.SubElement(pt, "pos_b")
            pos_b.text = "0.000000 0.000000 0.000000"

    def _add_joints(self, parent: ET.Element):
        """Add <joints> element with the kinematic tree."""
        joints_elem = ET.SubElement(parent, "joints")
        for joint_label, parent_seg, child_seg in JOINTS:
            joint = ET.SubElement(joints_elem, "joint")
            joint.set("label", joint_label)
            c1 = ET.SubElement(joint, "connector1")
            c1.text = f"{parent_seg}/origin_{parent_seg}"
            c2 = ET.SubElement(joint, "connector2")
            c2.text = f"{child_seg}/origin_{child_seg}"

    def _add_frames(
        self,
        parent: ET.Element,
        positions: np.ndarray,
        tpose_positions: np.ndarray,
        fps: float,
        num_frames: int,
    ):
        """
        Add <frames> element.

        Calibration frames (identity, tpose, tpose-isb) use first-frame positions.
        Normal frames write only <position> (no <orientation> or <jointAngle>).
        """
        frames_elem = ET.SubElement(parent, "frames")
        frames_elem.set("segmentCount", str(SEGMENT_COUNT))
        frames_elem.set("sensorCount",  "0")
        frames_elem.set("jointCount",   str(JOINT_COUNT))

        tpose_str = _arr_str(tpose_positions)

        # ── identity frame ───────────────────────────────────────────────────
        fr = ET.SubElement(frames_elem, "frame")
        fr.set("time",  "0")
        fr.set("index", "")
        fr.set("tc",    "00:00:00:00")
        fr.set("ms",    "0")
        fr.set("type",  "identity")
        ET.SubElement(fr, "position").text = tpose_str

        # ── tpose frame ──────────────────────────────────────────────────────
        fr = ET.SubElement(frames_elem, "frame")
        fr.set("time",  "0")
        fr.set("index", "")
        fr.set("tc",    "00:00:00:00")
        fr.set("ms",    "0")
        fr.set("type",  "tpose")
        ET.SubElement(fr, "position").text = tpose_str

        # ── tpose-isb frame ──────────────────────────────────────────────────
        fr = ET.SubElement(frames_elem, "frame")
        fr.set("time",  "0")
        fr.set("index", "")
        fr.set("tc",    "00:00:00:00")
        fr.set("ms",    "0")
        fr.set("type",  "tpose-isb")
        ET.SubElement(fr, "position").text = tpose_str

        # ── normal frames ────────────────────────────────────────────────────
        dt_ms = 1000.0 / fps
        for t in range(num_frames):
            time_ms = int(round(t * dt_ms))

            fr = ET.SubElement(frames_elem, "frame")
            fr.set("time",  str(time_ms))
            fr.set("index", str(t + 1))
            fr.set("tc",    "00:00:00:00")
            fr.set("ms",    str(time_ms))
            fr.set("type",  "normal")

            ET.SubElement(fr, "position").text = _arr_str(positions[t])


# ── convenience function ──────────────────────────────────────────────────────

def convert_trc_to_mvnx(
    trc_path: str,
    output_path: str,
    original_filename: Optional[str] = None,
) -> str:
    """
    Convert a TRC file to MVNX v4.

    Args:
        trc_path:          Path to the input .trc file
        output_path:       Desired output .mvnx path
        original_filename: Optional label for subject metadata

    Returns:
        Path to the created .mvnx file
    """
    return TRCToMVNX().export(
        trc_path          = trc_path,
        output_path       = output_path,
        original_filename = original_filename,
    )
