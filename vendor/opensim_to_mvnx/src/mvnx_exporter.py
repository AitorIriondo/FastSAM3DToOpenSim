"""
MVNX v4 (Xsens MVN Open XML) exporter.

Generates .mvnx files from existing pipeline outputs (.trc, .mot, .osim).
Compatible with Visual3D, MATLAB, C-Motion, and other tools that consume
Xsens motion data.

Reference: https://base.xsens.com/hc/en-us/articles/360012672099-MVNX-Version-4-File-Structure
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .opensim_ik import load_mot
from .trc_exporter import load_trc
from ..utils.xsens_model import (
    SEGMENT_LABELS, SEGMENT_COUNT, SEGMENT_IDS, SEGMENT_POINTS,
    JOINT_LABELS, JOINT_COUNT, JOINTS,
    ERGONOMIC_JOINT_ANGLES, FOOT_CONTACT_DEFINITIONS,
    SENSOR_COUNT, SENSOR_LABELS,
)
from ..utils.opensim_to_xsens import (
    compute_segment_positions,
    compute_tpose_positions,
    compute_orientations_from_positions,
    compute_fk_positions,
    compute_joint_angles_from_orientations,
)


class MVNXExporter:
    """
    Export pipeline outputs to MVNX v4 format.

    Takes existing .trc, .mot, and optional .osim files and produces
    a valid MVNX v4 XML file.
    """

    def __init__(
        self,
        fps: float = 30.0,
        subject_height: float = 1.75,
    ):
        """
        Initialize MVNX exporter.

        Args:
            fps: Frame rate in Hz
            subject_height: Subject height in meters
        """
        self.fps = fps
        self.subject_height = subject_height

    def export(
        self,
        trc_path: str,
        mot_path: str,
        output_path: str,
        osim_path: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> str:
        """
        Export to MVNX v4 format.

        Args:
            trc_path: Path to TRC file with marker trajectories
            mot_path: Path to MOT file with joint angles
            output_path: Output MVNX file path
            osim_path: Path to .osim model (optional, for segment info)
            original_filename: Original video filename for metadata

        Returns:
            Path to created MVNX file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load input data
        print("  Loading TRC markers...")
        trc_data = load_trc(str(trc_path))
        trc_markers = trc_data["markers"]       # (T, M, 3)
        trc_marker_names = trc_data["marker_names"]
        trc_fps = trc_data["fps"]

        print("  Loading MOT joint angles...")
        mot_data = load_mot(str(mot_path))
        mot_time = mot_data["time"]             # (T,)
        mot_coords = mot_data["coordinates"]    # Dict[str, (T,)]

        # Use TRC fps if available, fall back to constructor value
        fps = trc_fps if trc_fps > 0 else self.fps

        # Determine frame count (minimum of TRC and MOT)
        num_frames_trc = trc_markers.shape[0]
        num_frames_mot = len(mot_time)
        num_frames = min(num_frames_trc, num_frames_mot)
        trc_markers = trc_markers[:num_frames]

        # Trim MOT data
        mot_time = mot_time[:num_frames]
        mot_coords = {k: v[:num_frames] for k, v in mot_coords.items()}

        print(f"  Frames: {num_frames}, FPS: {fps}")

        # Step 1: Compute raw segment positions from TRC markers (OpenSim Y-up)
        print("  Computing segment positions from markers...")
        raw_positions = compute_segment_positions(
            trc_markers, trc_marker_names, self.subject_height
        )

        # Step 2: Transform positions from OpenSim (Y-up) to Xsens (Z-up)
        # OpenSim: X=forward, Y=up, Z=right
        # Xsens:   X=forward, Y=left, Z=up
        # (x, y, z) -> (x, -z, y)
        print("  Converting to Xsens coordinate system (Z-up)...")
        raw_positions_xsens = np.empty_like(raw_positions)
        raw_positions_xsens[:, :, 0] = raw_positions[:, :, 0]
        raw_positions_xsens[:, :, 1] = -raw_positions[:, :, 2]
        raw_positions_xsens[:, :, 2] = raw_positions[:, :, 1]

        # Step 3: T-pose positions (already in Xsens Z-up)
        print("  Computing T-pose skeleton...")
        tpose_positions = compute_tpose_positions(self.subject_height)

        # Step 4: Compute orientations from bone directions
        # Uses position data to determine each segment's global rotation
        print("  Computing orientations from bone directions...")
        orientations = compute_orientations_from_positions(
            raw_positions_xsens, tpose_positions
        )

        # Step 5: Compute FK-consistent positions
        # Reconstructs positions from orientations + T-pose bone offsets
        # This guarantees a connected skeleton in MVN Studio
        print("  Computing FK-consistent positions...")
        root_positions = raw_positions_xsens[:, 0, :]  # Pelvis from markers
        positions = compute_fk_positions(
            orientations, tpose_positions, root_positions
        )

        # Step 6: Derive joint angles from orientations
        # BABBq = GBAq* x GBBq, decomposed as ZXY Euler (degrees)
        print("  Computing joint angles from orientations...")
        joint_angles = compute_joint_angles_from_orientations(orientations)

        # Step 7: Build and write MVNX XML
        print("  Writing MVNX v4 XML...")
        mvnx_root = self._build_mvnx_xml(
            orientations=orientations,
            tpose_positions=tpose_positions,
            positions=positions,
            joint_angles=joint_angles,
            fps=fps,
            num_frames=num_frames,
            original_filename=original_filename,
        )

        # Write with XML declaration
        tree = ET.ElementTree(mvnx_root)
        ET.indent(tree, space="  ")
        tree.write(
            str(output_path),
            encoding="utf-8",
            xml_declaration=True,
        )

        print(f"  MVNX exported: {output_path}")
        return str(output_path)

    def _build_mvnx_xml(
        self,
        orientations: np.ndarray,
        tpose_positions: np.ndarray,
        positions: np.ndarray,
        joint_angles: np.ndarray,
        fps: float,
        num_frames: int,
        original_filename: Optional[str],
    ) -> ET.Element:
        """Build the complete MVNX v4 XML tree."""
        # Root element — no default xmlns (MVN Studio chokes on it)
        mvnx = ET.Element("mvnx")
        mvnx.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        mvnx.set(
            "xsi:schemaLocation",
            "http://www.xsens.com/mvn/mvnx http://www.xsens.com/mvn/mvnx/schema.xsd",
        )
        mvnx.set("version", "4")

        # MVN version info
        mvn = ET.SubElement(mvnx, "mvn")
        mvn.set("version", "2024.0.0")
        mvn.set("build", "Version 2024.0.0. Build 11031. Date 2023-12-12. Revision 1702394261.")

        # Comment
        comment = ET.SubElement(mvnx, "comment")
        comment.text = "Generated from monocular video by SAM3D-OpenSim pipeline"

        # Subject element
        subject = ET.SubElement(mvnx, "subject")
        subject.set("label", "MVN System 1")
        subject.set("torsoColor", "#ea6852")
        subject.set("frameRate", str(int(fps)))
        subject.set("segmentCount", str(SEGMENT_COUNT))
        now = datetime.now()
        subject.set("recDate", now.strftime("%a %b %d %H:%M:%S.000 %Y"))
        subject.set("recDateMSecsSinceEpoch", str(int(now.timestamp() * 1000)))
        if original_filename:
            subject.set("originalFilename", original_filename)
        subject.set("configuration", "FullBody")
        subject.set("userScenario", "singleLevel")
        subject.set("processingQuality", "HD")

        subject_comment = ET.SubElement(subject, "comment")
        subject_comment.text = (
            f"Video-derived motion capture. Height: {self.subject_height}m. "
            f"Frames: {num_frames}. FPS: {fps}."
        )

        # Segments
        self._add_segments(subject)

        # Sensors (required by MVN Studio)
        self._add_sensors(subject)

        # Joints (with connector1/connector2 format)
        self._add_joints(subject)

        # Ergonomic joint angles
        self._add_ergonomic_joint_angles(subject)

        # Foot contact definitions
        self._add_foot_contact_definitions(subject)

        # Frames
        self._add_frames(
            subject,
            orientations=orientations,
            tpose_positions=tpose_positions,
            positions=positions,
            joint_angles=joint_angles,
            fps=fps,
            num_frames=num_frames,
        )

        return mvnx

    def _add_segments(self, parent: ET.Element):
        """Add segment definitions with anatomical points.

        Matches Xsens MVNX v4 format: each segment has a <points> wrapper
        containing <point> elements with <pos_b> child elements.
        """
        segments_elem = ET.SubElement(parent, "segments")
        for seg_id, seg_label in zip(
            [SEGMENT_IDS[s] for s in SEGMENT_LABELS], SEGMENT_LABELS
        ):
            seg = ET.SubElement(segments_elem, "segment")
            seg.set("label", seg_label)
            seg.set("id", str(seg_id))

            # Add anatomical points inside <points> wrapper
            seg_points = SEGMENT_POINTS.get(seg_label, [])
            points_elem = ET.SubElement(seg, "points")
            for point_label, (px, py, pz) in seg_points:
                point = ET.SubElement(points_elem, "point")
                point.set("label", point_label)
                pos_b = ET.SubElement(point, "pos_b")
                pos_b.text = f"{px:.6f} {py:.6f} {pz:.6f}"

    def _add_sensors(self, parent: ET.Element):
        """Add sensor definitions (17 IMU sensors for full body)."""
        sensors_elem = ET.SubElement(parent, "sensors")
        for label in SENSOR_LABELS:
            sensor = ET.SubElement(sensors_elem, "sensor")
            sensor.set("label", label)

    def _add_joints(self, parent: ET.Element):
        """Add joint definitions with connector1/connector2 format.

        Format: connector1 = "ParentSegment/jointLabel"
                connector2 = "ChildSegment/jointLabel"
        """
        joints_elem = ET.SubElement(parent, "joints")
        for joint_label, parent_seg, child_seg in JOINTS:
            joint = ET.SubElement(joints_elem, "joint")
            joint.set("label", joint_label)
            c1 = ET.SubElement(joint, "connector1")
            c1.text = f"{parent_seg}/{joint_label}"
            c2 = ET.SubElement(joint, "connector2")
            c2.text = f"{child_seg}/{joint_label}"

    def _add_ergonomic_joint_angles(self, parent: ET.Element):
        """Add ergonomic joint angle definitions."""
        ergo_elem = ET.SubElement(parent, "ergonomicJointAngles")
        for idx, (label, parent_seg, child_seg) in enumerate(ERGONOMIC_JOINT_ANGLES):
            eja = ET.SubElement(ergo_elem, "ergonomicJointAngle")
            eja.set("label", label)
            eja.set("index", str(idx))
            eja.set("parentSegment", parent_seg)
            eja.set("childSegment", child_seg)

    def _add_foot_contact_definitions(self, parent: ET.Element):
        """Add foot contact definitions."""
        fcd_elem = ET.SubElement(parent, "footContactDefinition")
        for label, idx in FOOT_CONTACT_DEFINITIONS:
            cd = ET.SubElement(fcd_elem, "contactDefinition")
            cd.set("label", label)
            cd.set("index", str(idx))

    def _add_frames(
        self,
        parent: ET.Element,
        orientations: np.ndarray,
        tpose_positions: np.ndarray,
        positions: np.ndarray,
        joint_angles: np.ndarray,
        fps: float,
        num_frames: int,
    ):
        """Add calibration and normal frames per MVNX v4 spec."""
        # Frames element with counts (per MVN User Manual p.98)
        frames_elem = ET.SubElement(parent, "frames")
        frames_elem.set("segmentCount", str(SEGMENT_COUNT))
        frames_elem.set("sensorCount", str(SENSOR_COUNT))
        frames_elem.set("jointCount", str(JOINT_COUNT))

        # Pre-compute identity/tpose data
        identity_quats = np.tile([1.0, 0.0, 0.0, 0.0], SEGMENT_COUNT)
        identity_ori_str = _array_to_str(identity_quats)
        tpose_pos_str = _array_to_str(tpose_positions.flatten())

        # Identity frame: all orientations = unit quaternion
        frame = ET.SubElement(frames_elem, "frame")
        frame.set("time", "0")
        frame.set("index", "")
        frame.set("tc", "00:00:00:00")
        frame.set("ms", "0")
        frame.set("type", "identity")
        ori = ET.SubElement(frame, "orientation")
        ori.text = identity_ori_str
        pos = ET.SubElement(frame, "position")
        pos.text = tpose_pos_str

        # T-pose frame: same as identity for video-derived data
        frame = ET.SubElement(frames_elem, "frame")
        frame.set("time", "0")
        frame.set("index", "")
        frame.set("tc", "00:00:00:00")
        frame.set("ms", "0")
        frame.set("type", "tpose")
        ori = ET.SubElement(frame, "orientation")
        ori.text = identity_ori_str
        pos = ET.SubElement(frame, "position")
        pos.text = tpose_pos_str

        # T-pose-ISB frame: same positions, identity orientations
        frame = ET.SubElement(frames_elem, "frame")
        frame.set("time", "0")
        frame.set("index", "")
        frame.set("tc", "00:00:00:00")
        frame.set("ms", "0")
        frame.set("type", "tpose-isb")
        ori = ET.SubElement(frame, "orientation")
        ori.text = identity_ori_str
        pos = ET.SubElement(frame, "position")
        pos.text = tpose_pos_str

        # Normal motion frames: orientation, position, jointAngle
        dt_ms = int(1000 / fps)
        for t in range(num_frames):
            time_ms = t * dt_ms
            tc = "00:00:00:00"

            frame = ET.SubElement(frames_elem, "frame")
            frame.set("time", str(time_ms))
            frame.set("index", str(t + 1))
            frame.set("tc", tc)
            frame.set("ms", str(time_ms))
            frame.set("type", "normal")

            # Orientation first, then position (per MVN manual p.101)
            ori = ET.SubElement(frame, "orientation")
            ori.text = _array_to_str(orientations[t].flatten())

            pos = ET.SubElement(frame, "position")
            pos.text = _array_to_str(positions[t].flatten())

            ja = ET.SubElement(frame, "jointAngle")
            ja.text = _array_to_str(joint_angles[t].flatten())


def export_mvnx(
    trc_path: str,
    mot_path: str,
    output_path: str,
    fps: float = 30.0,
    subject_height: float = 1.75,
    osim_path: Optional[str] = None,
    original_filename: Optional[str] = None,
) -> str:
    """
    Convenience function to export MVNX v4 file.

    Args:
        trc_path: Path to TRC file
        mot_path: Path to MOT file
        output_path: Output MVNX file path
        fps: Frame rate
        subject_height: Subject height in meters
        osim_path: Path to .osim model (optional)
        original_filename: Original video filename

    Returns:
        Path to created MVNX file
    """
    exporter = MVNXExporter(fps=fps, subject_height=subject_height)
    return exporter.export(
        trc_path=trc_path,
        mot_path=mot_path,
        output_path=output_path,
        osim_path=osim_path,
        original_filename=original_filename,
    )


def _array_to_str(arr: np.ndarray, precision: int = 6) -> str:
    """Convert numpy array to space-separated string."""
    return " ".join(f"{v:.{precision}f}" for v in arr)
