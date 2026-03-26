"""
OpenSim / Pose2Sim ↔ Xsens mapping and conversion utilities.

Maps the Pose2Sim OpenSim model (which has fewer spine segments) to the
Xsens 23-segment model. Provides forward kinematics to compute global
segment orientations from joint angles, and joint angle convention conversion.

The Pose2Sim model has these body segments:
  Pelvis, Torso (single spine), Head, Neck (if present),
  Humerus_r/l, Radius_r/l (or Ulna), Hand_r/l,
  Femur_r/l, Tibia_r/l, Calcn_r/l (foot), Toes_r/l

The Xsens model has 23 segments with 4 spine segments (L5, L3, T12, T8).
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, List, Optional, Tuple

from .xsens_model import (
    SEGMENT_LABELS, SEGMENT_COUNT, JOINT_LABELS, JOINT_COUNT,
    JOINTS, SEGMENT_PARENT, SEGMENT_TO_PARENT_JOINT,
)


# =============================================================================
# OpenSim Pose2Sim MOT coordinate name → Xsens joint mapping
# =============================================================================

# OpenSim MOT coordinate names (from the 40 DOF model)
# and their mapping to Xsens joints.
#
# OpenSim uses various decomposition orders per joint.
# The Pose2Sim model typically uses XYZ (flexion, adduction, rotation).
# Xsens MVNX uses ZXY Euler angles.

# Map from OpenSim coordinate names → (xsens_joint_label, axis_index, sign)
# axis_index: which of the 3 ZXY Euler angles this OpenSim DOF maps to
# ZXY = [Z_rotation(0)=flexion, X_rotation(1)=abduction, Y_rotation(2)=int_rotation]
OPENSIM_TO_XSENS_DOF: Dict[str, List[Tuple[str, int, float]]] = {
    # Pelvis (6 DOF) - pelvis global orientation
    # OpenSim pelvis: tilt=X, list=Z, rotation=Y (XZY intrinsic)
    # These define the root segment orientation, handled specially

    # Spine: L5_S1 (the single spine joint in Pose2Sim)
    # Maps to jL5S1 + distributed to jL4L3, jL1T12, jT9T8
    "L5_S1_Flex_Ext": [
        ("jL5S1", 0, 1.0),      # Flexion → Z rotation
        ("jL4L3", 0, 0.0),      # Will be interpolated
        ("jL1T12", 0, 0.0),
        ("jT9T8", 0, 0.0),
    ],
    "L5_S1_Lat_Bending": [
        ("jL5S1", 1, 1.0),      # Lateral bending → X rotation
        ("jL4L3", 1, 0.0),
        ("jL1T12", 1, 0.0),
        ("jT9T8", 1, 0.0),
    ],
    "L5_S1_axial_rotation": [
        ("jL5S1", 2, 1.0),      # Axial rotation → Y rotation
        ("jL4L3", 2, 0.0),
        ("jL1T12", 2, 0.0),
        ("jT9T8", 2, 0.0),
    ],

    # Neck (3 DOF)
    "neck_flexion": [("jT1C7", 0, 1.0)],     # Flexion → Z
    "neck_bending": [("jT1C7", 1, 1.0)],     # Lateral → X
    "neck_rotation": [("jT1C7", 2, 1.0)],    # Rotation → Y

    # Head (mapped to jC1Head) — no separate head DOF in Pose2Sim,
    # head orientation comes from neck. We leave jC1Head at identity.

    # Right arm
    "arm_flex_r": [("jRightShoulder", 0, 1.0)],     # Flexion → Z
    "arm_add_r": [("jRightShoulder", 1, 1.0)],      # Adduction → X
    "arm_rot_r": [("jRightShoulder", 2, 1.0)],      # Rotation → Y
    "elbow_flex_r": [("jRightElbow", 0, 1.0)],      # Flexion → Z
    "pro_sup_r": [("jRightElbow", 2, 1.0)],         # Pronation/supination → Y
    "wrist_flex_r": [("jRightWrist", 0, 1.0)],      # Flexion → Z
    "wrist_dev_r": [("jRightWrist", 1, 1.0)],       # Deviation → X

    # Left arm
    "arm_flex_l": [("jLeftShoulder", 0, 1.0)],      # Flexion → Z
    "arm_add_l": [("jLeftShoulder", 1, 1.0)],       # Adduction → X
    "arm_rot_l": [("jLeftShoulder", 2, 1.0)],       # Rotation → Y
    "elbow_flex_l": [("jLeftElbow", 0, 1.0)],       # Flexion → Z
    "pro_sup_l": [("jLeftElbow", 2, 1.0)],          # Pronation/supination → Y
    "wrist_flex_l": [("jLeftWrist", 0, 1.0)],       # Flexion → Z
    "wrist_dev_l": [("jLeftWrist", 1, 1.0)],        # Deviation → X

    # Right leg
    "hip_flexion_r": [("jRightHip", 0, 1.0)],       # Flexion → Z
    "hip_adduction_r": [("jRightHip", 1, 1.0)],     # Adduction → X
    "hip_rotation_r": [("jRightHip", 2, 1.0)],      # Rotation → Y
    "knee_angle_r": [("jRightKnee", 0, 1.0)],       # Flexion → Z
    "ankle_angle_r": [("jRightAnkle", 0, 1.0)],     # Flexion → Z
    "subtalar_angle_r": [("jRightAnkle", 1, 1.0)],  # Inversion → X
    "mtp_angle_r": [("jRightBallFoot", 0, 1.0)],    # Flexion → Z

    # Left leg
    "hip_flexion_l": [("jLeftHip", 0, 1.0)],        # Flexion → Z
    "hip_adduction_l": [("jLeftHip", 1, 1.0)],      # Adduction → X
    "hip_rotation_l": [("jLeftHip", 2, 1.0)],       # Rotation → Y
    "knee_angle_l": [("jLeftKnee", 0, 1.0)],        # Flexion → Z
    "ankle_angle_l": [("jLeftAnkle", 0, 1.0)],      # Flexion → Z
    "subtalar_angle_l": [("jLeftAnkle", 1, 1.0)],   # Inversion → X
    "mtp_angle_l": [("jLeftBallFoot", 0, 1.0)],     # Flexion → Z
}

# Spine distribution weights: the single OpenSim spine DOF is distributed
# across 4 Xsens spine joints (jL5S1, jL4L3, jL1T12, jT9T8)
SPINE_WEIGHTS = {
    "jL5S1": 0.40,    # Lumbar-sacral gets the most motion
    "jL4L3": 0.25,    # Upper lumbar
    "jL1T12": 0.20,   # Thoraco-lumbar junction
    "jT9T8": 0.15,    # Mid-thoracic
}

# Shoulder scapula joints (jRightT4Shoulder, jLeftT4Shoulder) get small
# coupled motion from arm elevation. These are approximated.
SCAPULA_COUPLING_RATIO = 0.33  # Scapula contributes ~1/3 of arm elevation


def build_xsens_joint_angles_from_mot(
    mot_coordinates: Dict[str, np.ndarray],
    num_frames: int,
) -> np.ndarray:
    """
    Convert OpenSim MOT joint angles to Xsens joint angles (ZXY Euler, degrees).

    Args:
        mot_coordinates: Dict mapping OpenSim coordinate names to (T,) arrays in degrees
        num_frames: Number of frames

    Returns:
        (T, 22, 3) array of Xsens joint angles in ZXY Euler degrees
    """
    joint_angles = np.zeros((num_frames, JOINT_COUNT, 3))

    # Map each OpenSim DOF to the corresponding Xsens joint angle slot
    for osim_coord, mappings in OPENSIM_TO_XSENS_DOF.items():
        if osim_coord not in mot_coordinates:
            continue
        values = mot_coordinates[osim_coord]  # (T,) in degrees

        for joint_label, axis_idx, sign in mappings:
            joint_idx = JOINT_LABELS.index(joint_label)
            joint_angles[:, joint_idx, axis_idx] += sign * values

    # Distribute spine DOFs across 4 spine joints
    _distribute_spine(joint_angles, mot_coordinates)

    # Add scapula coupling from arm elevation
    _add_scapula_coupling(joint_angles)

    return joint_angles


def _distribute_spine(
    joint_angles: np.ndarray,
    mot_coordinates: Dict[str, np.ndarray],
):
    """Distribute the single OpenSim spine joint across 4 Xsens spine joints."""
    spine_dofs = [
        ("L5_S1_Flex_Ext", 0),      # Flexion → Z
        ("L5_S1_Lat_Bending", 1),   # Lateral → X
        ("L5_S1_axial_rotation", 2), # Rotation → Y
    ]

    for osim_name, axis_idx in spine_dofs:
        if osim_name not in mot_coordinates:
            continue
        total_angle = mot_coordinates[osim_name]  # (T,) degrees

        for joint_label, weight in SPINE_WEIGHTS.items():
            joint_idx = JOINT_LABELS.index(joint_label)
            joint_angles[:, joint_idx, axis_idx] = total_angle * weight


def _add_scapula_coupling(joint_angles: np.ndarray):
    """Add coupled scapula motion from arm elevation.

    Scapulohumeral rhythm: for every 3° of arm elevation, ~1° is from scapula.
    """
    # Right scapula from right shoulder flexion/abduction
    r_shoulder_idx = JOINT_LABELS.index("jRightShoulder")
    r_scapula_idx = JOINT_LABELS.index("jRightT4Shoulder")
    joint_angles[:, r_scapula_idx, :] = (
        joint_angles[:, r_shoulder_idx, :] * SCAPULA_COUPLING_RATIO
    )

    # Left scapula from left shoulder
    l_shoulder_idx = JOINT_LABELS.index("jLeftShoulder")
    l_scapula_idx = JOINT_LABELS.index("jLeftT4Shoulder")
    joint_angles[:, l_scapula_idx, :] = (
        joint_angles[:, l_shoulder_idx, :] * SCAPULA_COUPLING_RATIO
    )


def compute_segment_orientations(
    joint_angles_zxy: np.ndarray,
    pelvis_tilt: np.ndarray,
    pelvis_list: np.ndarray,
    pelvis_rotation: np.ndarray,
) -> np.ndarray:
    """
    Compute global segment orientations via forward kinematics.

    Takes joint angles in ZXY Euler (degrees) and computes global quaternions
    for each segment by traversing the kinematic chain from root (Pelvis).

    Args:
        joint_angles_zxy: (T, 22, 3) joint angles in ZXY Euler degrees
        pelvis_tilt: (T,) pelvis tilt angle in degrees (X rotation)
        pelvis_list: (T,) pelvis list angle in degrees (Z rotation)
        pelvis_rotation: (T,) pelvis rotation angle in degrees (Y rotation)

    Returns:
        (T, 23, 4) quaternions in scalar-first (w, x, y, z) order, global orientations
    """
    T = joint_angles_zxy.shape[0]
    orientations = np.zeros((T, SEGMENT_COUNT, 4))

    for t in range(T):
        # Pelvis global orientation from OpenSim pelvis angles
        # OpenSim pelvis: tilt=forward/backward, list=lateral, rotation=yaw
        # We compose as intrinsic ZXY: list(Z) * tilt(X) * rotation(Y)
        pelvis_euler = np.array([
            pelvis_list[t],
            pelvis_tilt[t],
            pelvis_rotation[t],
        ])
        R_pelvis = Rotation.from_euler('ZXY', pelvis_euler, degrees=True)
        orientations[t, 0, :] = _to_scalar_first(R_pelvis.as_quat())

        # Forward kinematics: traverse from pelvis to all children
        _forward_kinematics_frame(
            t, joint_angles_zxy[t], R_pelvis, orientations
        )

    return orientations


def _forward_kinematics_frame(
    t: int,
    joint_angles: np.ndarray,
    R_pelvis: Rotation,
    orientations: np.ndarray,
):
    """Compute global orientations for one frame via forward kinematics."""
    # Store computed rotations by segment name for chain traversal
    segment_rotations: Dict[str, Rotation] = {"Pelvis": R_pelvis}

    # Process segments in order (they're topologically sorted already)
    for seg_idx, seg_label in enumerate(SEGMENT_LABELS):
        if seg_label == "Pelvis":
            continue  # Already set

        parent_label = SEGMENT_PARENT.get(seg_label)
        if parent_label is None or parent_label not in segment_rotations:
            # Orphan segment — use identity
            orientations[t, seg_idx, :] = np.array([1.0, 0.0, 0.0, 0.0])
            segment_rotations[seg_label] = Rotation.identity()
            continue

        # Get parent global rotation
        R_parent = segment_rotations[parent_label]

        # Get the joint connecting parent to this segment
        joint_label = SEGMENT_TO_PARENT_JOINT.get(seg_label)
        if joint_label is None:
            R_child = R_parent
        else:
            joint_idx = JOINT_LABELS.index(joint_label)
            # Joint angle is ZXY Euler in degrees
            euler_zxy = joint_angles[joint_idx]
            R_local = Rotation.from_euler('ZXY', euler_zxy, degrees=True)
            # Global = parent * local
            R_child = R_parent * R_local

        segment_rotations[seg_label] = R_child
        orientations[t, seg_idx, :] = _to_scalar_first(R_child.as_quat())


def compute_tpose_positions(subject_height: float = 1.75) -> np.ndarray:
    """
    Compute T-pose segment positions for the MVNX identity frame.

    Returns positions in Xsens coordinate system (X=forward, Y=left, Z=up)
    for a standing T-pose with arms extended sideways.

    Based on reference dimensions from a 1.78m subject (Xsens MVN),
    scaled proportionally to the given subject height.

    Args:
        subject_height: Subject height in meters

    Returns:
        (23, 3) array of segment positions in meters, Xsens Z-up coords
    """
    scale = subject_height / 1.78

    # Reference T-pose positions for a 1.78m subject (from Xsens MVN data).
    # Coordinate system: X=forward, Y=left, Z=up.
    ref = np.array([
        [ 0.000000,  0.000000,  0.900000],  #  0 Pelvis
        [-0.011717,  0.000000,  1.005455],  #  1 L5
        [-0.011717,  0.000000,  1.122493],  #  2 L3
        [-0.011770,  0.000000,  1.229357],  #  3 T12
        [-0.011823,  0.000000,  1.336102],  #  4 T8
        [-0.011823,  0.000000,  1.485381],  #  5 Neck
        [-0.011709,  0.000000,  1.584776],  #  6 Head
        [-0.011823, -0.032589,  1.419812],  #  7 RightShoulder
        [-0.011823, -0.185798,  1.419812],  #  8 RightUpperArm
        [-0.011823, -0.512878,  1.419812],  #  9 RightForeArm
        [-0.011823, -0.780403,  1.419812],  # 10 RightHand
        [-0.011823,  0.032589,  1.419812],  # 11 LeftShoulder
        [-0.011823,  0.185798,  1.419812],  # 12 LeftUpperArm
        [-0.011823,  0.512878,  1.419812],  # 13 LeftForeArm
        [-0.011823,  0.780403,  1.419812],  # 14 LeftHand
        [-0.000016, -0.086917,  0.900141],  # 15 RightUpperLeg
        [-0.000028, -0.086917,  0.453344],  # 16 RightLowerLeg
        [-0.000049, -0.086917,  0.016445],  # 17 RightFoot
        [ 0.208830, -0.086917, -0.074883],  # 18 RightToe
        [-0.000013,  0.086917,  0.900117],  # 19 LeftUpperLeg
        [-0.000026,  0.086917,  0.453320],  # 20 LeftLowerLeg
        [-0.000047,  0.086917,  0.016422],  # 21 LeftFoot
        [ 0.208832,  0.086917, -0.074906],  # 22 LeftToe
    ])

    return ref * scale


def compute_segment_positions(
    trc_markers: np.ndarray,
    trc_marker_names: List[str],
    subject_height: float = 1.75,
) -> np.ndarray:
    """
    Compute segment origin positions from TRC marker data.

    Each Xsens segment origin is at the proximal joint of that segment.

    Args:
        trc_markers: (T, M, 3) marker positions in meters (OpenSim coords)
        trc_marker_names: List of marker names
        subject_height: Subject height in meters

    Returns:
        (T, 23, 3) segment positions in meters
    """
    T = trc_markers.shape[0]
    positions = np.zeros((T, SEGMENT_COUNT, 3))

    # Build marker name → index lookup
    marker_idx = {name: i for i, name in enumerate(trc_marker_names)}

    def get_marker(name: str) -> Optional[np.ndarray]:
        if name in marker_idx:
            return trc_markers[:, marker_idx[name], :]
        return None

    def midpoint(name1: str, name2: str) -> Optional[np.ndarray]:
        m1, m2 = get_marker(name1), get_marker(name2)
        if m1 is not None and m2 is not None:
            return (m1 + m2) / 2.0
        return None

    # Pelvis = midpoint of hips
    hip_mid = midpoint("LHip", "RHip")
    if hip_mid is not None:
        positions[:, 0, :] = hip_mid

    # Neck/shoulder midpoint
    shoulder_mid = midpoint("LShoulder", "RShoulder")
    neck_pos = get_marker("Neck")
    if neck_pos is None:
        neck_pos = shoulder_mid

    # Spine segments: interpolate between pelvis and neck
    if hip_mid is not None and neck_pos is not None:
        spine_vec = neck_pos - hip_mid
        # L5: 15% up from pelvis
        positions[:, 1, :] = hip_mid + 0.15 * spine_vec
        # L3: 30% up
        positions[:, 2, :] = hip_mid + 0.30 * spine_vec
        # T12: 45% up
        positions[:, 3, :] = hip_mid + 0.45 * spine_vec
        # T8: 65% up (shoulder level)
        positions[:, 4, :] = hip_mid + 0.65 * spine_vec

    # Neck: at neck marker or shoulder midpoint
    if neck_pos is not None:
        positions[:, 5, :] = neck_pos

    # Head: at nose or above neck
    head_pos = get_marker("Nose")
    if head_pos is not None:
        positions[:, 6, :] = head_pos
    elif neck_pos is not None:
        positions[:, 6, :] = neck_pos + np.array([0, 0.1, 0])

    # Shoulders
    r_shoulder = get_marker("RShoulder")
    l_shoulder = get_marker("LShoulder")
    if r_shoulder is not None:
        positions[:, 7, :] = r_shoulder  # RightShoulder (scapula)
        positions[:, 8, :] = r_shoulder  # RightUpperArm origin
    if l_shoulder is not None:
        positions[:, 11, :] = l_shoulder  # LeftShoulder
        positions[:, 12, :] = l_shoulder  # LeftUpperArm origin

    # Elbows
    r_elbow = get_marker("RElbow")
    l_elbow = get_marker("LElbow")
    if r_elbow is not None:
        positions[:, 9, :] = r_elbow  # RightForeArm
    if l_elbow is not None:
        positions[:, 13, :] = l_elbow  # LeftForeArm

    # Wrists/Hands
    r_wrist = get_marker("RWrist")
    l_wrist = get_marker("LWrist")
    if r_wrist is not None:
        positions[:, 10, :] = r_wrist  # RightHand
    if l_wrist is not None:
        positions[:, 14, :] = l_wrist  # LeftHand

    # Hips → UpperLeg
    r_hip = get_marker("RHip")
    l_hip = get_marker("LHip")
    if r_hip is not None:
        positions[:, 15, :] = r_hip  # RightUpperLeg
    if l_hip is not None:
        positions[:, 19, :] = l_hip  # LeftUpperLeg

    # Knees → LowerLeg
    r_knee = get_marker("RKnee")
    l_knee = get_marker("LKnee")
    if r_knee is not None:
        positions[:, 16, :] = r_knee  # RightLowerLeg
    if l_knee is not None:
        positions[:, 20, :] = l_knee  # LeftLowerLeg

    # Ankles → Foot
    r_ankle = get_marker("RAnkle")
    l_ankle = get_marker("LAnkle")
    if r_ankle is not None:
        positions[:, 17, :] = r_ankle  # RightFoot
    if l_ankle is not None:
        positions[:, 21, :] = l_ankle  # LeftFoot

    # Toes — offset forward from ankle
    if r_ankle is not None:
        positions[:, 18, :] = r_ankle + np.array([0.15, 0, 0])  # RightToe
    if l_ankle is not None:
        positions[:, 22, :] = l_ankle + np.array([0.15, 0, 0])  # LeftToe

    return positions


def compute_orientations_from_positions(
    positions: np.ndarray,
    tpose_positions: np.ndarray,
) -> np.ndarray:
    """
    Compute global segment orientations from per-frame positions.

    Uses bone direction vectors (segment origin → child joint) to determine
    each segment's rotation relative to T-pose. A secondary reference vector
    (lateral direction for spine, joint plane for limbs) constrains twist.

    This ensures orientations are consistent with position data, so
    FK reconstruction (parent_pos + R_parent * tpose_offset) produces a
    connected skeleton.

    Args:
        positions: (T, 23, 3) segment positions in Xsens Z-up frame
        tpose_positions: (23, 3) T-pose positions in Xsens Z-up frame

    Returns:
        (T, 23, 4) quaternions in scalar-first (w, x, y, z) order
    """
    T = positions.shape[0]
    orientations = np.zeros((T, SEGMENT_COUNT, 4))
    orientations[:, :, 0] = 1.0  # Default to identity

    # Primary bone direction: segment → child segment index
    BONE_CHILD = {
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6,  # Spine
        7: 8, 8: 9, 9: 10,                      # Right arm
        11: 12, 12: 13, 13: 14,                  # Left arm
        15: 16, 16: 17, 17: 18,                  # Right leg
        19: 20, 20: 21, 21: 22,                  # Left leg
    }

    for t in range(T):
        # Pre-compute normalized bone vectors
        bones_c = {}  # current frame
        bones_t = {}  # T-pose
        for seg_idx, child_idx in BONE_CHILD.items():
            bc = positions[t, child_idx] - positions[t, seg_idx]
            bt = tpose_positions[child_idx] - tpose_positions[seg_idx]
            bc_len = np.linalg.norm(bc)
            bt_len = np.linalg.norm(bt)
            if bc_len > 1e-6 and bt_len > 1e-6:
                bones_c[seg_idx] = bc / bc_len
                bones_t[seg_idx] = bt / bt_len

        # Lateral reference vectors
        lat_hip_c = positions[t, 19] - positions[t, 15]   # RUpperLeg → LUpperLeg
        lat_hip_t = tpose_positions[19] - tpose_positions[15]
        lat_sho_c = positions[t, 11] - positions[t, 7]    # RShoulder → LShoulder
        lat_sho_t = tpose_positions[11] - tpose_positions[7]

        for seg_idx in range(SEGMENT_COUNT):
            if seg_idx not in bones_c:
                # Leaf or coincident positions: copy parent orientation
                parent_label = SEGMENT_PARENT.get(SEGMENT_LABELS[seg_idx])
                if parent_label:
                    parent_idx = SEGMENT_LABELS.index(parent_label)
                    orientations[t, seg_idx, :] = orientations[t, parent_idx, :]
                continue

            primary_c = bones_c[seg_idx]
            primary_t = bones_t[seg_idx]

            # Determine secondary (twist) reference
            sec_c, sec_t = None, None

            if seg_idx == 0:  # Pelvis
                sec_c, sec_t = lat_hip_c, lat_hip_t
            elif 1 <= seg_idx <= 3:  # L5, L3, T12 (lower spine)
                sec_c, sec_t = lat_hip_c, lat_hip_t
            elif seg_idx in (4, 5):  # T8, Neck (upper spine)
                sec_c, sec_t = lat_sho_c, lat_sho_t
            elif seg_idx in (7, 11):  # Shoulders
                sec_c, sec_t = lat_sho_c, lat_sho_t
            elif seg_idx in (8, 12):  # Upper arms — elbow plane
                child_idx = BONE_CHILD[seg_idx]
                if child_idx in bones_c:
                    plane_c = np.cross(bones_c[seg_idx], bones_c[child_idx])
                    plane_t = np.cross(bones_t[seg_idx], bones_t[child_idx])
                    if np.linalg.norm(plane_c) > 1e-6 and np.linalg.norm(plane_t) > 1e-6:
                        sec_c, sec_t = plane_c, plane_t
            elif seg_idx in (15, 19):  # Upper legs — knee plane
                child_idx = BONE_CHILD[seg_idx]
                if child_idx in bones_c:
                    plane_c = np.cross(bones_c[seg_idx], bones_c[child_idx])
                    plane_t = np.cross(bones_t[seg_idx], bones_t[child_idx])
                    if np.linalg.norm(plane_c) > 1e-6 and np.linalg.norm(plane_t) > 1e-6:
                        sec_c, sec_t = plane_c, plane_t

            # Compute rotation via vector alignment
            if sec_c is not None and sec_t is not None:
                sc_len = np.linalg.norm(sec_c)
                st_len = np.linalg.norm(sec_t)
                if sc_len > 1e-6 and st_len > 1e-6:
                    R, _ = Rotation.align_vectors(
                        [primary_c, sec_c / sc_len],
                        [primary_t, sec_t / st_len],
                        weights=[1.0, 0.3],
                    )
                else:
                    R, _ = Rotation.align_vectors([primary_c], [primary_t])
            else:
                R, _ = Rotation.align_vectors([primary_c], [primary_t])

            orientations[t, seg_idx, :] = _to_scalar_first(R.as_quat())

    return orientations


def compute_fk_positions(
    orientations: np.ndarray,
    tpose_positions: np.ndarray,
    root_positions: np.ndarray,
) -> np.ndarray:
    """
    Compute FK-consistent positions from orientations and T-pose geometry.

    For each child segment:
        child_pos = parent_pos + R_parent * (child_tpose - parent_tpose)

    This guarantees a connected skeleton because positions are derived
    directly from orientations + bone offsets.

    Args:
        orientations: (T, 23, 4) scalar-first quaternions
        tpose_positions: (23, 3) T-pose positions in Xsens Z-up frame
        root_positions: (T, 3) pelvis positions per frame

    Returns:
        (T, 23, 3) FK-consistent positions
    """
    T = orientations.shape[0]
    positions = np.zeros((T, SEGMENT_COUNT, 3))

    # Pre-compute T-pose offsets (child - parent) for each joint
    offsets = {}  # child_seg_idx → offset in parent body frame
    for _, parent_seg, child_seg in JOINTS:
        parent_idx = SEGMENT_LABELS.index(parent_seg)
        child_idx = SEGMENT_LABELS.index(child_seg)
        offsets[child_idx] = tpose_positions[child_idx] - tpose_positions[parent_idx]

    for t in range(T):
        positions[t, 0] = root_positions[t]

        for seg_idx in range(1, SEGMENT_COUNT):
            seg_label = SEGMENT_LABELS[seg_idx]
            parent_label = SEGMENT_PARENT.get(seg_label)
            if parent_label is None:
                continue
            parent_idx = SEGMENT_LABELS.index(parent_label)

            q = orientations[t, parent_idx]
            R_parent = Rotation.from_quat([q[1], q[2], q[3], q[0]])

            positions[t, seg_idx] = (
                positions[t, parent_idx] + R_parent.apply(offsets[seg_idx])
            )

    return positions


def compute_joint_angles_from_orientations(
    orientations: np.ndarray,
) -> np.ndarray:
    """
    Derive ZXY joint angles from global segment orientations.

    Uses the MVNX formula: BABBq = GBAq* (x) GBBq
    Then decomposes into ZXY Euler angles (degrees).

    Args:
        orientations: (T, 23, 4) scalar-first quaternions

    Returns:
        (T, 22, 3) joint angles in ZXY Euler degrees
    """
    T = orientations.shape[0]
    joint_angles = np.zeros((T, JOINT_COUNT, 3))

    for t in range(T):
        for j_idx, (_, parent_seg, child_seg) in enumerate(JOINTS):
            parent_idx = SEGMENT_LABELS.index(parent_seg)
            child_idx = SEGMENT_LABELS.index(child_seg)

            q_p = orientations[t, parent_idx]
            q_c = orientations[t, child_idx]

            R_parent = Rotation.from_quat([q_p[1], q_p[2], q_p[3], q_p[0]])
            R_child = Rotation.from_quat([q_c[1], q_c[2], q_c[3], q_c[0]])

            # Local rotation: parent_inv * child
            R_local = R_parent.inv() * R_child
            joint_angles[t, j_idx, :] = R_local.as_euler('ZXY', degrees=True)

    return joint_angles


def compute_velocities(
    positions: np.ndarray,
    fps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity and acceleration by central finite differences.

    Args:
        positions: (T, N, 3) positions in meters
        fps: Frame rate

    Returns:
        velocities: (T, N, 3) in m/s
        accelerations: (T, N, 3) in m/s²
    """
    dt = 1.0 / fps
    T = positions.shape[0]

    # Velocity: central differences (forward/backward at endpoints)
    velocities = np.zeros_like(positions)
    if T >= 3:
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        velocities[0] = (positions[1] - positions[0]) / dt
        velocities[-1] = (positions[-1] - positions[-2]) / dt
    elif T == 2:
        v = (positions[1] - positions[0]) / dt
        velocities[0] = v
        velocities[1] = v

    # Acceleration: central differences of velocity
    accelerations = np.zeros_like(positions)
    if T >= 3:
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * dt)
        accelerations[0] = (velocities[1] - velocities[0]) / dt
        accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
    elif T == 2:
        a = (velocities[1] - velocities[0]) / dt
        accelerations[0] = a
        accelerations[1] = a

    return velocities, accelerations


def compute_angular_velocity(
    orientations: np.ndarray,
    fps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute angular velocity and angular acceleration from quaternion sequences.

    Uses quaternion finite differences: ω = 2 * (q(t+1) * q(t)^-1 - I) / dt

    Args:
        orientations: (T, N, 4) quaternions in scalar-first (w, x, y, z) order
        fps: Frame rate

    Returns:
        angular_velocities: (T, N, 3) in rad/s
        angular_accelerations: (T, N, 3) in rad/s²
    """
    T, N = orientations.shape[:2]
    dt = 1.0 / fps

    angular_velocities = np.zeros((T, N, 3))
    angular_accelerations = np.zeros((T, N, 3))

    for n in range(N):
        for t in range(T - 1):
            q_curr = _from_scalar_first(orientations[t, n])
            q_next = _from_scalar_first(orientations[t + 1, n])

            R_curr = Rotation.from_quat(q_curr)
            R_next = Rotation.from_quat(q_next)

            # Relative rotation: R_delta = R_next * R_curr^-1
            R_delta = R_next * R_curr.inv()
            rotvec = R_delta.as_rotvec()  # axis-angle representation
            angular_velocities[t, n] = rotvec / dt

        # Last frame: copy from previous
        if T > 1:
            angular_velocities[-1, n] = angular_velocities[-2, n]

    # Angular acceleration from velocity finite differences
    if T >= 3:
        angular_accelerations[1:-1] = (
            (angular_velocities[2:] - angular_velocities[:-2]) / (2 * dt)
        )
        angular_accelerations[0] = (
            (angular_velocities[1] - angular_velocities[0]) / dt
        )
        angular_accelerations[-1] = (
            (angular_velocities[-1] - angular_velocities[-2]) / dt
        )

    return angular_velocities, angular_accelerations


def detect_foot_contacts(
    trc_markers: np.ndarray,
    trc_marker_names: List[str],
    threshold: float = 0.02,
) -> np.ndarray:
    """
    Detect foot contacts from marker vertical positions.

    Args:
        trc_markers: (T, M, 3) marker positions in meters (OpenSim: Y-up)
        trc_marker_names: List of marker names
        threshold: Contact threshold in meters (default 2cm)

    Returns:
        (T, 4) binary array: [LeftHeel, LeftToe, RightHeel, RightToe]
    """
    T = trc_markers.shape[0]
    contacts = np.zeros((T, 4), dtype=int)

    marker_idx = {name: i for i, name in enumerate(trc_marker_names)}

    # Foot contact marker pairs
    contact_markers = [
        ("LAnkle", 0),    # Left heel proxy
        ("LAnkle", 1),    # Left toe proxy (using ankle if no toe marker)
        ("RAnkle", 2),    # Right heel proxy
        ("RAnkle", 3),    # Right toe proxy
    ]

    for marker_name, contact_idx in contact_markers:
        if marker_name in marker_idx:
            # Y is the vertical axis in OpenSim coordinate system
            heights = trc_markers[:, marker_idx[marker_name], 1]
            contacts[:, contact_idx] = (heights < threshold).astype(int)

    return contacts


def compute_center_of_mass(
    positions: np.ndarray,
) -> np.ndarray:
    """
    Approximate center of mass from segment positions.

    Uses simplified segment mass fractions based on standard anthropometric data.

    Args:
        positions: (T, 23, 3) segment positions

    Returns:
        (T, 3) center of mass position
    """
    # Approximate segment mass fractions (de Leva 1996)
    mass_fractions = np.zeros(SEGMENT_COUNT)
    mass_fractions[0] = 0.142   # Pelvis
    mass_fractions[1] = 0.035   # L5
    mass_fractions[2] = 0.035   # L3
    mass_fractions[3] = 0.035   # T12
    mass_fractions[4] = 0.081   # T8
    mass_fractions[5] = 0.015   # Neck
    mass_fractions[6] = 0.069   # Head
    mass_fractions[7] = 0.005   # RightShoulder
    mass_fractions[8] = 0.028   # RightUpperArm
    mass_fractions[9] = 0.016   # RightForeArm
    mass_fractions[10] = 0.006  # RightHand
    mass_fractions[11] = 0.005  # LeftShoulder
    mass_fractions[12] = 0.028  # LeftUpperArm
    mass_fractions[13] = 0.016  # LeftForeArm
    mass_fractions[14] = 0.006  # LeftHand
    mass_fractions[15] = 0.100  # RightUpperLeg
    mass_fractions[16] = 0.047  # RightLowerLeg
    mass_fractions[17] = 0.013  # RightFoot
    mass_fractions[18] = 0.002  # RightToe
    mass_fractions[19] = 0.100  # LeftUpperLeg
    mass_fractions[20] = 0.047  # LeftLowerLeg
    mass_fractions[21] = 0.013  # LeftFoot
    mass_fractions[22] = 0.002  # LeftToe

    # Normalize to sum to 1
    mass_fractions /= mass_fractions.sum()

    # CoM = weighted sum of segment positions
    # (T, 23, 3) * (23, 1) → (T, 3) via einsum
    com = np.einsum('tsc,s->tc', positions, mass_fractions)

    return com


def _to_scalar_first(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert scipy (x, y, z, w) quaternion to Xsens scalar-first (w, x, y, z)."""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _from_scalar_first(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert Xsens scalar-first (w, x, y, z) to scipy (x, y, z, w)."""
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
