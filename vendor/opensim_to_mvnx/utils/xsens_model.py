"""
Xsens MVN 23-segment body model definitions.

Defines the segment hierarchy, joint definitions, anthropometric proportions,
ergonomic joint angles, and foot contact points for the Xsens MVN system
as used in MVNX v4 files.

Reference: https://base.xsens.com/hc/en-us/articles/360012672099-MVNX-Version-4-File-Structure
"""

from typing import Dict, List, Tuple


# =============================================================================
# 23 Xsens segments with IDs (1-indexed as per MVNX spec)
# =============================================================================
SEGMENTS: List[Tuple[int, str]] = [
    (1, "Pelvis"),
    (2, "L5"),
    (3, "L3"),
    (4, "T12"),
    (5, "T8"),
    (6, "Neck"),
    (7, "Head"),
    (8, "RightShoulder"),
    (9, "RightUpperArm"),
    (10, "RightForeArm"),
    (11, "RightHand"),
    (12, "LeftShoulder"),
    (13, "LeftUpperArm"),
    (14, "LeftForeArm"),
    (15, "LeftHand"),
    (16, "RightUpperLeg"),
    (17, "RightLowerLeg"),
    (18, "RightFoot"),
    (19, "RightToe"),
    (20, "LeftUpperLeg"),
    (21, "LeftLowerLeg"),
    (22, "LeftFoot"),
    (23, "LeftToe"),
]

SEGMENT_LABELS: List[str] = [label for _, label in SEGMENTS]
SEGMENT_IDS: Dict[str, int] = {label: sid for sid, label in SEGMENTS}
SEGMENT_COUNT = 23


# =============================================================================
# 22 Xsens joints connecting parent -> child segments
# =============================================================================
JOINTS: List[Tuple[str, str, str]] = [
    # (joint_label, parent_segment, child_segment)
    ("jL5S1", "Pelvis", "L5"),
    ("jL4L3", "L5", "L3"),
    ("jL1T12", "L3", "T12"),
    ("jT9T8", "T12", "T8"),
    ("jT1C7", "T8", "Neck"),
    ("jC1Head", "Neck", "Head"),
    ("jRightT4Shoulder", "T8", "RightShoulder"),
    ("jRightShoulder", "RightShoulder", "RightUpperArm"),
    ("jRightElbow", "RightUpperArm", "RightForeArm"),
    ("jRightWrist", "RightForeArm", "RightHand"),
    ("jLeftT4Shoulder", "T8", "LeftShoulder"),
    ("jLeftShoulder", "LeftShoulder", "LeftUpperArm"),
    ("jLeftElbow", "LeftUpperArm", "LeftForeArm"),
    ("jLeftWrist", "LeftForeArm", "LeftHand"),
    ("jRightHip", "Pelvis", "RightUpperLeg"),
    ("jRightKnee", "RightUpperLeg", "RightLowerLeg"),
    ("jRightAnkle", "RightLowerLeg", "RightFoot"),
    ("jRightBallFoot", "RightFoot", "RightToe"),
    ("jLeftHip", "Pelvis", "LeftUpperLeg"),
    ("jLeftKnee", "LeftUpperLeg", "LeftLowerLeg"),
    ("jLeftAnkle", "LeftLowerLeg", "LeftFoot"),
    ("jLeftBallFoot", "LeftFoot", "LeftToe"),
]

JOINT_LABELS: List[str] = [label for label, _, _ in JOINTS]
JOINT_COUNT = 22

# Parent segment for each joint (for forward kinematics traversal)
JOINT_PARENT: Dict[str, str] = {label: parent for label, parent, _ in JOINTS}
JOINT_CHILD: Dict[str, str] = {label: child for label, _, child in JOINTS}

# Map child segment -> joint that connects to it
SEGMENT_TO_PARENT_JOINT: Dict[str, str] = {
    child: label for label, _, child in JOINTS
}

# Map segment -> its parent segment (via joints)
SEGMENT_PARENT: Dict[str, str] = {}
for _, parent, child in JOINTS:
    SEGMENT_PARENT[child] = parent
# Pelvis has no parent (root)


# =============================================================================
# Kinematic chain: ordered traversal from root (Pelvis) to leaves
# Used for forward kinematics computation
# =============================================================================
KINEMATIC_CHAINS: Dict[str, List[str]] = {
    # Spine chain
    "spine": ["Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head"],
    # Right arm
    "right_arm": ["T8", "RightShoulder", "RightUpperArm", "RightForeArm", "RightHand"],
    # Left arm
    "left_arm": ["T8", "LeftShoulder", "LeftUpperArm", "LeftForeArm", "LeftHand"],
    # Right leg
    "right_leg": ["Pelvis", "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe"],
    # Left leg
    "left_leg": ["Pelvis", "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe"],
}


# =============================================================================
# Default anatomical points per segment (joint positions in body frame)
# These define the segment coordinate frames and joint locations.
# Positions are in meters, relative to segment origin, for a ~1.75m subject.
# =============================================================================
SEGMENT_POINTS: Dict[str, List[Tuple[str, Tuple[float, float, float]]]] = {
    # Full anatomical points matching Xsens MVN format.
    # Each segment has incoming joint at origin + outgoing joints + anatomical landmarks.
    # Values from a ~1.78m subject; positions in meters, Xsens body frame (X-fwd, Y-lat, Z-up).
    "Pelvis": [
        ("pHipOrigin", (0.0, 0.0, 0.0)),
        ("jL5S1", (-0.012, 0.0, 0.105)),
        ("jRightHip", (0.0, -0.087, 0.0)),
        ("jLeftHip", (0.0, 0.087, 0.0)),
        ("pRightSIPS", (-0.06, -0.05, 0.05)),
        ("pLeftSIPS", (-0.06, 0.05, 0.05)),
        ("pRightASI", (0.08, -0.12, 0.02)),
        ("pLeftASI", (0.08, 0.12, 0.02)),
        ("pRightCSI", (0.0, -0.14, -0.03)),
        ("pLeftCSI", (0.0, 0.14, -0.03)),
        ("pRightIschialTub", (-0.06, -0.06, -0.08)),
        ("pLeftIschialTub", (-0.06, 0.06, -0.08)),
        ("pSacrum", (-0.08, 0.0, 0.05)),
        ("pCentralButtock", (-0.09, 0.0, -0.03)),
        ("pThoracolumbarFascia", (-0.06, 0.0, 0.105)),
    ],
    "L5": [
        ("jL5S1", (0.0, 0.0, 0.0)),
        ("jL4L3", (0.0, 0.0, 0.117038)),
        ("pL5SpinalProcess", (-0.032686, 0.0, -0.000261)),
    ],
    "L3": [
        ("jL4L3", (0.0, 0.0, 0.0)),
        ("jL1T12", (-5.3e-05, 0.0, 0.106864)),
        ("pL3SpinalProcess", (-0.032565, 0.0, 0.003011)),
    ],
    "T12": [
        ("jL1T12", (0.0, 0.0, 0.0)),
        ("jT9T8", (-5.3e-05, 0.0, 0.106745)),
        ("pT12SpinalProcess", (-0.032511, 0.0, 0.003011)),
    ],
    "T8": [
        ("jT9T8", (0.0, 0.0, 0.0)),
        ("jT1C7", (0.0, 0.0, 0.149279)),
        ("jRightT4Shoulder", (0.0, -0.032589, 0.08371)),
        ("jLeftT4Shoulder", (0.0, 0.032589, 0.08371)),
        ("pPX", (0.130356, 0.0, -0.003194)),
        ("pIJ", (0.124925, 0.0, 0.127162)),
        ("pT4SpinalProcess", (-0.032589, 0.0, 0.095225)),
        ("pT8SpinalProcess", (-0.032589, 0.0, -0.000261)),
    ],
    "Neck": [
        ("jT1C7", (0.0, 0.0, 0.0)),
        ("jC1Head", (0.000114, 0.0, 0.099395)),
        ("pC7SpinalProcess", (-0.03187, 0.0, -0.007397)),
    ],
    "Head": [
        ("jC1Head", (0.0, 0.0, 0.0)),
        ("pTopOfHead", (1.5e-05, 0.0, 0.184193)),
        ("pRightAuricularis", (0.006385, -0.075928, 0.076173)),
        ("pLeftAuricularis", (0.006385, 0.075928, 0.076173)),
        ("pBackOfHead", (-0.05317, 0.0, 0.07267)),
        ("pCenterOfHead", (0.006385, 0.0, 0.081091)),
        ("pNose", (0.081366, 0.0, 0.081091)),
        ("pRightEye", (0.059673, -0.033625, 0.097361)),
        ("pLeftEye", (0.059673, 0.033625, 0.097361)),
    ],
    "RightShoulder": [
        ("jRightT4Shoulder", (0.0, 0.0, 0.0)),
        ("jRightShoulder", (0.0, -0.153209, 0.0)),
        ("pRightAcromion", (0.0, -0.175096, 0.054717)),
    ],
    "RightUpperArm": [
        ("jRightShoulder", (0.0, 0.0, 0.0)),
        ("jRightElbow", (0.0, -0.32708, 0.0)),
        ("pRightArmLatEpicondyle", (0.0, -0.32708, 0.032708)),
        ("pRightArmMedEpicondyle", (0.0, -0.32708, -0.038159)),
    ],
    "RightForeArm": [
        ("jRightElbow", (0.0, 0.0, 0.0)),
        ("jRightWrist", (0.0, -0.267525, 0.0)),
        ("pRightUlnarStyloid", (-0.027253, -0.266413, 0.0)),
        ("pRightRadialStyloid", (0.027253, -0.268638, 0.0)),
        ("pRightOlecranon", (-0.027253, 0.001112, 0.0)),
        ("pRightUlnarPosteriorBorder", (-0.027253, -0.134197, 0.0)),
    ],
    "RightHand": [
        ("jRightWrist", (0.0, 0.0, 0.0)),
        ("pRightTopOfHand", (0.0, -0.200157, 0.0)),
        ("pRightPinky", (-0.04375, -0.114844, 0.0)),
        ("pRightBallHand", (-0.021875, -0.038281, -0.010938)),
        ("pRightHandPalm", (0.021875, -0.098438, -0.010938)),
        ("pRightHandController", (0.098438, -0.147657, -0.010938)),
    ],
    "LeftShoulder": [
        ("jLeftT4Shoulder", (0.0, 0.0, 0.0)),
        ("jLeftShoulder", (0.0, 0.153209, 0.0)),
        ("pLeftAcromion", (0.0, 0.175096, 0.054717)),
    ],
    "LeftUpperArm": [
        ("jLeftShoulder", (0.0, 0.0, 0.0)),
        ("jLeftElbow", (0.0, 0.32708, 0.0)),
        ("pLeftArmLatEpicondyle", (0.0, 0.32708, 0.032708)),
        ("pLeftArmMedEpicondyle", (0.0, 0.32708, -0.038159)),
    ],
    "LeftForeArm": [
        ("jLeftElbow", (0.0, 0.0, 0.0)),
        ("jLeftWrist", (0.0, 0.267525, 0.0)),
        ("pLeftUlnarStyloid", (-0.027253, 0.266413, 0.0)),
        ("pLeftRadialStyloid", (0.027253, 0.268638, 0.0)),
        ("pLeftOlecranon", (-0.027253, -0.001112, 0.0)),
        ("pLeftUlnarPosteriorBorder", (-0.027253, 0.134197, 0.0)),
    ],
    "LeftHand": [
        ("jLeftWrist", (0.0, 0.0, 0.0)),
        ("pLeftTopOfHand", (0.0, 0.200157, 0.0)),
        ("pLeftPinky", (-0.04375, 0.114844, 0.0)),
        ("pLeftBallHand", (-0.021875, 0.038281, -0.010938)),
        ("pLeftHandPalm", (0.021875, 0.098438, -0.010938)),
        ("pLeftHandController", (0.098438, 0.147657, -0.010938)),
    ],
    "RightUpperLeg": [
        ("jRightHip", (0.0, 0.0, 0.0)),
        ("jRightKnee", (-1.3e-05, 0.0, -0.446797)),
        ("pRightGreaterTrochanter", (0.001592, -0.075139, -0.033107)),
        ("pRightKneeLatEpicondyle", (-0.007512, -0.042937, -0.447288)),
        ("pRightKneeMedEpicondyle", (-0.007512, 0.03757, -0.447288)),
        ("pRightPatella", (0.03752, 0.0, -0.445123)),
    ],
    "RightLowerLeg": [
        ("jRightKnee", (0.0, 0.0, 0.0)),
        ("jRightAnkle", (-2.1e-05, 0.0, -0.436899)),
        ("pRightLatMalleolus", (-1.1e-05, -0.032275, -0.437029)),
        ("pRightMedMalleolus", (-1.1e-05, 0.032275, -0.437029)),
        ("pRightTibialTub", (0.051236, 0.0, -0.033832)),
        ("pRightFibula", (-1.1e-05, -0.032275, -0.290346)),
    ],
    "RightFoot": [
        ("jRightAnkle", (0.0, 0.0, 0.0)),
        ("jRightBallFoot", (0.174, 0.0, -0.048)),
        ("pRightHeelFoot", (-0.065, 0.0, -0.048)),
        ("pRightFirstMetatarsal", (0.174, 0.025, -0.028)),
        ("pRightFifthMetatarsal", (0.174, -0.025, -0.028)),
        ("pRightPivotFoot", (0.120, 0.0, -0.048)),
        ("pRightHeelCenter", (-0.047, 0.0, -0.048)),
        ("pRightTopOfFoot", (0.091, 0.0, 0.011)),
    ],
    "RightToe": [
        ("jRightBallFoot", (0.0, 0.0, 0.0)),
        ("pRightToe", (0.050, 0.0, 0.0)),
        ("jRightToe", (0.050, 0.0, 0.0)),
    ],
    "LeftUpperLeg": [
        ("jLeftHip", (0.0, 0.0, 0.0)),
        ("jLeftKnee", (-1.3e-05, 0.0, -0.446797)),
        ("pLeftGreaterTrochanter", (0.001592, 0.075139, -0.033107)),
        ("pLeftKneeLatEpicondyle", (-0.007512, 0.042937, -0.447288)),
        ("pLeftKneeMedEpicondyle", (-0.007512, -0.03757, -0.447288)),
        ("pLeftPatella", (0.03752, 0.0, -0.445123)),
    ],
    "LeftLowerLeg": [
        ("jLeftKnee", (0.0, 0.0, 0.0)),
        ("jLeftAnkle", (-2.1e-05, 0.0, -0.436899)),
        ("pLeftLatMalleolus", (-1.1e-05, 0.032275, -0.437029)),
        ("pLeftMedMalleolus", (-1.1e-05, -0.032275, -0.437029)),
        ("pLeftTibialTub", (0.051236, 0.0, -0.033832)),
        ("pLeftFibula", (-1.1e-05, 0.032275, -0.290346)),
    ],
    "LeftFoot": [
        ("jLeftAnkle", (0.0, 0.0, 0.0)),
        ("jLeftBallFoot", (0.174, 0.0, -0.048)),
        ("pLeftHeelFoot", (-0.065, 0.0, -0.048)),
        ("pLeftFirstMetatarsal", (0.174, -0.025, -0.028)),
        ("pLeftFifthMetatarsal", (0.174, 0.025, -0.028)),
        ("pLeftPivotFoot", (0.120, 0.0, -0.048)),
        ("pLeftHeelCenter", (-0.047, 0.0, -0.048)),
        ("pLeftTopOfFoot", (0.091, 0.0, 0.011)),
    ],
    "LeftToe": [
        ("jLeftBallFoot", (0.0, 0.0, 0.0)),
        ("pLeftToe", (0.050, 0.0, 0.0)),
        ("jLeftToe", (0.050, 0.0, 0.0)),
    ],
}


# =============================================================================
# Anthropometric proportions (fraction of body height)
# Used to estimate segment lengths when no model data is available
# =============================================================================
ANTHROPOMETRIC_PROPORTIONS: Dict[str, float] = {
    # Segment length as fraction of total body height
    "Pelvis": 0.057,        # Pelvis height (hip to L5S1)
    "L5": 0.040,            # L5 to L3
    "L3": 0.040,            # L3 to T12
    "T12": 0.051,           # T12 to T8
    "T8": 0.097,            # T8 to neck
    "Neck": 0.040,          # Neck to head
    "Head": 0.130,          # Head height
    "RightShoulder": 0.057, # Shoulder breadth (half)
    "RightUpperArm": 0.160, # Upper arm length
    "RightForeArm": 0.143,  # Forearm length
    "RightHand": 0.057,     # Hand length
    "LeftShoulder": 0.057,
    "LeftUpperArm": 0.160,
    "LeftForeArm": 0.143,
    "LeftHand": 0.057,
    "RightUpperLeg": 0.240, # Thigh length
    "RightLowerLeg": 0.228, # Shank length
    "RightFoot": 0.057,     # Foot height
    "RightToe": 0.040,      # Toe length
    "LeftUpperLeg": 0.240,
    "LeftLowerLeg": 0.228,
    "LeftFoot": 0.057,
    "LeftToe": 0.040,
}


# =============================================================================
# Ergonomic joint angle definitions
# =============================================================================
ERGONOMIC_JOINT_ANGLES: List[Tuple[str, str, str]] = [
    # (label, parent_segment, child_segment) — matches Xsens MVN reference
    ("T8_Head", "T8", "Head"),
    ("T8_LeftUpperArm", "T8", "LeftUpperArm"),
    ("T8_RightUpperArm", "T8", "RightUpperArm"),
    ("Pelvis_T8", "Pelvis", "T8"),
    ("Vertical_Pelvis", "Vertical", "Pelvis"),
    ("Vertical_T8", "Vertical", "T8"),
]


# =============================================================================
# Foot contact definitions
# =============================================================================
FOOT_CONTACT_DEFINITIONS: List[Tuple[str, int]] = [
    ("LeftFoot_Heel", 0),
    ("LeftFoot_Toe", 1),
    ("RightFoot_Heel", 2),
    ("RightFoot_Toe", 3),
]


# =============================================================================
# 17 Xsens sensor labels (MVN uses 17 IMU sensors for the full body)
# In video-derived data, we don't have real sensors, but the count is needed
# for the MVNX schema.
# =============================================================================
SENSOR_COUNT = 17
SENSOR_LABELS: List[str] = [
    "Pelvis",
    "T8",
    "Head",
    "RightShoulder",
    "RightUpperArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftForeArm",
    "LeftHand",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
]
