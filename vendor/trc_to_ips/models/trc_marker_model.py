"""
TRC marker model for MVNX export.

Defines the 73 body landmark segments (in TRC file order) and the kinematic
tree (72 joints) that connects them. This structure maps directly to the
MVNX <segments> and <joints> sections so that IPS can interpret the
position data correctly.

Segment order matches the TRC marker order exactly:
  Nose, LEye, REye, LEar, REar,
  LShoulder, RShoulder, LElbow, RElbow,
  LHip, RHip, LKnee, RKnee, LAnkle, RAnkle,
  LBigToe, LSmallToe, LHeel, RBigToe, RSmallToe, RHeel,
  RThumbTip, RThumb1, RThumb2, RThumb3,
  RIndexTip, RIndex1, RIndex2, RIndex3,
  RMiddleTip, RMiddle1, RMiddle2, RMiddle3,
  RRingTip, RRing1, RRing2, RRing3,
  RPinkyTip, RPinky1, RPinky2, RPinky3, RWrist,
  LThumbTip, LThumb1, LThumb2, LThumb3,
  LIndexTip, LIndex1, LIndex2, LIndex3,
  LMiddleTip, LMiddle1, LMiddle2, LMiddle3,
  LRingTip, LRing1, LRing2, LRing3,
  LPinkyTip, LPinky1, LPinky2, LPinky3, LWrist,
  LOlecranon, ROlecranon, LCubitalFossa, RCubitalFossa,
  LAcromion, RAcromion, Neck,
  PelvisCenter, Thorax, SpineMid

Joints are listed in order of their child segment's TRC index,
so the MVNX structure mirrors the TRC ordering.
"""

from typing import Dict, List, Tuple


# =============================================================================
# 73 segments in TRC marker order (1-indexed IDs)
# =============================================================================
SEGMENTS: List[Tuple[int, str]] = [
    (1,  "Nose"),
    (2,  "LEye"),
    (3,  "REye"),
    (4,  "LEar"),
    (5,  "REar"),
    (6,  "LShoulder"),
    (7,  "RShoulder"),
    (8,  "LElbow"),
    (9,  "RElbow"),
    (10, "LHip"),
    (11, "RHip"),
    (12, "LKnee"),
    (13, "RKnee"),
    (14, "LAnkle"),
    (15, "RAnkle"),
    (16, "LBigToe"),
    (17, "LSmallToe"),
    (18, "LHeel"),
    (19, "RBigToe"),
    (20, "RSmallToe"),
    (21, "RHeel"),
    (22, "RThumbTip"),
    (23, "RThumb1"),
    (24, "RThumb2"),
    (25, "RThumb3"),
    (26, "RIndexTip"),
    (27, "RIndex1"),
    (28, "RIndex2"),
    (29, "RIndex3"),
    (30, "RMiddleTip"),
    (31, "RMiddle1"),
    (32, "RMiddle2"),
    (33, "RMiddle3"),
    (34, "RRingTip"),
    (35, "RRing1"),
    (36, "RRing2"),
    (37, "RRing3"),
    (38, "RPinkyTip"),
    (39, "RPinky1"),
    (40, "RPinky2"),
    (41, "RPinky3"),
    (42, "RWrist"),
    (43, "LThumbTip"),
    (44, "LThumb1"),
    (45, "LThumb2"),
    (46, "LThumb3"),
    (47, "LIndexTip"),
    (48, "LIndex1"),
    (49, "LIndex2"),
    (50, "LIndex3"),
    (51, "LMiddleTip"),
    (52, "LMiddle1"),
    (53, "LMiddle2"),
    (54, "LMiddle3"),
    (55, "LRingTip"),
    (56, "LRing1"),
    (57, "LRing2"),
    (58, "LRing3"),
    (59, "LPinkyTip"),
    (60, "LPinky1"),
    (61, "LPinky2"),
    (62, "LPinky3"),
    (63, "LWrist"),
    (64, "LOlecranon"),
    (65, "ROlecranon"),
    (66, "LCubitalFossa"),
    (67, "RCubitalFossa"),
    (68, "LAcromion"),
    (69, "RAcromion"),
    (70, "Neck"),
    (71, "PelvisCenter"),
    (72, "Thorax"),
    (73, "SpineMid"),
]

SEGMENT_LABELS: List[str] = [label for _, label in SEGMENTS]
SEGMENT_IDS: Dict[str, int] = {label: sid for sid, label in SEGMENTS}
SEGMENT_COUNT: int = len(SEGMENTS)

# Root segment (has no parent joint)
ROOT_SEGMENT = "PelvisCenter"


# =============================================================================
# 72 joints (kinematic tree), listed in child-segment TRC-index order.
#
# Each entry: (joint_label, parent_segment, child_segment)
#
# Kinematic hierarchy:
#   PelvisCenter (root)
#     └─ SpineMid ─ Thorax ─ Neck ─ Nose ─ LEye/REye/LEar/REar
#                         ├─ LAcromion ─ LShoulder ─ LElbow ─ LOlecranon (leaf)
#                         │                                 ├─ LCubitalFossa (leaf)
#                         │                                 └─ LWrist ─ finger chains
#                         └─ RAcromion ─ RShoulder ─ RElbow ─ ROlecranon (leaf)
#                                                           ├─ RCubitalFossa (leaf)
#                                                           └─ RWrist ─ finger chains
#     ├─ LHip ─ LKnee ─ LAnkle ─ LBigToe / LSmallToe / LHeel
#     └─ RHip ─ RKnee ─ RAnkle ─ RBigToe / RSmallToe / RHeel
# =============================================================================
JOINTS: List[Tuple[str, str, str]] = [
    # child TRC idx 0  (Nose)
    ("jNeck_Nose",                 "Neck",          "Nose"),
    # child TRC idx 1  (LEye)
    ("jNose_LEye",                 "Nose",          "LEye"),
    # child TRC idx 2  (REye)
    ("jNose_REye",                 "Nose",          "REye"),
    # child TRC idx 3  (LEar)
    ("jNose_LEar",                 "Nose",          "LEar"),
    # child TRC idx 4  (REar)
    ("jNose_REar",                 "Nose",          "REar"),
    # child TRC idx 5  (LShoulder)
    ("jLAcromion_LShoulder",       "LAcromion",     "LShoulder"),
    # child TRC idx 6  (RShoulder)
    ("jRAcromion_RShoulder",       "RAcromion",     "RShoulder"),
    # child TRC idx 7  (LElbow)
    ("jLShoulder_LElbow",          "LShoulder",     "LElbow"),
    # child TRC idx 8  (RElbow)
    ("jRShoulder_RElbow",          "RShoulder",     "RElbow"),
    # child TRC idx 9  (LHip)
    ("jPelvisCenter_LHip",         "PelvisCenter",  "LHip"),
    # child TRC idx 10 (RHip)
    ("jPelvisCenter_RHip",         "PelvisCenter",  "RHip"),
    # child TRC idx 11 (LKnee)
    ("jLHip_LKnee",                "LHip",          "LKnee"),
    # child TRC idx 12 (RKnee)
    ("jRHip_RKnee",                "RHip",          "RKnee"),
    # child TRC idx 13 (LAnkle)
    ("jLKnee_LAnkle",              "LKnee",         "LAnkle"),
    # child TRC idx 14 (RAnkle)
    ("jRKnee_RAnkle",              "RKnee",         "RAnkle"),
    # child TRC idx 15 (LBigToe)
    ("jLAnkle_LBigToe",            "LAnkle",        "LBigToe"),
    # child TRC idx 16 (LSmallToe)
    ("jLAnkle_LSmallToe",          "LAnkle",        "LSmallToe"),
    # child TRC idx 17 (LHeel)
    ("jLAnkle_LHeel",              "LAnkle",        "LHeel"),
    # child TRC idx 18 (RBigToe)
    ("jRAnkle_RBigToe",            "RAnkle",        "RBigToe"),
    # child TRC idx 19 (RSmallToe)
    ("jRAnkle_RSmallToe",          "RAnkle",        "RSmallToe"),
    # child TRC idx 20 (RHeel)
    ("jRAnkle_RHeel",              "RAnkle",        "RHeel"),
    # child TRC idx 21 (RThumbTip)
    ("jRThumb1_RThumbTip",         "RThumb1",       "RThumbTip"),
    # child TRC idx 22 (RThumb1)
    ("jRThumb2_RThumb1",           "RThumb2",       "RThumb1"),
    # child TRC idx 23 (RThumb2)
    ("jRThumb3_RThumb2",           "RThumb3",       "RThumb2"),
    # child TRC idx 24 (RThumb3)
    ("jRWrist_RThumb3",            "RWrist",        "RThumb3"),
    # child TRC idx 25 (RIndexTip)
    ("jRIndex1_RIndexTip",         "RIndex1",       "RIndexTip"),
    # child TRC idx 26 (RIndex1)
    ("jRIndex2_RIndex1",           "RIndex2",       "RIndex1"),
    # child TRC idx 27 (RIndex2)
    ("jRIndex3_RIndex2",           "RIndex3",       "RIndex2"),
    # child TRC idx 28 (RIndex3)
    ("jRWrist_RIndex3",            "RWrist",        "RIndex3"),
    # child TRC idx 29 (RMiddleTip)
    ("jRMiddle1_RMiddleTip",       "RMiddle1",      "RMiddleTip"),
    # child TRC idx 30 (RMiddle1)
    ("jRMiddle2_RMiddle1",         "RMiddle2",      "RMiddle1"),
    # child TRC idx 31 (RMiddle2)
    ("jRMiddle3_RMiddle2",         "RMiddle3",      "RMiddle2"),
    # child TRC idx 32 (RMiddle3)
    ("jRWrist_RMiddle3",           "RWrist",        "RMiddle3"),
    # child TRC idx 33 (RRingTip)
    ("jRRing1_RRingTip",           "RRing1",        "RRingTip"),
    # child TRC idx 34 (RRing1)
    ("jRRing2_RRing1",             "RRing2",        "RRing1"),
    # child TRC idx 35 (RRing2)
    ("jRRing3_RRing2",             "RRing3",        "RRing2"),
    # child TRC idx 36 (RRing3)
    ("jRWrist_RRing3",             "RWrist",        "RRing3"),
    # child TRC idx 37 (RPinkyTip)
    ("jRPinky1_RPinkyTip",         "RPinky1",       "RPinkyTip"),
    # child TRC idx 38 (RPinky1)
    ("jRPinky2_RPinky1",           "RPinky2",       "RPinky1"),
    # child TRC idx 39 (RPinky2)
    ("jRPinky3_RPinky2",           "RPinky3",       "RPinky2"),
    # child TRC idx 40 (RPinky3)
    ("jRWrist_RPinky3",            "RWrist",        "RPinky3"),
    # child TRC idx 41 (RWrist)
    ("jRElbow_RWrist",             "RElbow",        "RWrist"),
    # child TRC idx 42 (LThumbTip)
    ("jLThumb1_LThumbTip",         "LThumb1",       "LThumbTip"),
    # child TRC idx 43 (LThumb1)
    ("jLThumb2_LThumb1",           "LThumb2",       "LThumb1"),
    # child TRC idx 44 (LThumb2)
    ("jLThumb3_LThumb2",           "LThumb3",       "LThumb2"),
    # child TRC idx 45 (LThumb3)
    ("jLWrist_LThumb3",            "LWrist",        "LThumb3"),
    # child TRC idx 46 (LIndexTip)
    ("jLIndex1_LIndexTip",         "LIndex1",       "LIndexTip"),
    # child TRC idx 47 (LIndex1)
    ("jLIndex2_LIndex1",           "LIndex2",       "LIndex1"),
    # child TRC idx 48 (LIndex2)
    ("jLIndex3_LIndex2",           "LIndex3",       "LIndex2"),
    # child TRC idx 49 (LIndex3)
    ("jLWrist_LIndex3",            "LWrist",        "LIndex3"),
    # child TRC idx 50 (LMiddleTip)
    ("jLMiddle1_LMiddleTip",       "LMiddle1",      "LMiddleTip"),
    # child TRC idx 51 (LMiddle1)
    ("jLMiddle2_LMiddle1",         "LMiddle2",      "LMiddle1"),
    # child TRC idx 52 (LMiddle2)
    ("jLMiddle3_LMiddle2",         "LMiddle3",      "LMiddle2"),
    # child TRC idx 53 (LMiddle3)
    ("jLWrist_LMiddle3",           "LWrist",        "LMiddle3"),
    # child TRC idx 54 (LRingTip)
    ("jLRing1_LRingTip",           "LRing1",        "LRingTip"),
    # child TRC idx 55 (LRing1)
    ("jLRing2_LRing1",             "LRing2",        "LRing1"),
    # child TRC idx 56 (LRing2)
    ("jLRing3_LRing2",             "LRing3",        "LRing2"),
    # child TRC idx 57 (LRing3)
    ("jLWrist_LRing3",             "LWrist",        "LRing3"),
    # child TRC idx 58 (LPinkyTip)
    ("jLPinky1_LPinkyTip",         "LPinky1",       "LPinkyTip"),
    # child TRC idx 59 (LPinky1)
    ("jLPinky2_LPinky1",           "LPinky2",       "LPinky1"),
    # child TRC idx 60 (LPinky2)
    ("jLPinky3_LPinky2",           "LPinky3",       "LPinky2"),
    # child TRC idx 61 (LPinky3)
    ("jLWrist_LPinky3",            "LWrist",        "LPinky3"),
    # child TRC idx 62 (LWrist)
    ("jLElbow_LWrist",             "LElbow",        "LWrist"),
    # child TRC idx 63 (LOlecranon)
    ("jLElbow_LOlecranon",         "LElbow",        "LOlecranon"),
    # child TRC idx 64 (ROlecranon)
    ("jRElbow_ROlecranon",         "RElbow",        "ROlecranon"),
    # child TRC idx 65 (LCubitalFossa)
    ("jLElbow_LCubitalFossa",      "LElbow",        "LCubitalFossa"),
    # child TRC idx 66 (RCubitalFossa)
    ("jRElbow_RCubitalFossa",      "RElbow",        "RCubitalFossa"),
    # child TRC idx 67 (LAcromion)
    ("jThorax_LAcromion",          "Thorax",        "LAcromion"),
    # child TRC idx 68 (RAcromion)
    ("jThorax_RAcromion",          "Thorax",        "RAcromion"),
    # child TRC idx 69 (Neck)
    ("jThorax_Neck",               "Thorax",        "Neck"),
    # PelvisCenter (idx 70) is ROOT — no joint
    # child TRC idx 71 (Thorax)
    ("jSpineMid_Thorax",           "SpineMid",      "Thorax"),
    # child TRC idx 72 (SpineMid)
    ("jPelvisCenter_SpineMid",     "PelvisCenter",  "SpineMid"),
]

JOINT_LABELS: List[str] = [label for label, _, _ in JOINTS]
JOINT_COUNT: int = len(JOINTS)
