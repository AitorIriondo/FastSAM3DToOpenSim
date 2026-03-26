"""
Vendored OpenSim-to-MVNX converter.
Source: https://github.com/AitorIriondo/OpenSim-to-MVNX

Public API:
    export_mvnx(trc_path, mot_path, output_path, fps, subject_height, ...) -> str
"""
from .src.mvnx_exporter import MVNXExporter


def export_mvnx(
    trc_path: str,
    mot_path: str,
    output_path: str,
    fps: float = 30.0,
    subject_height: float = 1.75,
    osim_path: str = None,
    original_filename: str = None,
) -> str:
    """
    Convert TRC + MOT → MVNX v4 (full Xsens format with orientations & joint angles).

    Args:
        trc_path: Path to TRC marker file
        mot_path: Path to OpenSim IK MOT file (joint angles)
        output_path: Output .mvnx file path
        fps: Frame rate in Hz
        subject_height: Subject height in metres
        osim_path: Optional path to .osim model
        original_filename: Optional source video filename for metadata

    Returns:
        Path to created .mvnx file
    """
    exporter = MVNXExporter(fps=fps, subject_height=subject_height)
    return exporter.export(
        trc_path=trc_path,
        mot_path=mot_path,
        output_path=output_path,
        osim_path=osim_path,
        original_filename=original_filename,
    )
