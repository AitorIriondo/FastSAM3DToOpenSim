"""
Vendored TRCtoIPS converter.
Source: https://github.com/AitorIriondo/TRCtoIPS

Public API:
    convert_trc_to_ipsmvnx(trc_path, output_path, original_filename) -> str
"""
from .src.mvnx_builder import convert_trc_to_mvnx as _convert_trc_to_mvnx


def convert_trc_to_ipsmvnx(
    trc_path: str,
    output_path: str,
    original_filename: str = None,
) -> str:
    """
    Convert a 73-marker TRC file → IPS MVNX (.ipsmvnx).

    The output uses the Xsens Z-up convention and contains only segment
    positions (no orientations / joint angles). Intended for Industrial Path
    Solutions (IPS IMMA) ergonomics simulation.

    Args:
        trc_path: Path to input TRC file (must contain all 73 expected markers)
        output_path: Output .ipsmvnx file path
        original_filename: Optional source filename for XML metadata

    Returns:
        Path to created .ipsmvnx file
    """
    _convert_trc_to_mvnx(
        trc_path=trc_path,
        output_path=output_path,
        original_filename=original_filename or trc_path,
    )
    return output_path
