"""
File I/O and project directory management.

Numerical geometry is normalized to millimeters before entering
the ISF thickness pipeline.
"""

import os
import csv
import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_DIRS = {
    "input": os.path.join(PROJECT_ROOT, "data", "input"),
    "patches": os.path.join(PROJECT_ROOT, "data", "intermediate", "patches"),
    "contours": os.path.join(PROJECT_ROOT, "data", "intermediate", "contours"),
    "flattened": os.path.join(
        PROJECT_ROOT, "data", "output", "flattened_patches"
    ),
    "thickness_maps": os.path.join(
        PROJECT_ROOT, "data", "output", "thickness_maps"
    ),
    "visualizations": os.path.join(
        PROJECT_ROOT, "data", "output", "visualizations"
    ),
}


def ensure_data_dirs():
    """Create all project data directories."""
    for path in DATA_DIRS.values():
        os.makedirs(path, exist_ok=True)


def save_thickness_csv(
    path: str,
    r: np.ndarray,
    z: np.ndarray,
    theta_deg: np.ndarray,
    t0: np.ndarray,
):
    """
    Save thickness distribution.

    Parameters
    ----------
    r : radial coordinate [mm]
    z : axial coordinate [mm]
    theta_deg : tangent/wall angle from horizontal [deg]
    t0 : initial blank thickness [mm]
    """

    r = np.asarray(r)
    z = np.asarray(z)
    theta_deg = np.asarray(theta_deg)
    t0 = np.asarray(t0)

    if not (
        len(r)
        == len(z)
        == len(theta_deg)
        == len(t0)
    ):
        raise ValueError("CSV arrays must have identical lengths.")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "r_mm",
            "z_mm",
            "theta_deg",
            "T0_mm",
        ])

        for ri, zi, ti, t0i in zip(
            r, z, theta_deg, t0
        ):
            writer.writerow([
                f"{ri:.6f}",
                f"{zi:.6f}",
                f"{ti:.6f}",
                f"{t0i:.6f}",
            ])


def load_thickness_csv(path: str):
    """Load thickness CSV."""

    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
    )

    if data.ndim == 1:
        data = data[None, :]

    return (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
    )


def save_figure(fig, path: str, dpi: int = 150):
    """Save matplotlib figure."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def export_step(shapes: list, path: str):
    """Export OCP shapes as STEP."""

    from OCP.STEPControl import (
        STEPControl_Writer,
        STEPControl_AsIs,
    )
    from OCP.IFSelect import IFSelect_RetDone

    writer = STEPControl_Writer()

    for shape in shapes:
        writer.Transfer(shape, STEPControl_AsIs)

    if writer.Write(path) != IFSelect_RetDone:
        raise RuntimeError(f"STEP export failed: {path}")


def load_cad_mesh(path: str):
    """
    Load CAD as a triangulated trimesh.

    The STEP file declares millimeter units, but the current
    trimesh/cascadio import path returns the coordinates in meters.

    Therefore STEP geometry is converted back to mm here.

    This means every downstream numerical module operates in mm.
    """

    import trimesh

    mesh = trimesh.load(path, force="mesh")

    extension = os.path.splitext(path)[1].lower()

    if extension in {".step", ".stp"}:
        mesh.apply_scale(1000.0)

    return mesh


def load_cad_brep(path: str):
    """
    Load STEP as an exact OpenCascade B-Rep.
    """

    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()

    if reader.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError(
            f"Failed to read STEP file: {path}"
        )

    reader.TransferRoots()

    return reader.OneShape()