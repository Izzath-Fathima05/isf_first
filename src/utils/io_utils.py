"""
File I/O and project directory management -- matches the
ISF_Thickness_Map/data/ folder structure exactly, so every module
saves outputs to a predictable place instead of scattering files.
"""

import os
import csv
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIRS = {
    "input": os.path.join(PROJECT_ROOT, "data", "input"),
    "patches": os.path.join(PROJECT_ROOT, "data", "intermediate", "patches"),
    "contours": os.path.join(PROJECT_ROOT, "data", "intermediate", "contours"),
    "flattened": os.path.join(PROJECT_ROOT, "data", "output", "flattened_patches"),
    "thickness_maps": os.path.join(PROJECT_ROOT, "data", "output", "thickness_maps"),
    "visualizations": os.path.join(PROJECT_ROOT, "data", "output", "visualizations"),
}


def ensure_data_dirs():
    """Creates the full data/ folder tree if it doesn't already exist.
    Call this once at the start of run_pipeline.py."""
    for path in DATA_DIRS.values():
        os.makedirs(path, exist_ok=True)


def save_thickness_csv(path: str, r: np.ndarray, z: np.ndarray,
                        theta_deg: np.ndarray, t0: np.ndarray):
    """Saves the (r, z, theta, T0) data table -- the actual 'UV
    thickness map' data, in tabular form for CAM/downstream use."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_mm", "z_mm", "theta_deg", "T0_mm"])
        for ri, zi, ti, t0i in zip(r, z, theta_deg, t0):
            writer.writerow([f"{ri:.4f}", f"{zi:.4f}", f"{ti:.3f}", f"{t0i:.4f}"])


def load_thickness_csv(path: str):
    """Reads back a thickness CSV into numpy arrays."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]  # r, z, theta, T0


def save_figure(fig, path: str, dpi: int = 150):
    fig.savefig(path, dpi=dpi)


def export_step(shapes: list, path: str):
    """Exports OCP shapes as a STEP file -- shared by cad_native_patches.py
    and general_flatten.py to avoid duplicating this."""
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone

    writer = STEPControl_Writer()
    for s in shapes:
        writer.Transfer(s, STEPControl_AsIs)
    if writer.Write(path) != IFSelect_RetDone:
        raise RuntimeError(f"STEP export failed: {path}")


def load_cad_mesh(path: str):
    """Loads a CAD file as a triangulated mesh (trimesh + cascadio for
    STEP support) -- used by the numerical pipeline."""
    import trimesh
    return trimesh.load(path, force="mesh")


def load_cad_brep(path: str):
    """Loads a CAD file as a real B-Rep (OCP) -- used by the CAD-native
    pipeline, where exact surface types matter."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path}")
    reader.TransferRoots()
    return reader.OneShape()