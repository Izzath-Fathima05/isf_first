"""
CAD -> SPIF-Exposed Profile -> Patches -> Flattened Layout
=============================================================

Handles the AXISYMMETRIC case (bodies of revolution). For general,
non-axisymmetric B-Reps, use general_flatten.py + cad_native_patches.py
instead.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_cad_as_mesh(path: str) -> trimesh.Trimesh:
    """Loads any CAD file trimesh understands (STEP via cascadio, STL, etc.)."""
    return trimesh.load(path, force="mesh")


def check_axisymmetry(mesh: trimesh.Trimesh, axis=np.array([0, 0, 1.0]),
                       axis_origin=np.array([0, 0, 0])) -> dict:
    """
    Confirms (or flags violation of) the axisymmetry assumption that
    makes the rest of this pipeline valid.
    """
    axis = axis / np.linalg.norm(axis)
    rel = mesh.vertices - axis_origin
    u = rel @ axis
    perp = rel - np.outer(u, axis)
    radius = np.linalg.norm(perp, axis=1)

    n_bins = 20
    bins = np.linspace(u.min(), u.max(), n_bins + 1)
    bin_idx = np.digitize(u, bins) - 1
    spreads = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 1:
            spreads.append(radius[mask].std())
    max_spread = max(spreads) if spreads else 0.0

    return {
        "u": u, "radius": radius,
        "max_radial_spread": max_spread,
        "likely_axisymmetric": max_spread < 0.05 * radius.max(),
    }


def extract_axisymmetric_profile(mesh: trimesh.Trimesh, n_profile_points: int = 60,
                                  axis=np.array([0, 0, 1.0]),
                                  axis_origin=np.array([0, 0, 0])):
    """
    Collapses the 3D mesh surface onto its 2D axisymmetric profile
    (r(z)) by binning vertices by height along the axis and taking the
    mean radius per bin -- robust to angular (azimuthal) mesh noise.
    """
    axis = axis / np.linalg.norm(axis)
    rel = mesh.vertices - axis_origin
    u = rel @ axis
    perp = rel - np.outer(u, axis)
    radius = np.linalg.norm(perp, axis=1)

    bins = np.linspace(u.min(), u.max(), n_profile_points + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    r_profile = np.array([
        radius[(u >= bins[i]) & (u < bins[i + 1])].mean()
        if np.any((u >= bins[i]) & (u < bins[i + 1])) else np.nan
        for i in range(n_profile_points)
    ])

    valid = ~np.isnan(r_profile)
    return bin_centers[valid], r_profile[valid]


def segment_into_patches(z_profile: np.ndarray, r_profile: np.ndarray, n_patches: int = 5):
    """
    Divides the profile into n_patches contiguous segments of roughly
    EQUAL ARC LENGTH along the profile curve.
    """
    seg_lengths = np.sqrt(np.diff(r_profile) ** 2 + np.diff(z_profile) ** 2)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    patch_boundaries = np.linspace(0, total_length, n_patches + 1)
    patches = []
    for i in range(n_patches):
        mask = (cum_length >= patch_boundaries[i]) & (cum_length <= patch_boundaries[i + 1])
        idx = np.where(mask)[0]
        if len(idx) < 2:
            idx = np.searchsorted(cum_length, [patch_boundaries[i], patch_boundaries[i + 1]])
        patches.append({
            "z": z_profile[idx], "r": r_profile[idx],
            "arc_length": patch_boundaries[i + 1] - patch_boundaries[i],
            "label": chr(ord("A") + i),
        })
    return patches


def flatten_patches_to_strip(patches: list, thickness: np.ndarray = None):
    """
    Lays each patch flat, side by side -- width = patch arc length,
    height = patch thickness (uniform placeholder if not yet computed
    by backward_solve.py).
    """
    if thickness is None:
        thickness = np.ones(len(patches))

    x_start = 0.0
    layout = []
    for patch, t in zip(patches, thickness):
        layout.append({"label": patch["label"], "x_start": x_start,
                        "width": patch["arc_length"], "height": t})
        x_start += patch["arc_length"]
    return layout


def plot_profile_and_patches(z_profile, r_profile, patches, layout, save_path):
    """Two-panel plot: profile with colored patch overlay on top,
    flattened strip layout below."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(patches)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

    ax1.plot(r_profile, z_profile, color="black", linewidth=1)
    for patch, c in zip(patches, colors):
        ax1.plot(patch["r"], patch["z"], color=c, linewidth=6, solid_capstyle="round")
        mid = len(patch["r"]) // 2
        ax1.annotate(patch["label"], (patch["r"][mid], patch["z"][mid]),
                     textcoords="offset points", xytext=(-15, 0), fontsize=12, fontweight="bold")
    ax1.set_xlabel("r (mm)"); ax1.set_ylabel("z (mm)")
    ax1.set_title("SPIF-exposed profile, divided into patches")
    ax1.set_aspect("equal")

    for item, c in zip(layout, colors):
        rect = mpatches.Rectangle((item["x_start"], 0), item["width"], item["height"],
                                   facecolor=c, edgecolor="black")
        ax2.add_patch(rect)
        ax2.annotate(item["label"], (item["x_start"] + item["width"] / 2, -0.15),
                     ha="center", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, layout[-1]["x_start"] + layout[-1]["width"])
    ax2.set_ylim(-0.3, max(item["height"] for item in layout) * 1.3)
    ax2.set_xlabel("flattened arc-length position (mm)")
    ax2.set_title("Patches flattened and laid side by side")
    ax2.set_aspect("auto")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    return fig