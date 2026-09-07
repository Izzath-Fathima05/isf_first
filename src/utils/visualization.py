"""
Centralized plotting for the ISF thickness-map pipeline. All figures
in the project should go through here, not be built ad hoc in
individual modules -- keeps plot styling consistent and outputs
easy to find.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import griddata


def plot_uv_contour(u, v, T0, grid_res=200,
                     title="Initial Sheet Thickness T\u2080(u,v)"):
    """True 2D UV contour map (see uv_contour.py for the math)."""
    u_grid = np.linspace(u.min(), u.max(), grid_res)
    v_grid = np.linspace(v.min(), v.max(), grid_res)
    UU, VV = np.meshgrid(u_grid, v_grid)
    TT = griddata((u, v), T0, (UU, VV), method="linear")

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(np.degrees(VV), UU, TT, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="T\u2080 (mm)")
    ax.set_xlabel("v \u2014 angle around axis (deg)")
    ax.set_ylabel("u \u2014 height along axis (mm)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_profile_and_patches(z_profile, r_profile, patches, layout, title=None):
    """Two-panel: profile with colored patches, and flattened strip layout."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(patches)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))

    ax1.plot(r_profile, z_profile, color="black", linewidth=1)
    for patch, c in zip(patches, colors):
        ax1.plot(patch["r"], patch["z"], color=c, linewidth=6, solid_capstyle="round")
        mid = len(patch["r"]) // 2
        ax1.annotate(patch["label"], (patch["r"][mid], patch["z"][mid]),
                     textcoords="offset points", xytext=(-15, 0), fontsize=12, fontweight="bold")
    ax1.set_xlabel("r (mm)"); ax1.set_ylabel("z (mm)")
    ax1.set_title(title or "SPIF-exposed profile, divided into patches")
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
    plt.tight_layout()
    return fig


def plot_thickness_vs_radius(r_flat, t_initial, t_target, title="Required Initial Thickness"):
    """Matches the validation-style plot used against Wu et al.'s benchmark cone."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r_flat, t_initial, 'o-', color="#0F766E", markersize=4, label="Computed t\u2080")
    ax.axhline(t_target, color="#C2703D", linestyle="--", label=f"Target t_df = {t_target} mm")
    ax.set_xlabel("r (flat blank radius, mm)")
    ax.set_ylabel("t\u2080 (mm)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_mesh_3d(mesh, color_values=None, title="Mesh", cmap="viridis"):
    """Generic 3D mesh viewer, optionally colored by a per-face scalar
    (e.g. thickness) -- consolidates the ad hoc mesh-rendering code
    used throughout early exploration."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    if color_values is not None:
        norm = plt.Normalize(color_values.min(), color_values.max())
        colors = plt.get_cmap(cmap)(norm(color_values))
        poly = Poly3DCollection(mesh.vertices[mesh.faces], facecolors=colors, edgecolor="none")
    else:
        poly = Poly3DCollection(mesh.vertices[mesh.faces], facecolor="lightsteelblue",
                                 edgecolor="k", linewidth=0.1)
    ax.add_collection3d(poly)
    ax.set_xlim(mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max())
    ax.set_ylim(mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())
    ax.set_zlim(mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max())
    ax.set_title(title)
    plt.tight_layout()
    return fig