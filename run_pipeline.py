"""
End-to-end ISF thickness pipeline.

CAD STEP
    ↓
unit-normalized mesh [mm]
    ↓
outer axisymmetric surface
    ↓
2D r(z) profile
    ↓
tangent angle theta
    ↓
sine-law angle alpha
    ↓
actual 2-degree alpha bins
    ↓
initial blank thickness
    ↓
CSV + plots + UV map
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Project path
# -------------------------------------------------------------

SRC_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "src",
)

sys.path.insert(
    0,
    SRC_DIR,
)


# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------

from utils.io_utils import (
    ensure_data_dirs,
    load_cad_mesh,
    save_thickness_csv,
)

from numerical.patch_extraction import (
    check_axisymmetry,
    extract_axisymmetric_profile,
    segment_into_patches,
    flatten_patches_to_strip,
    plot_profile_and_patches,
)

from numerical.alpha_bin_solver import (
    compute_tangent_angles,
    tangent_to_sine_law_angle,
    solve_thickness,
)

from numerical.uv_contour import (
    generate_uv_contour_map,
)
from utils.visualization import (
    plot_thickness_segment_heatmap,
)


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="ISF alpha-bin thickness pipeline"
)

parser.add_argument(
    "--input",
    required=True,
)

parser.add_argument(
    "--t_target",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--n_patches",
    type=int,
    default=5,
)

parser.add_argument(
    "--angle_step",
    type=float,
    default=2.0,
)

parser.add_argument(
    "--surface",
    choices=["outer", "inner"],
    default="outer",
)

parser.add_argument(
    "--n_profile_points",
    type=int,
    default=361,
)

args = parser.parse_args()


# -------------------------------------------------------------
# Directories
# -------------------------------------------------------------

ensure_data_dirs()

os.makedirs(
    "data/output/patches",
    exist_ok=True,
)

os.makedirs(
    "data/output/thickness_maps",
    exist_ok=True,
)

os.makedirs(
    "data/output/visualizations",
    exist_ok=True,
)


# -------------------------------------------------------------
# 1. Load CAD
# -------------------------------------------------------------

print(
    f"Loading CAD file: {args.input}"
)

mesh = load_cad_mesh(
    args.input
)

print(
    f"  {len(mesh.vertices)} vertices, "
    f"{len(mesh.faces)} faces"
)

print(
    "  Mesh extent [mm]:",
    mesh.extents
)


# -------------------------------------------------------------
# 2. Axisymmetry
# -------------------------------------------------------------

symmetry = check_axisymmetry(
    mesh,
    surface=args.surface,
)

print(
    "Axisymmetry check:"
)

print(
    f"  selected surface : {args.surface}"
)

print(
    f"  max radial spread : "
    f"{symmetry['max_radial_spread']:.6f} mm"
)

print(
    f"  relative spread   : "
    f"{100 * symmetry['relative_spread']:.4f}%"
)

print(
    f"  likely axisymmetric: "
    f"{symmetry['likely_axisymmetric']}"
)

if not symmetry["likely_axisymmetric"]:

    print(
        "  WARNING: selected surface "
        "still shows significant non-axisymmetry."
    )

    print(
        "  Results should be treated as approximate."
    )


# -------------------------------------------------------------
# 3. Extract surface profile
# -------------------------------------------------------------

print()
print(
    "Extracting axisymmetric profile..."
)

z_profile, r_profile = (
    extract_axisymmetric_profile(
        mesh,
        n_profile_points=args.n_profile_points,
        surface=args.surface,
    )
)

print(
    "Profile range:"
)

print(
    f"  r: {r_profile.min():.6f} "
    f"to {r_profile.max():.6f} mm"
)

print(
    f"  z: {z_profile.min():.6f} "
    f"to {z_profile.max():.6f} mm"
)

print(
    f"  Profile points: "
    f"{len(r_profile)}"
)


# -------------------------------------------------------------
# 4. Compute theta / alpha diagnostics
# -------------------------------------------------------------

theta_profile = compute_tangent_angles(
    r_profile,
    z_profile,
)

alpha_profile = tangent_to_sine_law_angle(
    theta_profile
)

print()
print(
    "Angle range:"
)

print(
    f"  theta: "
    f"{theta_profile.min():.3f}° "
    f"to {theta_profile.max():.3f}°"
)

print(
    f"  alpha: "
    f"{alpha_profile.min():.3f}° "
    f"to {alpha_profile.max():.3f}°"
)


# -------------------------------------------------------------
# 5. Create geometric patches
# -------------------------------------------------------------

print()
print(
    "Creating geometric patches..."
)

patches = segment_into_patches(
    z_profile,
    r_profile,
    n_patches=args.n_patches,
)

layout = flatten_patches_to_strip(
    patches
)

plot_profile_and_patches(
    z_profile,
    r_profile,
    patches,
    layout,
    "data/output/patches/profile_patches.png",
)


# -------------------------------------------------------------
# 6. Alpha-bin solver
# -------------------------------------------------------------

print()
print(
    "Running alpha-bin thickness solver..."
)

print(
    f"  Target final thickness : "
    f"{args.t_target:.4f} mm"
)

print(
    f"  Alpha-bin width        : "
    f"{args.angle_step:.2f} degrees"
)

elements, rows = solve_thickness(
    r_profile=r_profile,
    z_profile=z_profile,
    t_final=args.t_target,
    angle_step_deg=args.angle_step,
)


# -------------------------------------------------------------
# 7. Print physical elements
# -------------------------------------------------------------

print()
print(
    "Physical alpha-bin solution"
)

print(
    "---------------------------"
)

print(
    f"Total physical elements: "
    f"{len(elements)}"
)

print()

print(
    "Element   Alpha range      "
    "Theta      Alpha       T0"
)

print(
    "------------------------------------------------"
)

for element in elements:

    print(
        f"{element['index']:3d}       "
        f"{element['alpha_start']:5.1f}-"
        f"{element['alpha_end']:5.1f}°    "
        f"{element['theta_mid']:7.2f}°   "
        f"{element['alpha_mid']:7.2f}°   "
        f"{element['t_initial']:10.4f} mm"
    )


# -------------------------------------------------------------
# 8. Convert solver rows
# -------------------------------------------------------------

rows = np.asarray(
    rows,
    dtype=float,
)

if rows.ndim != 2 or rows.shape[1] < 5:

    raise ValueError(
        "Unexpected solver output format. "
        f"Expected rows with at least 5 columns, "
        f"got shape {rows.shape}."
    )

r_output = rows[:, 0]
z_output = rows[:, 1]
t_output = rows[:, 2]
theta_output = rows[:, 3]
alpha_output = rows[:, 4]


# -------------------------------------------------------------
# 9. Thickness plot
# -------------------------------------------------------------

print()
print(
    "Generating thickness distribution plot..."
)

order = np.argsort(
    r_output
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.plot(
    r_output[order],
    t_output[order],
    "o-",
    linewidth=1.5,
    markersize=4,
)

ax.set_xlabel(
    "Radius r (mm)"
)

ax.set_ylabel(
    "Initial thickness T₀ (mm)"
)

ax.set_title(
    "Predicted initial blank thickness"
)

ax.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    "data/output/thickness_maps/"
    "thickness_vs_radius.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)

# -------------------------------------------------------------
# 9B. Thickness heatmap across alpha segments
# -------------------------------------------------------------

print()
print(
    "Generating thickness segment heatmap..."
)

alpha_start = np.array(
    [
        element["alpha_start"]
        for element in elements
    ],
    dtype=float,
)

alpha_end = np.array(
    [
        element["alpha_end"]
        for element in elements
    ],
    dtype=float,
)

t_initial_segments = np.array(
    [
        element["t_initial"]
        for element in elements
    ],
    dtype=float,
)

fig_heatmap = plot_thickness_segment_heatmap(
    alpha_start=alpha_start,
    alpha_end=alpha_end,
    t_initial=t_initial_segments,
    title="Initial Thickness Across Alpha Segments",
)

heatmap_output_path = (
    "data/output/thickness_maps/"
    "thickness_segment_heatmap.png"
)

fig_heatmap.savefig(
    heatmap_output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig_heatmap)

print(
    f"  Thickness segment heatmap saved: "
    f"{heatmap_output_path}"
)

# -------------------------------------------------------------
# 10. Save CSV
# -------------------------------------------------------------

print(
    "Saving thickness table..."
)

save_thickness_csv(
    "data/output/thickness_maps/"
    "thickness_table.csv",
    r_output,
    z_output,
    theta_output,
    t_output,
)


# -------------------------------------------------------------
# 11. UV thickness map
# -------------------------------------------------------------

print()
print(
    "Generating UV contour map..."
)

try:

    # ---------------------------------------------------------
    # Use ACTUAL CAD vertices.
    #
    # Do not use r_profile / z_profile here.
    #
    # The profile has 361 samples.
    # The solver has 42 alpha-bin samples.
    # The UV map needs the original 3D CAD vertices.
    # ---------------------------------------------------------

    surface_vertices = np.asarray(
        mesh.vertices,
        dtype=float,
    )

    print(
        f"  CAD UV vertices      : "
        f"{len(surface_vertices)}"
    )


    # ---------------------------------------------------------
    # Extract alpha-bin centers
    # ---------------------------------------------------------

    alpha_centers = np.array(
        [
            element["alpha_mid"]
            for element in elements
        ],
        dtype=float,
    )

    print(
        f"  Alpha-bin samples    : "
        f"{len(alpha_centers)}"
    )


    # ---------------------------------------------------------
    # Extract initial thickness values
    # ---------------------------------------------------------

    thickness_values = np.array(
        [
            element["t_initial"]
            for element in elements
        ],
        dtype=float,
    )

    print(
        f"  Thickness samples    : "
        f"{len(thickness_values)}"
    )


    # ---------------------------------------------------------
    # Estimate spherical forming surface
    # ---------------------------------------------------------
    #
    # Sphere equation:
    #
    #     r² + (z - zc)² = R²
    #
    # Expanding:
    #
    #     r² + z² - 2 z zc + zc² = R²
    #
    # Therefore:
    #
    #     r² + z² = 2 z zc + C
    #
    # where:
    #
    #     C = R² - zc²
    #
    # IMPORTANT:
    #
    # The coefficient MUST be +2*z.
    #
    # The previous version used -2*z, which produced:
    #
    #     zc = +0.469850 mm
    #
    # instead of:
    #
    #     zc = -0.469850 mm
    #
    # That sign error caused the UV alpha range to become
    # approximately 90°–180° and all thickness values to
    # clamp to the final alpha-bin value.
    # ---------------------------------------------------------

    profile_r = np.asarray(
        r_profile,
        dtype=float,
    )

    profile_z = np.asarray(
        z_profile,
        dtype=float,
    )

    valid_profile = (
        np.isfinite(profile_r)
        & np.isfinite(profile_z)
    )

    rp = profile_r[
        valid_profile
    ]

    zp = profile_z[
        valid_profile
    ]

    if len(rp) < 3:

        raise ValueError(
            "Not enough valid profile points "
            "to fit spherical surface."
        )


    # ---------------------------------------------------------
    # Correct sphere fit
    # ---------------------------------------------------------

    A = np.column_stack(
        (
            2.0 * zp,
            np.ones_like(zp),
        )
    )

    b = (
        rp ** 2
        + zp ** 2
    )

    fit, _, _, _ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    sphere_center_z = float(
        fit[0]
    )

    sphere_constant = float(
        fit[1]
    )


    # ---------------------------------------------------------
    # Recover sphere radius
    #
    # C = R² - zc²
    #
    # therefore:
    #
    # R² = C + zc²
    # ---------------------------------------------------------

    sphere_radius_squared = (
        sphere_constant
        + sphere_center_z ** 2
    )

    if sphere_radius_squared <= 0:

        raise ValueError(
            "Invalid fitted sphere radius."
        )

    sphere_radius = float(
        np.sqrt(
            sphere_radius_squared
        )
    )


    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    print(
        f"  Sphere radius        : "
        f"{sphere_radius:.6f} mm"
    )

    print(
        f"  Sphere center z      : "
        f"{sphere_center_z:.6f} mm"
    )


    # ---------------------------------------------------------
    # Expected values for current bowl
    # ---------------------------------------------------------

    print(
        "  Expected approximately:"
    )

    print(
        "    R  ≈ 47.586930 mm"
    )

    print(
        "    zc ≈ -0.469850 mm"
    )


    # ---------------------------------------------------------
    # Sphere center
    # ---------------------------------------------------------

    sphere_center = np.array(
        [
            0.0,
            0.0,
            sphere_center_z,
        ],
        dtype=float,
    )


    # ---------------------------------------------------------
    # Generate UV map
    # ---------------------------------------------------------

    (
        fig,
        u,
        v,
        thickness_surface,
        alpha_surface,
    ) = generate_uv_contour_map(
        vertices=surface_vertices,
        alpha_centers=alpha_centers,
        thickness_values=thickness_values,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        axis=np.array(
            [0.0, 0.0, 1.0]
        ),
        axis_origin=np.array(
            [0.0, 0.0, 0.0]
        ),
        grid_res=200,
        title="Initial Sheet Thickness T₀(u,v)",
    )


    # ---------------------------------------------------------
    # UV diagnostics
    # ---------------------------------------------------------

    print(
        f"  CAD surface vertices : "
        f"{len(surface_vertices)}"
    )

    print(
        f"  UV samples           : "
        f"{len(u)}"
    )

    print(
        f"  Solver alpha bins    : "
        f"{len(alpha_centers)}"
    )

    print(
        f"  Surface thickness    : "
        f"{len(thickness_surface)}"
    )

    print(
        f"  Alpha range          : "
        f"{alpha_surface.min():.3f}° "
        f"to "
        f"{alpha_surface.max():.3f}°"
    )

    print(
        f"  Thickness range      : "
        f"{thickness_surface.min():.4f} "
        f"to "
        f"{thickness_surface.max():.4f} mm"
    )


    # ---------------------------------------------------------
    # Sanity check
    # ---------------------------------------------------------

    if (
        alpha_surface.min() > 90.0
        or alpha_surface.max() > 100.0
    ):

        print()
        print(
            "  WARNING: UV alpha range looks incorrect."
        )

        print(
            "  Check the alpha convention in "
            "src/numerical/uv_contour.py."
        )


    # ---------------------------------------------------------
    # Save UV map
    # ---------------------------------------------------------

    uv_output_path = (
        "data/output/visualizations/"
        "uv_contour_map.png"
    )

    fig.savefig(
        uv_output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"  UV contour map saved: "
        f"{uv_output_path}"
    )


except Exception as exc:

    print(
        "  UV map generation skipped:"
    )

    print(
        f"  {exc}"
    )


# -------------------------------------------------------------
# 12. Summary
# -------------------------------------------------------------

print()
print(
    "============================================================"
)

print(
    "PIPELINE COMPLETE"
)

print(
    "============================================================"
)

print(
    f"Physical elements : "
    f"{len(elements)}"
)

print(
    f"Output samples    : "
    f"{len(rows)}"
)

print(
    f"Target thickness  : "
    f"{args.t_target:.4f} mm"
)

print()

print(
    "Outputs:"
)

print(
    "  - data/output/patches/"
    "profile_patches.png"
)

print(
    "  - data/output/thickness_maps/"
    "thickness_vs_radius.png"
)

print(
    "  - data/output/thickness_maps/"
    "thickness_table.csv"
)

print(
    "  - data/output/visualizations/"
    "uv_contour_map.png"
)