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
physical regions / 2-degree alpha bins
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


# =============================================================
# Project path
# =============================================================

SRC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src",
)

sys.path.insert(0, SRC_DIR)


# =============================================================
# Imports
# =============================================================

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# =============================================================
# Project path
# =============================================================

SRC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src",
)

sys.path.insert(
    0,
    SRC_DIR,
)


# =============================================================
# Project imports
# =============================================================

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

from utils.wall_angle_plot import (
    plot_wall_angle_vs_radius,
)

# =============================================================
# CLI
# =============================================================

parser = argparse.ArgumentParser(
    description="ISF alpha-bin thickness pipeline"
)

parser.add_argument(
    "--input",
    required=True,
    help="Input STEP file",
)

parser.add_argument(
    "--t_target",
    type=float,
    default=1.0,
    help="Target final wall thickness [mm]",
)

parser.add_argument(
    "--n_patches",
    type=int,
    default=5,
    help="Number of geometric patches",
)

parser.add_argument(
    "--angle_step",
    type=float,
    default=2.0,
    help="Alpha-bin width [degrees]",
)

parser.add_argument(
    "--surface",
    choices=["outer", "inner"],
    default="outer",
    help="Surface used for axisymmetric profile extraction",
)

parser.add_argument(
    "--n_profile_points",
    type=int,
    default=361,
    help="Number of profile samples",
)

args = parser.parse_args()

# =============================================================
# Directories
# =============================================================

ensure_data_dirs()

output_patches_dir = "data/output/patches"
output_thickness_maps_dir = "data/output/thickness_maps"
output_visualizations_dir = "data/output/visualizations"

os.makedirs(
    output_patches_dir,
    exist_ok=True,
)

os.makedirs(
    output_thickness_maps_dir,
    exist_ok=True,
)

os.makedirs(
    output_visualizations_dir,
    exist_ok=True,
)


# =============================================================
# 1. Load CAD
# =============================================================

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
    mesh.extents,
)


# =============================================================
# 2. Axisymmetry
# =============================================================

symmetry = check_axisymmetry(
    mesh,
    surface=args.surface,
)

print()
print("Axisymmetry check:")

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


# =============================================================
# 3. Extract axisymmetric profile
# =============================================================

print()
print(
    "Extracting axisymmetric profile..."
)

z_profile, r_profile = extract_axisymmetric_profile(
    mesh,
    n_profile_points=args.n_profile_points,
    surface=args.surface,
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


# =============================================================
# 4. Theta / alpha diagnostics
# =============================================================

theta_profile = compute_tangent_angles(
    r_profile,
    z_profile,
)

alpha_profile = tangent_to_sine_law_angle(
    theta_profile,
)

# -------------------------------------------------------------
# Diagnostic: wall angle versus radius
# -------------------------------------------------------------

print()
print(
    "Generating wall-angle diagnostic plot..."
)

wall_angle_path = os.path.join(
    output_visualizations_dir,
    "wall_angle_vs_radius.png",
)

wall_angle_fig = plot_wall_angle_vs_radius(
    r_profile,
    alpha_profile,
    save_path=wall_angle_path,
)

plt.close(
    wall_angle_fig
)

print(
    f"  Wall-angle diagnostic saved: "
    f"{wall_angle_path}"
)


print()
print("Angle range:")

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


# =============================================================
# 5. Create geometric patches
# =============================================================

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


# =============================================================
# 6. Alpha-bin thickness solver
# =============================================================

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

solver_result = solve_thickness(
    r_profile=r_profile,
    z_profile=z_profile,
    t_final=args.t_target,
    angle_step_deg=args.angle_step,
)


# =============================================================
# Validate solver result
# =============================================================

if not isinstance(solver_result, dict):

    raise TypeError(
        "solve_thickness() must return a dictionary. "
        f"Got {type(solver_result)}."
    )

if "elements" not in solver_result:

    raise KeyError(
        "solve_thickness() result is missing "
        "'elements'. "
        f"Available keys: {list(solver_result.keys())}"
    )

if "rows" not in solver_result:

    raise KeyError(
        "solve_thickness() result is missing "
        "'rows'. "
        f"Available keys: {list(solver_result.keys())}"
    )


raw_elements = solver_result["elements"]
raw_rows = solver_result["rows"]

# =============================================================
# Normalize current alpha_bin_solver.py elements
# =============================================================
#
# Current solver element schema:
#
#   kind
#   start_idx
#   end_idx
#   r_start
#   z_start
#   r_end
#   z_end
#   dr
#   dz
#   arc_length
#   theta_mid
#   alpha_mid
#   t_initial
#   t_analytic
#   analytic_error_percent
#   t_final
#   target_volume
#   alpha_bin_index
#   alpha_bin_start
#   alpha_bin_end
#   region_arc_length
#   characteristic_spacing
#   min_arc_length
#   arc_length_ratio
#   label
#
# Do NOT derive alpha_mid/theta_mid here.
# They are already supplied by the current solver.
# =============================================================

elements = []

for index, element in enumerate(
    raw_elements,
    start=1,
):

    if not isinstance(element, dict):
        raise TypeError(
            f"Solver element {index} must be a dictionary. "
            f"Got {type(element)}."
        )

    required_keys = [
        "kind",
        "r_start",
        "z_start",
        "r_end",
        "z_end",
        "dr",
        "dz",
        "arc_length",
        "theta_mid",
        "alpha_mid",
        "t_initial",
        "t_analytic",
        "analytic_error_percent",
        "t_final",
        "target_volume",
        "alpha_bin_index",
        "alpha_bin_start",
        "alpha_bin_end",
        "region_arc_length",
        "characteristic_spacing",
        "min_arc_length",
        "arc_length_ratio",
        "label",
    ]

    missing = [
        key
        for key in required_keys
        if key not in element
    ]

    if missing:
        raise KeyError(
            f"Solver element {index} is missing required keys: "
            f"{missing}. "
            f"Available keys: {list(element.keys())}"
        )

    elements.append(
        {
            "index": index,

            "kind": element["kind"],
            "label": element["label"],

            "start_idx": int(
                element["start_idx"]
            ),

            "end_idx": int(
                element["end_idx"]
            ),

            "r_start": float(
                element["r_start"]
            ),

            "z_start": float(
                element["z_start"]
            ),

            "r_end": float(
                element["r_end"]
            ),

            "z_end": float(
                element["z_end"]
            ),

            "dr": float(
                element["dr"]
            ),

            "dz": float(
                element["dz"]
            ),

            "arc_length": float(
                element["arc_length"]
            ),

            "theta_mid": float(
                element["theta_mid"]
            ),

            "alpha_mid": float(
                element["alpha_mid"]
            ),

            "t_initial": float(
                element["t_initial"]
            ),

            "t_analytic": float(
                element["t_analytic"]
            ),

            "analytic_error_percent": float(
                element["analytic_error_percent"]
            ),

            "t_final": float(
                element["t_final"]
            ),

            "target_volume": float(
                element["target_volume"]
            ),

            "alpha_bin_index": element[
                "alpha_bin_index"
            ],

            "alpha_bin_start": float(
                element["alpha_bin_start"]
            ),

            "alpha_bin_end": float(
                element["alpha_bin_end"]
            ),

            "region_arc_length": float(
                element["region_arc_length"]
            ),

            "characteristic_spacing": float(
                element["characteristic_spacing"]
            ),

            "min_arc_length": float(
                element["min_arc_length"]
            ),

            "arc_length_ratio": float(
                element["arc_length_ratio"]
            ),
        }
    )


# =============================================================
# Convert current solver rows
# =============================================================
#
# The current solver already supplies alpha/theta information.
# Prefer the actual solver values instead of reconstructing them.
# =============================================================

numeric_rows = []

for index, row in enumerate(
    raw_rows,
    start=1,
):

    if not isinstance(row, dict):
        raise TypeError(
            f"Solver row {index} must be a dictionary. "
            f"Got {type(row)}."
        )

    required_keys = [
        "r",
        "z",
        "t_initial",
    ]

    missing = [
        key
        for key in required_keys
        if key not in row
    ]

    if missing:
        raise KeyError(
            f"Solver row {index} is missing required keys: "
            f"{missing}. "
            f"Available keys: {list(row.keys())}"
        )

    r_value = float(row["r"])
    z_value = float(row["z"])
    t_value = float(row["t_initial"])

    # Current solver may already provide these.
    # If not, derive them from the alpha-bin boundaries.
    if "alpha_mid" in row:
        alpha_value = float(
            row["alpha_mid"]
        )

    elif (
        "alpha_bin_start" in row
        and "alpha_bin_end" in row
    ):
        alpha_value = 0.5 * (
            float(row["alpha_bin_start"])
            + float(row["alpha_bin_end"])
        )

    else:
        raise KeyError(
            f"Solver row {index} has no alpha information. "
            f"Available keys: {list(row.keys())}"
        )

    if "theta_mid" in row:
        theta_value = float(
            row["theta_mid"]
        )

    else:
        # Current alpha convention:
        # alpha = 90 - theta_acute
        theta_value = 90.0 - alpha_value

    numeric_rows.append(
        [
            r_value,
            z_value,
            t_value,
            theta_value,
            alpha_value,
        ]
    )

if not numeric_rows:
    raise ValueError(
        "solve_thickness() returned no rows."
    )

rows = np.asarray(
    numeric_rows,
    dtype=float,
)

# =============================================================
# Numerical validation
# =============================================================

if not np.all(
    np.isfinite(rows)
):

    raise ValueError(
        "Solver rows contain NaN or infinite values."
    )

if np.any(
    rows[:, 2] <= 0.0
):

    bad_indices = np.where(
        rows[:, 2] <= 0.0
    )[0]

    raise ValueError(
        "Solver produced non-positive "
        "initial thickness at row indices: "
        f"{bad_indices.tolist()}"
    )


# =============================================================
# 7. Print physical elements
# =============================================================

print()
print("Physical alpha-bin solution")
print("---------------------------")

print(
    f"Total physical elements: {len(elements)}"
)

print()

print(
    "Element   Alpha range       "
    "Theta      T0(mm)      Arc(mm)"
)

print(
    "----------------------------------------------------------"
)

for element in elements:

    print(
        f"{element['index']:3d}       "
        f"{element['alpha_bin_start']:5.1f}-"
        f"{element['alpha_bin_end']:5.1f}°    "
        f"{element['theta_mid']:7.2f}°   "
        f"{element['t_initial']:9.4f}   "
        f"{element['arc_length']:9.4f}"
    )


# =============================================================
# 8. Output arrays
# =============================================================

print()
print(
    "Processing solver rows..."
)

r_output = rows[:, 0]
z_output = rows[:, 1]
t_output = rows[:, 2]
theta_output = rows[:, 3]
alpha_output = rows[:, 4]


# =============================================================
# 9. Thickness vs radius plot
# =============================================================

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


# =============================================================
# 9B. Thickness heatmap
# =============================================================

print()
print(
    "Generating thickness segment heatmap..."
)

alpha_start = np.array(
    [
        element["alpha_bin_start"]
        for element in elements
    ],
    dtype=float,
)

alpha_end = np.array(
    [
        element["alpha_bin_end"]
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

heatmap_output_path = os.path.join(
    output_thickness_maps_dir,
    "thickness_segment_heatmap.png",
)

fig_heatmap.savefig(
    heatmap_output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    fig_heatmap
)

print(
    f"  Thickness segment heatmap saved: "
    f"{heatmap_output_path}"
)


# =============================================================
# 10. Save CSV
# =============================================================

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


# =============================================================
# 11. UV thickness map
# =============================================================

print()
print(
    "Generating UV contour map..."
)

try:

    # ---------------------------------------------------------
    # Actual CAD vertices
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
    # Alpha-bin centers
    # ---------------------------------------------------------

    alpha_centers = np.array(
    [
        0.5 * (
            element["alpha_bin_start"]
            + element["alpha_bin_end"]
        )
        for element in elements
    ],
    dtype=float,
)

    thickness_values = np.array(
        [
            element["t_initial"]
            for element in elements
        ],
        dtype=float,
    )

    print(
        f"  Alpha-bin samples    : "
        f"{len(alpha_centers)}"
    )

    print(
        f"  Thickness samples    : "
        f"{len(thickness_values)}"
    )

    # ---------------------------------------------------------
    # Profile
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
    # Sphere fit
    #
    # r² + z² = 2*zc*z + C
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

    print(
        f"  Sphere radius        : "
        f"{sphere_radius:.6f} mm"
    )

    print(
        f"  Sphere center z      : "
        f"{sphere_center_z:.6f} mm"
    )

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


# =============================================================
# 12. Summary
# =============================================================

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
    "thickness_segment_heatmap.png"
)

print(
    "  - data/output/thickness_maps/"
    "thickness_table.csv"
)

print(
    "  - data/output/visualizations/"
    "uv_contour_map.png"
)