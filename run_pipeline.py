"""
End-to-end pipeline: CAD file -> patches -> thickness -> UV contour map.

Usage:
    python run_pipeline.py --input data/input/bowl.step --t_target 1.0 --n_patches 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.io_utils import (ensure_data_dirs, DATA_DIRS, save_thickness_csv, save_figure, load_cad_mesh)
from utils.visualization import plot_profile_and_patches, plot_thickness_vs_radius, plot_uv_contour
from numerical.patch_extraction import extract_axisymmetric_profile, segment_into_patches, flatten_patches_to_strip, check_axisymmetry
from numerical.backward_solve import backward_solve
from numerical.uv_contour import compute_uv_parametrization


def main():
    parser = argparse.ArgumentParser(description="ISF non-uniform thickness map pipeline")
    parser.add_argument("--input", required=True, help="Path to input CAD file (STEP/STL)")
    parser.add_argument("--t_target", type=float, default=1.0, help="Desired uniform final thickness (mm)")
    parser.add_argument("--n_patches", type=int, default=5, help="Number of patches to segment")
    args = parser.parse_args()

    ensure_data_dirs()

    print(f"Loading CAD file: {args.input}")
    mesh = load_cad_mesh(args.input)
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    axis_check = check_axisymmetry(mesh)
    print(f"Axisymmetry check: max radial spread = {axis_check['max_radial_spread']:.4f} mm "
          f"(likely_axisymmetric = {axis_check['likely_axisymmetric']})")
    if not axis_check["likely_axisymmetric"]:
        print("  WARNING: this part may not be a body of revolution -- "
              "results below assume axisymmetry and may not be valid.")

    print("Extracting profile and patches...")
    z_profile, r_profile = extract_axisymmetric_profile(mesh)
    patches = segment_into_patches(z_profile, r_profile, n_patches=args.n_patches)
    layout = flatten_patches_to_strip(patches)
    fig1 = plot_profile_and_patches(z_profile, r_profile, patches, layout)
    save_figure(fig1, os.path.join(DATA_DIRS["patches"], "profile_patches.png"))

    print(f"Running backward solve (t_df={args.t_target} mm)...")
    r_flat, t_initial = backward_solve(r_profile, z_profile, args.t_target)
    fig2 = plot_thickness_vs_radius(r_flat, t_initial, args.t_target)
    save_figure(fig2, os.path.join(DATA_DIRS["thickness_maps"], "thickness_vs_radius.png"))
    save_thickness_csv(
        os.path.join(DATA_DIRS["thickness_maps"], "thickness_table.csv"),
        r_profile, z_profile, [0] * len(r_profile), t_initial,
    )

    print("Generating UV contour map...")
    u, v, radius = compute_uv_parametrization(mesh.vertices)
    # tile t_initial across angle since the part is axisymmetric (see conversation notes)
    import numpy as np
    t_full = np.interp(u, z_profile, t_initial)
    fig3 = plot_uv_contour(u, v, t_full)
    save_figure(fig3, os.path.join(DATA_DIRS["visualizations"], "uv_contour_map.png"))

    print("\nPipeline complete. Outputs saved under data/output/ and data/intermediate/.")


if __name__ == "__main__":
    main()