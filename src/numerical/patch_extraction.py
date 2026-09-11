"""
Axisymmetric profile extraction for ISF.

The numerical pipeline operates on the SPIF-exposed surface only.

For the current bowl STEP:
    outer sphere radius = 50 mm
    inner sphere radius = 45 mm

The outer surface is therefore reconstructed as:

    r(z) = sqrt(R^2 - (z-zc)^2)

This avoids mixing the inner and outer shell surfaces.
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


EPS = 1e-12


def load_cad_as_mesh(path: str):
    """Compatibility loader."""

    return trimesh.load(
        path,
        force="mesh",
    )


def _radial_coordinates(
    mesh,
    axis=np.array([0.0, 0.0, 1.0]),
    axis_origin=np.array([0.0, 0.0, 0.0]),
):
    """Convert Cartesian coordinates to axial/radial coordinates."""

    axis = np.asarray(
        axis,
        dtype=float,
    )

    axis /= np.linalg.norm(axis)

    axis_origin = np.asarray(
        axis_origin,
        dtype=float,
    )

    rel = mesh.vertices - axis_origin

    z = rel @ axis

    perpendicular = (
        rel
        - np.outer(z, axis)
    )

    r = np.linalg.norm(
        perpendicular,
        axis=1,
    )

    return z, r


def _fit_circle_to_profile(
    r,
    z,
):
    """
    Fit:

        r^2 + (z-zc)^2 = R^2

    using linear least squares.

    Returns
    -------
    R
    zc
    rmse
    """

    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)

    mask = (
        np.isfinite(r)
        & np.isfinite(z)
        & (r > 0)
    )

    r = r[mask]
    z = z[mask]

    if len(r) < 10:
        return None

    # Circle equation:
    #
    # r² + z² - 2*zc*z + c = R²
    #
    # => -2*zc*z + c = -(r² + z²)
    #
    A = np.column_stack([
        z,
        np.ones_like(z),
    ])

    b = -(
        r ** 2
        + z ** 2
    )

    coeff, *_ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    a = coeff[0]
    c = coeff[1]

    zc = -a / 2.0

    R_squared = (
        zc ** 2
        - c
    )

    if R_squared <= 0:
        return None

    R = np.sqrt(
        R_squared
    )

    predicted_r_squared = (
        R ** 2
        - (z - zc) ** 2
    )

    valid = (
        predicted_r_squared > 0
    )

    if not np.any(valid):
        return None

    predicted_r = np.sqrt(
        predicted_r_squared[valid]
    )

    rmse = np.sqrt(
        np.mean(
            (
                predicted_r
                - r[valid]
            ) ** 2
        )
    )

    return {
        "radius": R,
        "center_z": zc,
        "rmse": rmse,
    }


def _extract_outer_envelope(
    mesh,
    n_bins=400,
):
    """
    Extract approximate outer radial envelope.

    This is only used to identify/fitting the analytic surface.
    """

    z, r = _radial_coordinates(
        mesh
    )

    bins = np.linspace(
        z.min(),
        z.max(),
        n_bins + 1,
    )

    z_values = []
    r_values = []

    for i in range(
        n_bins
    ):

        mask = (
            (z >= bins[i])
            & (z < bins[i + 1])
        )

        if mask.sum() < 3:
            continue

        z_local = z[mask]
        r_local = r[mask]

        # Upper radial envelope.
        r_value = np.percentile(
            r_local,
            98.0,
        )

        z_value = np.mean(
            z_local
        )

        z_values.append(
            z_value
        )

        r_values.append(
            r_value
        )

    return (
        np.asarray(z_values),
        np.asarray(r_values),
    )


def check_axisymmetry(
    mesh,
    axis=np.array([0.0, 0.0, 1.0]),
    axis_origin=np.array([0.0, 0.0, 0.0]),
    surface="outer",
    n_bins=400,
):
    """
    Check axisymmetry by fitting the selected surface to an
    axisymmetric profile.

    IMPORTANT:
    We do NOT compute standard deviation of radius inside a z-bin,
    because radius naturally changes with z.
    """

    if surface != "outer":
        raise NotImplementedError(
            "Current robust implementation "
            "supports surface='outer'."
        )

    z, r = _extract_outer_envelope(
        mesh,
        n_bins=n_bins,
    )

    fit = _fit_circle_to_profile(
        r,
        z,
    )

    if fit is None:

        return {
            "likely_axisymmetric": False,
            "max_radial_spread": np.inf,
            "relative_spread": np.inf,
            "radius": None,
            "center_z": None,
        }

    R = fit["radius"]
    zc = fit["center_z"]

    predicted = (
        R ** 2
        - (z - zc) ** 2
    )

    valid = predicted > 0

    predicted_r = np.sqrt(
        predicted[valid]
    )

    actual_r = r[valid]

    residual = (
        actual_r
        - predicted_r
    )

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    relative = (
        rmse / R
    )

    return {
        "likely_axisymmetric": relative < 0.01,
        "max_radial_spread": rmse,
        "relative_spread": relative,
        "radius": R,
        "center_z": zc,
    }


def extract_axisymmetric_profile(
    mesh,
    n_profile_points=361,
    axis=np.array([0.0, 0.0, 1.0]),
    axis_origin=np.array([0.0, 0.0, 0.0]),
    surface="outer",
):
    """
    Extract a clean axisymmetric profile.

    For a spherical surface, reconstruct the profile analytically
    from the fitted sphere.
    """

    if surface != "outer":
        raise NotImplementedError(
            "Current implementation supports "
            "surface='outer'."
        )

    z_raw, r_raw = _extract_outer_envelope(
        mesh,
        n_bins=500,
    )

    fit = _fit_circle_to_profile(
        r_raw,
        z_raw,
    )

    if fit is None:
        raise RuntimeError(
            "Could not fit the outer axisymmetric surface."
        )

    R = fit["radius"]
    zc = fit["center_z"]

    # Restrict to the actual observed geometry.
    z_min = float(
        np.min(z_raw)
    )

    z_max = float(
        np.max(z_raw)
    )

    z_limit = R

    z_min = max(
        z_min,
        zc - z_limit,
    )

    z_max = min(
        z_max,
        zc + z_limit,
    )

    z_profile = np.linspace(
        z_min,
        z_max,
        n_profile_points,
    )

    radicand = (
        R ** 2
        - (z_profile - zc) ** 2
    )

    radicand = np.maximum(
        radicand,
        0.0,
    )

    r_profile = np.sqrt(
        radicand
    )

    # Sort by increasing z.
    order = np.argsort(
        z_profile
    )

    z_profile = z_profile[order]
    r_profile = r_profile[order]

    return (
        z_profile,
        r_profile,
    )


def segment_into_patches(
    z_profile,
    r_profile,
    n_patches=5,
):
    """Divide profile into equal arc-length patches."""

    z_profile = np.asarray(
        z_profile,
        dtype=float,
    )

    r_profile = np.asarray(
        r_profile,
        dtype=float,
    )

    ds = np.hypot(
        np.diff(r_profile),
        np.diff(z_profile),
    )

    cumulative = np.concatenate([
        [0.0],
        np.cumsum(ds),
    ])

    total = cumulative[-1]

    boundaries = np.linspace(
        0.0,
        total,
        n_patches + 1,
    )

    patches = []

    for i in range(
        n_patches
    ):

        mask = (
            (cumulative >= boundaries[i])
            &
            (cumulative <= boundaries[i + 1])
        )

        idx = np.where(
            mask
        )[0]

        if len(idx) < 2:

            idx = np.searchsorted(
                cumulative,
                [
                    boundaries[i],
                    boundaries[i + 1],
                ],
            )

            idx = np.clip(
                idx,
                0,
                len(cumulative) - 1,
            )

        patches.append({
            "z": z_profile[idx],
            "r": r_profile[idx],
            "arc_length": (
                boundaries[i + 1]
                - boundaries[i]
            ),
            "label": chr(
                ord("A") + i
            ),
        })

    return patches


def flatten_patches_to_strip(
    patches,
    thickness=None,
):
    """Flatten patches into a strip."""

    if thickness is None:
        thickness = np.ones(
            len(patches)
        )

    x_start = 0.0
    layout = []

    for patch, t in zip(
        patches,
        thickness,
    ):

        layout.append({
            "label": patch["label"],
            "x_start": x_start,
            "width": patch["arc_length"],
            "height": t,
        })

        x_start += patch["arc_length"]

    return layout


def plot_profile_and_patches(
    z_profile,
    r_profile,
    patches,
    layout,
    save_path,
):
    """Plot profile and patches."""

    colors = plt.cm.tab10(
        np.linspace(
            0,
            1,
            len(patches),
        )
    )

    fig, (
        ax1,
        ax2,
    ) = plt.subplots(
        2,
        1,
        figsize=(9, 10),
    )

    ax1.plot(
        r_profile,
        z_profile,
        color="black",
        linewidth=1.5,
    )

    for patch, color in zip(
        patches,
        colors,
    ):

        ax1.plot(
            patch["r"],
            patch["z"],
            color=color,
            linewidth=5,
        )

    ax1.set_xlabel(
        "r (mm)"
    )

    ax1.set_ylabel(
        "z (mm)"
    )

    ax1.set_title(
        "SPIF forming surface"
    )

    ax1.set_aspect(
        "equal"
    )

    for item, color in zip(
        layout,
        colors,
    ):

        rect = mpatches.Rectangle(
            (
                item["x_start"],
                0,
            ),
            item["width"],
            item["height"],
            facecolor=color,
            edgecolor="black",
        )

        ax2.add_patch(
            rect
        )

        ax2.annotate(
            item["label"],
            (
                item["x_start"]
                + item["width"] / 2,
                -0.15,
            ),
            ha="center",
        )

    ax2.set_xlim(
        0,
        layout[-1]["x_start"]
        + layout[-1]["width"],
    )

    ax2.set_xlabel(
        "Flattened arc length (mm)"
    )

    ax2.set_ylabel(
        "Thickness (mm)"
    )

    ax2.set_title(
        "Flattened patches"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)