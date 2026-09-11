"""
UV Parametrization + 2D Thickness Contour Map
================================================

Maps alpha-bin thickness predictions onto the actual CAD surface
and generates a 2D cylindrical UV contour map.

Pipeline:

    CAD vertices
        |
        v
    cylindrical UV coordinates
        |
        v
    alpha at every CAD vertex
        |
        v
    interpolate solver alpha-bin T0 values
        |
        v
    one T0 value per CAD vertex
        |
        v
    scattered UV interpolation
        |
        v
    2D thickness contour map

Important physical handling
---------------------------
The thickness solver provides predictions only over its physical
alpha range.

CAD vertices outside that range are NOT extrapolated or clamped.

Instead:

    out-of-range alpha -> NaN -> masked in UV map

This prevents the visualization from inventing thickness values
where the physical solver has no prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# ============================================================
# UV PARAMETRIZATION
# ============================================================

def compute_uv_parametrization(
    vertices: np.ndarray,
    axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    axis_origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
):
    """
    Convert 3D CAD vertices into cylindrical UV coordinates.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        CAD surface vertices.

    axis : ndarray, shape (3,)
        Forming-axis direction.

    axis_origin : ndarray, shape (3,)
        Point on the forming axis.

    Returns
    -------
    u : ndarray, shape (N,)
        Axial coordinate.

    v : ndarray, shape (N,)
        Circumferential angle in radians.

    radius : ndarray, shape (N,)
        Radial distance from the forming axis.
    """

    vertices = np.asarray(vertices, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis_origin = np.asarray(axis_origin, dtype=float)

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must have shape (N, 3), got {vertices.shape}"
        )

    if axis.shape != (3,):
        raise ValueError(
            f"axis must have shape (3,), got {axis.shape}"
        )

    if axis_origin.shape != (3,):
        raise ValueError(
            f"axis_origin must have shape (3,), got {axis_origin.shape}"
        )

    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-12:
        raise ValueError(
            "Axis vector cannot have zero length."
        )

    axis = axis / axis_norm

    # --------------------------------------------------------
    # Position relative to axis origin
    # --------------------------------------------------------

    rel = vertices - axis_origin

    # --------------------------------------------------------
    # Axial coordinate
    # --------------------------------------------------------

    u = rel @ axis

    # --------------------------------------------------------
    # Remove axial component
    # --------------------------------------------------------

    perp = rel - np.outer(u, axis)

    # --------------------------------------------------------
    # Radial distance
    # --------------------------------------------------------

    radius = np.linalg.norm(perp, axis=1)

    # --------------------------------------------------------
    # Construct perpendicular basis
    # --------------------------------------------------------

    if abs(axis[0]) < 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    else:
        reference = np.array([0.0, 1.0, 0.0])

    e1 = reference - np.dot(reference, axis) * axis

    e1_norm = np.linalg.norm(e1)

    if e1_norm < 1e-12:
        raise ValueError(
            "Could not construct perpendicular basis."
        )

    e1 /= e1_norm

    e2 = np.cross(axis, e1)

    e2_norm = np.linalg.norm(e2)

    if e2_norm < 1e-12:
        raise ValueError(
            "Could not construct second perpendicular basis."
        )

    e2 /= e2_norm

    # --------------------------------------------------------
    # Coordinates in plane perpendicular to axis
    # --------------------------------------------------------

    x_local = perp @ e1
    y_local = perp @ e2

    # --------------------------------------------------------
    # Circumferential angle
    # --------------------------------------------------------

    v = np.arctan2(
        y_local,
        x_local,
    )

    return u, v, radius


# ============================================================
# ALPHA CALCULATION
# ============================================================

def compute_alpha_from_sphere(
    vertices: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
):
    """
    Compute ISF alpha angle for every CAD surface vertex.

    Convention used by the thickness solver
    -----------------------------------------

        alpha = 0°   -> bottom / pole
        alpha = 90°  -> equator

    For the fitted sphere:

        r² + (z-zc)² = R²

    alpha is measured from the downward vertical direction.

    This is consistent with the current physical convention
    used by the thickness solver, where alpha is the forming
    angle used in:

        T0 = Tf / sin(alpha)
    """

    vertices = np.asarray(
        vertices,
        dtype=float,
    )

    sphere_center = np.asarray(
        sphere_center,
        dtype=float,
    )

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must have shape (N, 3), got {vertices.shape}"
        )

    if sphere_center.shape != (3,):
        raise ValueError(
            "sphere_center must have shape (3,), "
            f"got {sphere_center.shape}"
        )

    if (
        not np.isfinite(sphere_radius)
        or sphere_radius <= 0
    ):
        raise ValueError(
            f"Invalid sphere radius: {sphere_radius}"
        )

    # --------------------------------------------------------
    # Relative position to sphere center
    # --------------------------------------------------------

    rel = vertices - sphere_center

    # --------------------------------------------------------
    # Alpha from downward vertical direction
    #
    # alpha = acos(-(z-zc)/R)
    # --------------------------------------------------------

    cos_alpha = (
        -rel[:, 2]
        / sphere_radius
    )

    # Numerical protection
    cos_alpha = np.clip(
        cos_alpha,
        -1.0,
        1.0,
    )

    alpha_deg = np.degrees(
        np.arccos(cos_alpha)
    )

    return alpha_deg


# ============================================================
# THICKNESS INTERPOLATION
# ============================================================

def interpolate_thickness_to_surface(
    vertices: np.ndarray,
    alpha_centers: np.ndarray,
    thickness_values: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
):
    """
    Map alpha-bin thickness values onto every CAD vertex.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        CAD surface vertices.

    alpha_centers : ndarray, shape (M,)
        Centers of the physical alpha bins.

    thickness_values : ndarray, shape (M,)
        Initial thickness prediction for each alpha bin.

    sphere_center : ndarray, shape (3,)
        Fitted sphere center.

    sphere_radius : float
        Fitted sphere radius.

    Returns
    -------
    thickness_surface : ndarray, shape (N,)
        One thickness value per CAD vertex.

        Vertices whose alpha lies outside the solver range
        are assigned NaN.

    Notes
    -----
    There is intentionally NO extrapolation.

    If:

        alpha < min(alpha_centers)

    or:

        alpha > max(alpha_centers)

    the corresponding surface point receives NaN.

    This avoids silently clamping or inventing physical
    predictions outside the solver's valid alpha range.
    """

    vertices = np.asarray(
        vertices,
        dtype=float,
    )

    alpha_centers = np.asarray(
        alpha_centers,
        dtype=float,
    ).ravel()

    thickness_values = np.asarray(
        thickness_values,
        dtype=float,
    ).ravel()

    sphere_center = np.asarray(
        sphere_center,
        dtype=float,
    )

    # --------------------------------------------------------
    # Validate vertex array
    # --------------------------------------------------------

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must have shape (N, 3), got {vertices.shape}"
        )

    n_vertices = len(vertices)

    # --------------------------------------------------------
    # Validate solver arrays
    # --------------------------------------------------------

    if len(alpha_centers) != len(thickness_values):
        raise ValueError(
            "alpha_centers and thickness_values must have "
            "the same length.\n"
            f"alpha_centers : {len(alpha_centers)}\n"
            f"thickness_values : {len(thickness_values)}"
        )

    if len(alpha_centers) < 2:
        raise ValueError(
            "At least two alpha-bin values are required."
        )

    # --------------------------------------------------------
    # Remove invalid solver samples
    # --------------------------------------------------------

    valid = (
        np.isfinite(alpha_centers)
        & np.isfinite(thickness_values)
    )

    alpha_centers = alpha_centers[valid]
    thickness_values = thickness_values[valid]

    if len(alpha_centers) < 2:
        raise ValueError(
            "Not enough valid alpha-bin samples."
        )

    # --------------------------------------------------------
    # Sort by alpha
    # --------------------------------------------------------

    order = np.argsort(
        alpha_centers
    )

    alpha_centers = alpha_centers[order]
    thickness_values = thickness_values[order]

    # --------------------------------------------------------
    # Remove duplicate alpha centers
    # --------------------------------------------------------

    alpha_centers, unique_indices = np.unique(
        alpha_centers,
        return_index=True,
    )

    thickness_values = (
        thickness_values[unique_indices]
    )

    if len(alpha_centers) < 2:
        raise ValueError(
            "Alpha-bin centers are not sufficiently distinct."
        )

    # --------------------------------------------------------
    # Compute alpha for EVERY CAD vertex
    # --------------------------------------------------------

    alpha_surface = compute_alpha_from_sphere(
        vertices=vertices,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
    )

    # --------------------------------------------------------
    # Determine physical solver range
    # --------------------------------------------------------

    alpha_min = float(
        np.min(alpha_centers)
    )

    alpha_max = float(
        np.max(alpha_centers)
    )

    # --------------------------------------------------------
    # Identify vertices inside physical solver range
    # --------------------------------------------------------

    in_range = (
        np.isfinite(alpha_surface)
        & (alpha_surface >= alpha_min)
        & (alpha_surface <= alpha_max)
    )

    # --------------------------------------------------------
    # Initialize ALL surface thicknesses as NaN
    #
    # This is intentional.
    #
    # Any vertex that does not have a valid physical
    # prediction remains undefined.
    # --------------------------------------------------------

    thickness_surface = np.full(
        n_vertices,
        np.nan,
        dtype=float,
    )

    # --------------------------------------------------------
    # Interpolate ONLY inside physical solver range
    # --------------------------------------------------------

    thickness_surface[in_range] = np.interp(
        alpha_surface[in_range],
        alpha_centers,
        thickness_values,
    )

    # --------------------------------------------------------
    # Handle invalid alpha values explicitly
    # --------------------------------------------------------

    invalid_alpha = (
        ~np.isfinite(alpha_surface)
    )

    if np.any(invalid_alpha):
        thickness_surface[
            invalid_alpha
        ] = np.nan

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    n_total = len(alpha_surface)

    n_in_range = int(
        np.count_nonzero(in_range)
    )

    n_out_of_range = (
        n_total - n_in_range
    )

    print(
        f"  Solver alpha range : "
        f"{alpha_min:.3f}° to "
        f"{alpha_max:.3f}°"
    )

    if np.any(np.isfinite(alpha_surface)):
        print(
            f"  CAD alpha range    : "
            f"{np.nanmin(alpha_surface):.3f}° to "
            f"{np.nanmax(alpha_surface):.3f}°"
        )

    print(
        f"  In-range vertices   : "
        f"{n_in_range}/{n_total} "
        f"({100.0 * n_in_range / n_total:.2f}%)"
    )

    print(
        f"  Out-of-range        : "
        f"{n_out_of_range}/{n_total} "
        f"({100.0 * n_out_of_range / n_total:.2f}%)"
    )

    if n_out_of_range > 0:
        print(
            "  WARNING: Out-of-range CAD vertices "
            "are masked as NaN rather than "
            "extrapolated/clamped."
        )

    # --------------------------------------------------------
    # HARD DIMENSION CHECK
    # --------------------------------------------------------

    if len(alpha_surface) != n_vertices:
        raise RuntimeError(
            "Internal alpha dimension error:\n"
            f"CAD vertices : {n_vertices}\n"
            f"alpha        : {len(alpha_surface)}"
        )

    if len(thickness_surface) != n_vertices:
        raise RuntimeError(
            "Internal thickness dimension error:\n"
            f"CAD vertices : {n_vertices}\n"
            f"T0           : {len(thickness_surface)}"
        )

    return thickness_surface


# ============================================================
# UV CONTOUR PLOT
# ============================================================

def plot_uv_contour(
    u: np.ndarray,
    v: np.ndarray,
    T0: np.ndarray,
    grid_res: int = 200,
    title: str = "Initial Sheet Thickness T₀(u,v)",
):
    """
    Generate a regular 2D UV contour map.

    Invalid / undefined surface points are ignored.

    The scattered interpolation uses linear interpolation only.
    No nearest-neighbour extrapolation is applied outside the
    convex hull of valid UV samples.
    """

    u = np.asarray(
        u,
        dtype=float,
    ).ravel()

    v = np.asarray(
        v,
        dtype=float,
    ).ravel()

    T0 = np.asarray(
        T0,
        dtype=float,
    ).ravel()

    # --------------------------------------------------------
    # Dimension check
    # --------------------------------------------------------

    if not (
        len(u)
        == len(v)
        == len(T0)
    ):
        raise ValueError(
            "\nUV DATA SIZE MISMATCH\n"
            f"  u  : {len(u)} samples\n"
            f"  v  : {len(v)} samples\n"
            f"  T0 : {len(T0)} samples\n\n"
            "u, v and T0 must contain exactly one value "
            "per surface point."
        )

    # --------------------------------------------------------
    # Remove invalid samples
    # --------------------------------------------------------

    valid = (
        np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(T0)
    )

    u = u[valid]
    v = v[valid]
    T0 = T0[valid]

    if len(u) < 3:
        raise ValueError(
            "Not enough valid UV samples to generate contour."
        )

    # --------------------------------------------------------
    # UV ranges
    # --------------------------------------------------------

    u_min = np.min(u)
    u_max = np.max(u)

    v_min = np.min(v)
    v_max = np.max(v)

    if abs(u_max - u_min) < 1e-12:
        raise ValueError(
            "Degenerate U coordinate range."
        )

    if abs(v_max - v_min) < 1e-12:
        raise ValueError(
            "Degenerate V coordinate range."
        )

    # --------------------------------------------------------
    # Regular UV grid
    # --------------------------------------------------------

    u_grid = np.linspace(
        u_min,
        u_max,
        grid_res,
    )

    v_grid = np.linspace(
        v_min,
        v_max,
        grid_res,
    )

    UU, VV = np.meshgrid(
        u_grid,
        v_grid,
    )

    # --------------------------------------------------------
    # Scattered -> regular grid
    #
    # IMPORTANT:
    #
    # Linear interpolation is used only where the UV grid
    # lies inside the convex hull of valid samples.
    #
    # We intentionally do NOT fill missing values using
    # nearest-neighbour extrapolation.
    # --------------------------------------------------------

    points = np.column_stack(
        (u, v)
    )

    TT = griddata(
        points,
        T0,
        (UU, VV),
        method="linear",
    )

    # --------------------------------------------------------
    # Do not fill NaNs.
    #
    # NaN regions represent UV locations where no valid
    # thickness prediction is available.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    contour = ax.contourf(
        np.degrees(VV),
        UU,
        TT,
        levels=30,
        cmap="viridis",
    )

    fig.colorbar(
        contour,
        ax=ax,
        label="T₀ (mm)",
    )

    ax.set_xlabel(
        "v — angle around axis (deg)"
    )

    ax.set_ylabel(
        "u — height along axis (mm)"
    )

    ax.set_title(
        title
    )

    plt.tight_layout()

    return fig


# ============================================================
# COMPLETE UV MAP PIPELINE
# ============================================================

def generate_uv_contour_map(
    vertices: np.ndarray,
    alpha_centers: np.ndarray,
    thickness_values: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
    axis: np.ndarray = np.array([0.0, 0.0, 1.0]),
    axis_origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
    grid_res: int = 200,
    title: str = "Initial Sheet Thickness T₀(u,v)",
):
    """
    Complete CAD -> UV -> alpha -> thickness -> contour pipeline.

    Returns
    -------
    fig
        Matplotlib figure.

    u
        Axial UV coordinate for every CAD vertex.

    v
        Circumferential UV coordinate for every CAD vertex.

    thickness_surface
        One T0 value per CAD vertex.

    alpha_surface
        One alpha value per CAD vertex.

    Notes
    -----
    CAD vertices outside the solver alpha range are assigned
    NaN thickness values and therefore remain masked in the
    final UV contour map.
    """

    vertices = np.asarray(
        vertices,
        dtype=float,
    )

    alpha_centers = np.asarray(
        alpha_centers,
        dtype=float,
    ).ravel()

    thickness_values = np.asarray(
        thickness_values,
        dtype=float,
    ).ravel()

    n_vertices = len(vertices)

    # --------------------------------------------------------
    # Validate main input
    # --------------------------------------------------------

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must have shape (N, 3), got {vertices.shape}"
        )

    # --------------------------------------------------------
    # Initial diagnostics
    # --------------------------------------------------------

    print(
        f"  CAD surface vertices : "
        f"{n_vertices}"
    )

    print(
        f"  Solver alpha bins    : "
        f"{len(alpha_centers)}"
    )

    print(
        f"  Solver thickness     : "
        f"{len(thickness_values)}"
    )

    # --------------------------------------------------------
    # 1. UV parametrization
    # --------------------------------------------------------

    u, v, radius = compute_uv_parametrization(
        vertices=vertices,
        axis=axis,
        axis_origin=axis_origin,
    )

    if (
        len(u) != n_vertices
        or len(v) != n_vertices
        or len(radius) != n_vertices
    ):
        raise RuntimeError(
            "UV parametrization returned incorrect dimensions:\n"
            f"vertices : {n_vertices}\n"
            f"u        : {len(u)}\n"
            f"v        : {len(v)}\n"
            f"radius   : {len(radius)}"
        )

    # --------------------------------------------------------
    # 2. Alpha at every CAD vertex
    # --------------------------------------------------------

    alpha_surface = compute_alpha_from_sphere(
        vertices=vertices,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
    )

    if len(alpha_surface) != n_vertices:
        raise RuntimeError(
            "Alpha calculation returned incorrect dimensions:\n"
            f"vertices : {n_vertices}\n"
            f"alpha    : {len(alpha_surface)}"
        )

    # --------------------------------------------------------
    # 3. Map solver T0 onto every CAD vertex
    # --------------------------------------------------------

    thickness_surface = (
        interpolate_thickness_to_surface(
            vertices=vertices,
            alpha_centers=alpha_centers,
            thickness_values=thickness_values,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
        )
    )

    # --------------------------------------------------------
    # 4. Final dimension check
    # --------------------------------------------------------

    print(
        f"  Surface alpha       : "
        f"{len(alpha_surface)}"
    )

    print(
        f"  Surface thickness   : "
        f"{len(thickness_surface)}"
    )

    if len(thickness_surface) != n_vertices:
        raise RuntimeError(
            "FINAL UV DATA SIZE MISMATCH:\n"
            f"CAD vertices : {n_vertices}\n"
            f"u            : {len(u)}\n"
            f"v            : {len(v)}\n"
            f"alpha        : {len(alpha_surface)}\n"
            f"T0           : {len(thickness_surface)}"
        )

    # --------------------------------------------------------
    # Additional thickness diagnostics
    # --------------------------------------------------------

    valid_thickness = np.isfinite(
        thickness_surface
    )

    n_valid_thickness = int(
        np.count_nonzero(valid_thickness)
    )

    n_invalid_thickness = (
        n_vertices - n_valid_thickness
    )

    print(
        f"  Valid T0 vertices   : "
        f"{n_valid_thickness}/{n_vertices} "
        f"({100.0 * n_valid_thickness / n_vertices:.2f}%)"
    )

    print(
        f"  Masked T0 vertices  : "
        f"{n_invalid_thickness}/{n_vertices} "
        f"({100.0 * n_invalid_thickness / n_vertices:.2f}%)"
    )

    finite_thickness = thickness_surface[np.isfinite(thickness_surface)]

    if finite_thickness.size > 0:
        print(
            f"  Thickness range      : "
            f"{finite_thickness.min():.4f} to "
            f"{finite_thickness.max():.4f} mm"
        )
    else:
        print("  Thickness range      : no valid values")

    # --------------------------------------------------------
    # 5. Generate contour
    # --------------------------------------------------------

    fig = plot_uv_contour(
        u=u,
        v=v,
        T0=thickness_surface,
        grid_res=grid_res,
        title=title,
    )

    # --------------------------------------------------------
    # 6. Return
    # --------------------------------------------------------

    return (
        fig,
        u,
        v,
        thickness_surface,
        alpha_surface,
    )