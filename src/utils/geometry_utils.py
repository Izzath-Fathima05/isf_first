"""
Shared geometric helper functions used across both the numerical and
CAD-native pipelines. Extracted here to avoid duplicating the same
math in point_projection.py and cad_native_patches.py.
"""

import numpy as np


def polygon_centroid_2d(points: np.ndarray):
    """
    Proper area-weighted polygon centroid (not a naive vertex average),
    via the standard shoelace-based formula. Used by
    point_projection.pappus_element_volume() for the ABCD quadrilateral.

    Parameters
    ----------
    points : (N, 2) array of (x, y) vertices, in order (CW or CCW).

    Returns
    -------
    (cx, cy), area
    """
    x, y = points[:, 0], points[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * np.sum(cross)

    area = abs(signed_area)
    if area < 1e-12:
        return (x.mean(), y.mean()), area

    cx = np.sum((x + x_next) * cross) / (6 * signed_area)
    cy = np.sum((y + y_next) * cross) / (6 * signed_area)
    return (cx, cy), area


def outward_normal_2d(pA: np.ndarray, pB: np.ndarray) -> np.ndarray:
    """
    Unit normal to segment AB, pointing toward increasing r (outward
    from the axis of revolution). Used by point_projection.py to
    construct the ABCD quadrilateral, and matches the same convention
    used for flat-blank elements (normal = local thickness direction).
    """
    alpha = np.arctan2(pB[1] - pA[1], pB[0] - pA[0])
    return np.array([np.sin(alpha), -np.cos(alpha)])


def cone_unroll_params(half_angle_rad: float, slant_inner: float, slant_outer: float):
    """
    Closed-form flat development of a conical frustum band.
    Returns the sector angle (radians) for the flattened annular sector.
    See cad_native_patches.py's flatten_conical_patch() for usage.
    """
    sector_angle = 2 * np.pi * np.sin(half_angle_rad)
    return {
        "slant_inner": slant_inner,
        "slant_outer": slant_outer,
        "sector_angle_rad": sector_angle,
        "sector_angle_deg": np.degrees(sector_angle),
    }


def cylinder_unroll_params(radius: float, u_range: tuple, v_range: tuple):
    """
    Closed-form flat development of a cylindrical face: a rectangle of
    width = arc length, height = axial extent. See
    general_flatten.py's flatten_cylindrical_face() for usage.
    """
    u1, u2 = u_range
    v1, v2 = v_range
    return {"width": radius * (u2 - u1), "height": v2 - v1}


def check_axisymmetry_spread(vertices: np.ndarray, axis: np.ndarray,
                              axis_origin: np.ndarray, n_bins: int = 20) -> float:
    """
    Returns the max radial spread within height-bins -- a diagnostic
    for whether the axisymmetry assumption holds for a given mesh.
    Shared by patch_extraction.check_axisymmetry().
    """
    axis = axis / np.linalg.norm(axis)
    rel = vertices - axis_origin
    u = rel @ axis
    perp = rel - np.outer(u, axis)
    radius = np.linalg.norm(perp, axis=1)

    bins = np.linspace(u.min(), u.max(), n_bins + 1)
    bin_idx = np.digitize(u, bins) - 1
    spreads = [radius[bin_idx == b].std() for b in range(n_bins) if (bin_idx == b).sum() > 1]
    return max(spreads) if spreads else 0.0