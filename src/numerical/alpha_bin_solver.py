"""
Alpha-bin physical-element thickness solver for axisymmetric ISF.

Architecture
------------
RAW PROFILE
    |
    v
orient profile in ascending z
    |
    v
compute geometric tangent angle theta
    |
    v
convert theta -> sine-law angle alpha
    |
    +-----------------------------+
    |                             |
constant-angle region       genuinely varying region
    |                             |
    v                             v
ONE physical element       check physical arc length
                                  |
                         +--------+--------+
                         |                 |
                    short transition   sufficiently long
                         |                 |
                         v                 v
                  ONE physical       split at alpha
                     element         2-degree bins
                                          |
                                          v
                                actual geometric boundaries
                                          |
                                          v
                                   Pappus solve
                                          |
                                          v
                                   t_initial
                                          |
                                          v
                                  output / UV map

Important
---------

A short geometric transition such as a corner/fillet can have a
large change in alpha while occupying only a tiny physical distance.
Such a region is treated as ONE physical element.

For a sufficiently long smooth varying region, alpha bins are used.

The numerical Pappus solve is the authoritative thickness solution.

"""

import numpy as np
from scipy.optimize import brentq

from .point_projection import pappus_element_volume


# ======================================================================
# 1. GEOMETRY -> TANGENT ANGLE
# ======================================================================

def compute_tangent_angles(r_profile, z_profile):
    """
    Compute the geometric tangent angle theta of an axisymmetric profile.

    Parameters
    ----------
    r_profile : array-like
        Radial coordinates.
    z_profile : array-like
        Axial coordinates.

    Returns
    -------
    theta : ndarray
        Tangent angle in degrees, normalized to [0, 180].

    Notes
    -----
    theta is the geometric profile tangent angle.

    The derivative is computed after the profile has been oriented,
    so reversing the input profile does not produce inconsistent
    theta/alpha values.
    """

    r = np.asarray(r_profile, dtype=float)
    z = np.asarray(z_profile, dtype=float)

    if r.ndim != 1 or z.ndim != 1:
        raise ValueError(
            "r_profile and z_profile must be 1-D arrays."
        )

    if len(r) != len(z):
        raise ValueError(
            "r_profile and z_profile must have the same length."
        )

    if len(r) < 2:
        raise ValueError(
            "At least two profile points are required."
        )

    if not np.all(np.isfinite(r)):
        raise ValueError(
            "r_profile contains non-finite values."
        )

    if not np.all(np.isfinite(z)):
        raise ValueError(
            "z_profile contains non-finite values."
        )

    dr = np.gradient(r)
    dz = np.gradient(z)

    theta = np.degrees(
        np.arctan2(dz, dr)
    )

    # Normalize to [0, 360)
    theta = np.mod(
        theta,
        360.0,
    )

    # Fold to [0, 180]
    theta = np.where(
        theta > 180.0,
        360.0 - theta,
        theta,
    )

    return theta

def tangent_to_sine_law_angle(theta_deg):
    """
    Convert geometric tangent angle theta to the alpha convention used
    by the sine-law thickness relation.
    """

    theta = np.asarray(
        theta_deg,
        dtype=float,
    )

    if not np.all(np.isfinite(theta)):
        raise ValueError(
            "theta contains non-finite values."
        )

    theta_acute = np.minimum(
        theta,
        180.0 - theta,
    )

    alpha = 90.0 - theta_acute

    return alpha

#sine law estimation
def sine_law_initial_thickness(
    t_final,
    alpha_deg,
):
    """
    Basic sine-law estimate:
    t_initial = t_final / sin(alpha)
    """

    if t_final <= 0:
        raise ValueError(
            "t_final must be positive."
        )

    alpha = np.asarray(
        alpha_deg,
        dtype=float,
    )

    sin_alpha = np.sin(
        np.radians(alpha)
    )

    eps = 1e-12

    sin_alpha = np.maximum(
        np.abs(sin_alpha),
        eps,
    )

    return t_final / sin_alpha

#pappus
def sine_law_correction_factor(
    r_mid,
    alpha_deg,
    t_final,
):
    """
    Analytic Pappus centroid correction.

    Current validated sign:

        C = 1 - t_final*cos(alpha)/(2*r_mid)

    This is used ONLY as a diagnostic estimate.

    The actual thickness used by the solver comes from the numerical
    Pappus volume-conservation solve.
    """

    if r_mid <= 0:
        raise ValueError(
            f"r_mid must be positive, got {r_mid}."
        )

    if t_final <= 0:
        raise ValueError(
            "t_final must be positive."
        )

    alpha = np.radians(
        alpha_deg
    )

    correction = (
        1.0
        - (
            t_final
            * np.cos(alpha)
        )
        / (
            2.0 * r_mid
        )
    )

    return float(correction)


def estimate_thickness_analytic(
    t_final,
    alpha_deg,
    r_start,
    r_end,
    z_start,
    z_end,
):
    """
    Diagnostic analytic thickness estimate.

    z_start and z_end are accepted for API consistency but are not
    explicitly required by this simplified analytic expression.
    """

    del z_start, z_end

    t_sl = float(
        sine_law_initial_thickness(
            t_final,
            alpha_deg,
        )
    )

    r_mid = 0.5 * (
        r_start + r_end
    )

    correction = sine_law_correction_factor(
        r_mid,
        alpha_deg,
        t_final,
    )

    return float(
        t_sl * correction
    )


# ======================================================================
# 5. NUMERICAL PAPPUS SOLVE FOR ONE PHYSICAL ELEMENT
# ======================================================================

def solve_single_element_thickness(
    r_start,
    z_start,
    r_end,
    z_end,
    t_final,
    t_min=1e-6,
    t_max_factor=100.0,
):
    """
    Solve one physical element thickness by volume conservation.

    Formed element:
        actual profile segment + final thickness t_final

    Source element:
        same radial endpoints mapped to z = 0
        + unknown initial thickness t_initial

    The root satisfies:

        V_source(t_initial) = V_formed(t_final)
    """

    values = [
        r_start,
        z_start,
        r_end,
        z_end,
        t_final,
    ]

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "Element geometry contains non-finite values."
        )

    if t_final <= 0:
        raise ValueError(
            "t_final must be positive."
        )

    if t_min <= 0:
        raise ValueError(
            "t_min must be positive."
        )

    if t_max_factor <= 1.0:
        raise ValueError(
            "t_max_factor must be greater than 1."
        )

    # --------------------------------------------------------------
    # Target formed volume
    # --------------------------------------------------------------

    target_volume = pappus_element_volume(
        r_start,
        z_start,
        r_end,
        z_end,
        t_final,
        t_final,
    )

    target_volume = float(
        target_volume
    )

    if not np.isfinite(
        target_volume
    ):
        raise RuntimeError(
            "Pappus target volume is non-finite."
        )

    if target_volume <= 0:
        raise RuntimeError(
            "Pappus target volume must be positive."
        )

    # --------------------------------------------------------------
    # Source-volume residual
    # --------------------------------------------------------------

    def residual(
        t_initial,
    ):
        source_volume = pappus_element_volume(
            r_start,
            0.0,
            r_end,
            0.0,
            t_initial,
            t_initial,
        )

        return float(
            source_volume - target_volume
        )

    # --------------------------------------------------------------
    # Bracket the root
    # --------------------------------------------------------------

    low = float(
        t_min
    )

    high = max(
        t_final * t_max_factor,
        t_final * 2.0,
        1.0,
    )

    f_low = residual(
        low
    )

    f_high = residual(
        high
    )

    expansion_count = 0

    while (
        f_low * f_high > 0.0
        and expansion_count < 50
    ):
        high *= 2.0

        f_high = residual(
            high
        )

        expansion_count += 1

    if (
        f_low * f_high > 0.0
    ):
        raise RuntimeError(
            "Could not bracket thickness root.\n"
            f"target_volume = {target_volume:.12e}\n"
            f"low           = {low:.12e}\n"
            f"high          = {high:.12e}\n"
            f"f_low         = {f_low:.12e}\n"
            f"f_high        = {f_high:.12e}"
        )

    # --------------------------------------------------------------
    # Root solve
    # --------------------------------------------------------------

    t_initial = brentq(
        residual,
        low,
        high,
        xtol=1e-12,
        rtol=1e-10,
        maxiter=200,
    )

    t_initial = float(
        t_initial
    )

    if (
        not np.isfinite(
            t_initial
        )
        or t_initial <= 0
    ):
        raise RuntimeError(
            f"Invalid solved thickness: {t_initial}"
        )

    return (
        t_initial,
        target_volume,
    )


# ======================================================================
# 6. PROFILE SAMPLING SPACING
# ======================================================================

def profile_median_spacing(
    r,
    z,
):
    """
    Robust characteristic profile point spacing.

    This is used only to decide whether a varying transition is
    physically long enough to justify alpha-bin discretization.
    """

    r = np.asarray(
        r,
        dtype=float,
    )

    z = np.asarray(
        z,
        dtype=float,
    )

    if len(r) < 2:
        raise ValueError(
            "At least two profile points are required."
        )

    ds = np.hypot(
        np.diff(r),
        np.diff(z),
    )

    ds = ds[
        ds > 1e-12
    ]

    if len(ds) == 0:
        raise ValueError(
            "Profile contains no non-zero-length segments."
        )

    return float(
        np.median(ds)
    )


# ======================================================================
# 7. DETECT CONSTANT / VARYING REGIONS
# ======================================================================

def detect_physical_regions(
    theta,
    const_tol_deg=1e-6,
):
    """
    Detect contiguous constant-angle and varying-angle regions.

    A region is classified based on the local difference between
    consecutive theta values.

    Parameters
    ----------
    theta : ndarray
    const_tol_deg : float
        Angular tolerance for declaring two adjacent samples equal.

    Returns
    -------
    list of dict
    """

    theta = np.asarray(
        theta,
        dtype=float,
    )

    if theta.ndim != 1:
        raise ValueError(
            "theta must be 1-D."
        )

    if len(theta) == 0:
        return []

    if len(theta) == 1:
        return [
            {
                "start": 0,
                "end": 0,
                "kind": "constant",
            }
        ]

    if const_tol_deg < 0:
        raise ValueError(
            "const_tol_deg cannot be negative."
        )

    dtheta = np.abs(
        np.diff(theta)
    )

    is_constant = (
        dtheta
        <= const_tol_deg
    )

    regions = []

    start = 0

    current_kind = (
        "constant"
        if is_constant[0]
        else "varying"
    )

    for i in range(
        1,
        len(is_constant),
    ):

        next_kind = (
            "constant"
            if is_constant[i]
            else "varying"
        )

        if (
            next_kind
            != current_kind
        ):
            regions.append(
                {
                    "start": start,
                    "end": i,
                    "kind": current_kind,
                }
            )

            start = i
            current_kind = (
                next_kind
            )

    regions.append(
        {
            "start": start,
            "end": len(theta) - 1,
            "kind": current_kind,
        }
    )

    return regions


# ======================================================================
# 8. SPLIT VARYING REGION AT ALPHA TURNING POINTS
# ======================================================================

def split_monotonic_alpha_regions(
    alpha,
    start,
    end,
    tolerance=1e-8,
):
    """
    Split a varying region into monotonic-alpha pieces.

    This prevents one alpha-bin traversal from crossing a local
    maximum/minimum of alpha.
    """

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    if start < 0 or end >= len(alpha):
        raise IndexError(
            "Invalid start/end indices."
        )

    if end < start:
        raise ValueError(
            "end must be >= start."
        )

    a = alpha[
        start:end + 1
    ]

    if len(a) < 2:
        return [
            {
                "start": start,
                "end": end,
            }
        ]

    da = np.diff(a)

    signs = np.zeros(
        len(da),
        dtype=int,
    )

    signs[
        da > tolerance
    ] = 1

    signs[
        da < -tolerance
    ] = -1

    nonzero = np.flatnonzero(
        signs
    )

    if len(nonzero) == 0:
        return [
            {
                "start": start,
                "end": end,
            }
        ]

    regions = []

    region_start = start

    previous_sign = int(
        signs[
            nonzero[0]
        ]
    )

    for k in range(
        nonzero[0] + 1,
        len(signs),
    ):

        current_sign = int(
            signs[k]
        )

        if (
            current_sign != 0
            and current_sign
            != previous_sign
        ):

            turn_index = (
                start + k
            )

            regions.append(
                {
                    "start": region_start,
                    "end": turn_index,
                }
            )

            region_start = (
                turn_index
            )

            previous_sign = (
                current_sign
            )

    regions.append(
        {
            "start": region_start,
            "end": end,
        }
    )

    return regions


# ======================================================================
# 9. INTERPOLATE PROFILE AT ALPHA
# ======================================================================

def interpolate_profile_at_alpha(
    r0,
    z0,
    a0,
    r1,
    z1,
    a1,
    alpha_target,
):
    """
    Linearly interpolate the actual profile segment at a requested
    alpha value.

    This function is the single source of truth for alpha-boundary
    interpolation.
    """

    denominator = (
        a1 - a0
    )

    if abs(
        denominator
    ) < 1e-14:
        raise ValueError(
            "Cannot interpolate alpha on a constant-alpha segment."
        )

    s = (
        alpha_target - a0
    ) / denominator

    if (
        s < -1e-8
        or s > 1.0 + 1e-8
    ):
        raise ValueError(
            "Requested alpha lies outside the segment."
        )

    s = np.clip(
        s,
        0.0,
        1.0,
    )

    r = (
        r0
        + s * (r1 - r0)
    )

    z = (
        z0
        + s * (z1 - z0)
    )

    return (
        float(r),
        float(z),
    )


# ======================================================================
# 10. EXTRACT ACTUAL GEOMETRY INSIDE AN ALPHA BIN
# ======================================================================

def extract_alpha_bin_geometry(
    r,
    z,
    alpha,
    start,
    end,
    bin_start,
    bin_end,
    tolerance=1e-10,
):
    """
    Extract the actual profile portion lying inside an alpha bin.

    Unlike point masking, this explicitly interpolates where the
    physical profile crosses the alpha-bin boundaries.

    This guarantees that neighboring alpha elements share geometric
    boundaries.
    """

    r = np.asarray(
        r,
        dtype=float,
    )

    z = np.asarray(
        z,
        dtype=float,
    )

    alpha = np.asarray(
        alpha,
        dtype=float,
    )

    rr = r[
        start:end + 1
    ]

    zz = z[
        start:end + 1
    ]

    aa = alpha[
        start:end + 1
    ]

    if len(rr) < 2:
        return None

    points = []

    # --------------------------------------------------------------
    # Traverse each original profile segment
    # --------------------------------------------------------------

    for k in range(
        len(rr) - 1
    ):

        r0 = float(
            rr[k]
        )

        z0 = float(
            zz[k]
        )

        a0 = float(
            aa[k]
        )

        r1 = float(
            rr[k + 1]
        )

        z1 = float(
            zz[k + 1]
        )

        a1 = float(
            aa[k + 1]
        )

        seg_min = min(
            a0,
            a1,
        )

        seg_max = max(
            a0,
            a1,
        )

        if (
            seg_max
            < bin_start - tolerance
        ):
            continue

        if (
            seg_min
            > bin_end + tolerance
        ):
            continue

        candidate_alphas = []

        # ----------------------------------------------------------
        # Bin boundaries
        # ----------------------------------------------------------

        if (
            seg_min - tolerance
            <= bin_start
            <= seg_max + tolerance
        ):
            candidate_alphas.append(
                bin_start
            )

        if (
            seg_min - tolerance
            <= bin_end
            <= seg_max + tolerance
        ):
            candidate_alphas.append(
                bin_end
            )

        # ----------------------------------------------------------
        # Existing profile endpoints
        # ----------------------------------------------------------

        if (
            bin_start - tolerance
            <= a0
            <= bin_end + tolerance
        ):
            candidate_alphas.append(
                a0
            )

        if (
            bin_start - tolerance
            <= a1
            <= bin_end + tolerance
        ):
            candidate_alphas.append(
                a1
            )

        # ----------------------------------------------------------
        # Interpolate each candidate
        # ----------------------------------------------------------

        for target_alpha in (
            candidate_alphas
        ):

            if abs(
                a1 - a0
            ) < tolerance:

                # Constant-alpha segment.
                # It contributes only if it lies inside the bin.
                if abs(
                    target_alpha - a0
                ) > tolerance:
                    continue

                s = 0.0

                rp = r0
                zp = z0

            else:

                s = (
                    target_alpha - a0
                ) / (
                    a1 - a0
                )

                if (
                    s < -tolerance
                    or s
                    > 1.0 + tolerance
                ):
                    continue

                rp, zp = (
                    interpolate_profile_at_alpha(
                        r0,
                        z0,
                        a0,
                        r1,
                        z1,
                        a1,
                        target_alpha,
                    )
                )

            points.append(
                (
                    float(target_alpha),
                    float(rp),
                    float(zp),
                )
            )

    if len(points) < 2:
        return None

    # --------------------------------------------------------------
    # Preserve physical traversal direction
    # --------------------------------------------------------------

    increasing = (
        aa[-1] >= aa[0]
    )

    points.sort(
        key=lambda p: p[0],
        reverse=not increasing,
    )

    # --------------------------------------------------------------
    # Remove duplicate geometric points
    # --------------------------------------------------------------

    cleaned = []

    for point in points:

        if not cleaned:
            cleaned.append(
                point
            )
            continue

        _, r_prev, z_prev = (
            cleaned[-1]
        )

        _, r_curr, z_curr = (
            point
        )

        if (
            np.hypot(
                r_curr - r_prev,
                z_curr - z_prev,
            )
            < tolerance
        ):
            continue

        cleaned.append(
            point
        )

    if len(cleaned) < 2:
        return None

    alpha_bin = np.array(
        [p[0] for p in cleaned],
        dtype=float,
    )

    r_bin = np.array(
        [p[1] for p in cleaned],
        dtype=float,
    )

    z_bin = np.array(
        [p[2] for p in cleaned],
        dtype=float,
    )

    return (
        r_bin,
        z_bin,
        alpha_bin,
    )


# ======================================================================
# 11. BUILD ONE PHYSICAL ELEMENT
# ======================================================================

def build_element_from_endpoints(
    r_start,
    z_start,
    r_end,
    z_end,
    alpha_mid,
    theta_mid,
    t_final,
    kind,
    alpha_bin_index=None,
    alpha_bin_start=None,
    alpha_bin_end=None,
):
    """
    Build one physical element and solve its initial thickness.
    """

    dr = (
        r_end - r_start
    )

    dz = (
        z_end - z_start
    )

    arc_length = float(
        np.hypot(
            dr,
            dz,
        )
    )

    if arc_length < 1e-12:
        return None

    (
        t_initial,
        target_volume,
    ) = solve_single_element_thickness(
        r_start,
        z_start,
        r_end,
        z_end,
        t_final,
    )

    # Diagnostic analytic result
    try:

        t_analytic = float(
            estimate_thickness_analytic(
                t_final=t_final,
                alpha_deg=alpha_mid,
                r_start=r_start,
                r_end=r_end,
                z_start=z_start,
                z_end=z_end,
            )
        )

    except (
        ValueError,
        FloatingPointError,
    ):

        t_analytic = np.nan

    # Diagnostic error
    if (
        np.isfinite(t_analytic)
        and abs(t_initial) > 1e-14
    ):

        analytic_error_percent = (
            abs(
                t_analytic
                - t_initial
            )
            / abs(t_initial)
            * 100.0
        )

    else:

        analytic_error_percent = np.nan

    return {
        "kind": kind,

        "start_idx": None,
        "end_idx": None,

        "r_start": float(
            r_start
        ),
        "z_start": float(
            z_start
        ),

        "r_end": float(
            r_end
        ),
        "z_end": float(
            z_end
        ),

        "dr": float(
            dr
        ),
        "dz": float(
            dz
        ),

        "arc_length": float(
            arc_length
        ),

        "theta_mid": float(
            theta_mid
        ),
        "alpha_mid": float(
            alpha_mid
        ),

        "t_initial": float(
            t_initial
        ),
        "t_analytic": float(
            t_analytic
        ),

        "analytic_error_percent": float(
            analytic_error_percent
        ),

        "t_final": float(
            t_final
        ),

        "target_volume": float(
            target_volume
        ),

        "alpha_bin_index":
            alpha_bin_index,

        "alpha_bin_start":
            alpha_bin_start,

        "alpha_bin_end":
            alpha_bin_end,
    }


# ======================================================================
# 12. SOLVE CONSTANT-ANGLE REGION
# ======================================================================

def solve_constant_region(
    r,
    z,
    theta,
    alpha,
    i0,
    i1,
    t_final,
):
    """
    A constant-angle region receives ONE physical thickness unknown.
    """

    if i1 <= i0:
        return None

    theta_mid = float(
        np.mean(
            theta[i0:i1 + 1]
        )
    )

    alpha_mid = float(
        np.mean(
            alpha[i0:i1 + 1]
        )
    )

    element = (
        build_element_from_endpoints(
            r_start=float(
                r[i0]
            ),
            z_start=float(
                z[i0]
            ),
            r_end=float(
                r[i1]
            ),
            z_end=float(
                z[i1]
            ),
            alpha_mid=alpha_mid,
            theta_mid=theta_mid,
            t_final=t_final,
            kind="constant",
        )
    )

    if element is None:
        return None

    element[
        "start_idx"
    ] = i0

    element[
        "end_idx"
    ] = i1

    element[
        "label"
    ] = (
        f"{theta_mid:.2f} deg "
        "(constant-angle region)"
    )

    return element


# ======================================================================
# 13. SOLVE VARYING REGION
# ======================================================================

def solve_monotonic_varying_region(
    r,
    z,
    theta,
    alpha,
    i0,
    i1,
    t_final,
    angle_step_deg=2.0,
    min_arc_length_factor=3.0,
):
    """
    Solve a monotonic varying-alpha region.

    Decision rule
    -------------
    If:

        region_arc_length
            <
        min_arc_length_factor * characteristic_spacing

    the region is treated as ONE physical transition element.

    Otherwise it is discretized using alpha bins.

    This prevents tiny corners/fillets from producing many
    nearly-degenerate independent thickness unknowns.
    """

    if i1 <= i0:
        return []

    if angle_step_deg <= 0:
        raise ValueError(
            "angle_step_deg must be positive."
        )

    if angle_step_deg > 180:
        raise ValueError(
            "angle_step_deg must be <= 180."
        )

    if min_arc_length_factor <= 0:
        raise ValueError(
            "min_arc_length_factor must be positive."
        )

    # --------------------------------------------------------------
    # Characteristic profile spacing
    # --------------------------------------------------------------

    characteristic_spacing = (
        profile_median_spacing(
            r,
            z,
        )
    )

    # --------------------------------------------------------------
    # Arc length of this region
    # --------------------------------------------------------------

    ds_region = np.hypot(
        np.diff(
            r[i0:i1 + 1]
        ),
        np.diff(
            z[i0:i1 + 1]
        ),
    )

    region_arc_length = float(
        np.sum(
            ds_region
        )
    )

    min_arc_length = (
        min_arc_length_factor
        * characteristic_spacing
    )

    # --------------------------------------------------------------
    # SHORT VARYING REGION
    # --------------------------------------------------------------

    if (
        region_arc_length
        < min_arc_length
    ):

        theta_mid = float(
            np.mean(
                theta[
                    i0:i1 + 1
                ]
            )
        )

        alpha_mid = float(
            np.mean(
                alpha[
                    i0:i1 + 1
                ]
            )
        )

        element = (
            build_element_from_endpoints(
                r_start=float(
                    r[i0]
                ),
                z_start=float(
                    z[i0]
                ),
                r_end=float(
                    r[i1]
                ),
                z_end=float(
                    z[i1]
                ),
                alpha_mid=alpha_mid,
                theta_mid=theta_mid,
                t_final=t_final,
                kind="short_varying",
            )
        )

        if element is None:
            return []

        element[
            "start_idx"
        ] = i0

        element[
            "end_idx"
        ] = i1

        element[
            "region_arc_length"
        ] = region_arc_length

        element[
            "characteristic_spacing"
        ] = characteristic_spacing

        element[
            "min_arc_length"
        ] = min_arc_length

        element[
            "arc_length_ratio"
        ] = (
            region_arc_length
            / characteristic_spacing
        )

        element[
            "label"
        ] = (
            f"{alpha_mid:.2f} deg "
            "(short varying transition)"
        )

        return [
            element
        ]

    # --------------------------------------------------------------
    # NORMAL VARYING REGION
    # --------------------------------------------------------------

    alpha_local = alpha[
        i0:i1 + 1
    ]

    alpha_min = max(
        0.0,
        float(
            np.min(
                alpha_local
            )
        ),
    )

    alpha_max = min(
        180.0,
        float(
            np.max(
                alpha_local
            )
        ),
    )

    # Number of possible bins over [0, 180]
    n_bins = int(
        np.ceil(
            180.0
            / angle_step_deg
        )
    )

    first_bin = int(
        np.floor(
            alpha_min
            / angle_step_deg
        )
    )

    last_bin = int(
        np.floor(
            alpha_max
            / angle_step_deg
        )
    )

    first_bin = int(
        np.clip(
            first_bin,
            0,
            n_bins - 1,
        )
    )

    last_bin = int(
        np.clip(
            last_bin,
            0,
            n_bins - 1,
        )
    )

    elements = []

    for bin_id in range(
        first_bin,
        last_bin + 1,
    ):

        bin_start = (
            bin_id
            * angle_step_deg
        )

        bin_end = min(
            (
                bin_id + 1
            )
            * angle_step_deg,
            180.0,
        )

        geometry = (
            extract_alpha_bin_geometry(
                r=r,
                z=z,
                alpha=alpha,
                start=i0,
                end=i1,
                bin_start=bin_start,
                bin_end=bin_end,
            )
        )

        if geometry is None:
            continue

        (
            r_bin,
            z_bin,
            alpha_bin,
        ) = geometry

        if len(
            r_bin
        ) < 2:
            continue

        r_start = float(
            r_bin[0]
        )

        z_start = float(
            z_bin[0]
        )

        r_end = float(
            r_bin[-1]
        )

        z_end = float(
            z_bin[-1]
        )

        alpha_mid = float(
            np.mean(
                alpha_bin
            )
        )

        # Compute local theta from the extracted geometry.
        theta_local = (
            compute_tangent_angles(
                r_bin,
                z_bin,
            )
        )

        theta_mid = float(
            np.mean(
                theta_local
            )
        )

        element = (
            build_element_from_endpoints(
                r_start=r_start,
                z_start=z_start,
                r_end=r_end,
                z_end=z_end,
                alpha_mid=alpha_mid,
                theta_mid=theta_mid,
                t_final=t_final,
                kind="varying",
                alpha_bin_index=bin_id,
                alpha_bin_start=float(
                    bin_start
                ),
                alpha_bin_end=float(
                    bin_end
                ),
            )
        )

        if element is None:
            continue

        element[
            "region_arc_length"
        ] = region_arc_length

        element[
            "characteristic_spacing"
        ] = characteristic_spacing

        element[
            "min_arc_length"
        ] = min_arc_length

        element[
            "arc_length_ratio"
        ] = (
            region_arc_length
            / characteristic_spacing
        )

        element[
            "start_idx"
        ] = i0

        element[
            "end_idx"
        ] = i1

        element[
            "label"
        ] = (
            f"{bin_start:.0f}-"
            f"{bin_end:.0f} deg"
        )

        elements.append(
            element
        )

    return elements


# ======================================================================
# 14. SOLVE ALL PHYSICAL ELEMENTS
# ======================================================================

def solve_physical_elements(
    r,
    z,
    theta,
    alpha,
    physical_regions,
    t_final,
    angle_step_deg=2.0,
    min_arc_length_factor=3.0,
):
    """
    Solve all physical regions.

    Constant region:
        one physical element.

    Short varying region:
        one physical element.

    Long varying region:
        alpha-binned physical elements.
    """

    elements = []

    for region in (
        physical_regions
    ):

        i0 = int(
            region["start"]
        )

        i1 = int(
            region["end"]
        )

        if (
            region["kind"]
            == "constant"
        ):

            element = (
                solve_constant_region(
                    r,
                    z,
                    theta,
                    alpha,
                    i0,
                    i1,
                    t_final,
                )
            )

            if element is not None:
                elements.append(
                    element
                )

            continue

        # ----------------------------------------------------------
        # Varying region:
        # first split at alpha turning points.
        # ----------------------------------------------------------

        monotonic_regions = (
            split_monotonic_alpha_regions(
                alpha,
                i0,
                i1,
            )
        )

        for mono in (
            monotonic_regions
        ):

            m0 = int(
                mono["start"]
            )

            m1 = int(
                mono["end"]
            )

            if m1 <= m0:
                continue

            sub_elements = (
                solve_monotonic_varying_region(
                    r=r,
                    z=z,
                    theta=theta,
                    alpha=alpha,
                    i0=m0,
                    i1=m1,
                    t_final=t_final,
                    angle_step_deg=angle_step_deg,
                    min_arc_length_factor=(
                        min_arc_length_factor
                    ),
                )
            )

            elements.extend(
                sub_elements
            )

    return elements


# ======================================================================
# 15. OUTPUT RESAMPLING
# ======================================================================

def resample_solution_for_output(
    elements,
    samples_per_constant_region=10,
):
    """
    Resample solved physical elements for visualization/output.

    IMPORTANT:
    This does NOT create new physical thickness unknowns.

    The thickness remains constant over each solved physical element.
    """

    if samples_per_constant_region < 2:
        raise ValueError(
            "samples_per_constant_region must be >= 2."
        )

    rows = []

    for element in elements:

        r0 = float(
            element["r_start"]
        )

        z0 = float(
            element["z_start"]
        )

        r1 = float(
            element["r_end"]
        )

        z1 = float(
            element["z_end"]
        )

        t_initial = float(
            element["t_initial"]
        )

        t_analytic = float(
            element["t_analytic"]
        )

        kind = element[
            "kind"
        ]

        if kind == "constant":
            n = max(
                2,
                int(
                    samples_per_constant_region
                ),
            )

        else:
            n = 2

        for j in range(n):

            # Prevent duplicate shared boundaries.
            if (
                rows
                and j == 0
            ):
                continue

            s = (
                j
                / (n - 1)
            )

            rj = (
                r0
                + s * (r1 - r0)
            )

            zj = (
                z0
                + s * (z1 - z0)
            )

            rows.append(
                {
                    "r": float(
                        rj
                    ),
                    "z": float(
                        zj
                    ),

                    "t_initial": float(
                        t_initial
                    ),

                    "t_analytic": float(
                        t_analytic
                    ),

                    "analytic_error_percent":
                        element.get(
                            "analytic_error_percent",
                            np.nan,
                        ),

                    "kind": kind,

                    "label": element.get(
                        "label",
                        "",
                    ),

                    "alpha_bin_index":
                        element.get(
                            "alpha_bin_index"
                        ),

                    "alpha_bin_start":
                        element.get(
                            "alpha_bin_start"
                        ),

                    "alpha_bin_end":
                        element.get(
                            "alpha_bin_end"
                        ),

                    "arc_length":
                        element.get(
                            "arc_length"
                        ),

                    "target_volume":
                        element.get(
                            "target_volume"
                        ),
                }
            )

    return rows


# ======================================================================
# 16. VALIDATION HELPERS
# ======================================================================

def validate_element_chain(
    elements,
    tolerance=1e-8,
):
    """
    Verify that consecutive physical elements share boundaries.

    Returns
    -------
    dict
        Validation information.
    """

    if len(elements) <= 1:
        return {
            "valid": True,
            "max_gap": 0.0,
            "num_boundaries": 0,
        }

    max_gap = 0.0
    gaps = []

    for i in range(
        len(elements) - 1
    ):

        current = elements[i]
        nxt = elements[i + 1]

        gap = float(
            np.hypot(
                nxt["r_start"]
                - current["r_end"],
                nxt["z_start"]
                - current["z_end"],
            )
        )

        gaps.append(
            gap
        )

        max_gap = max(
            max_gap,
            gap,
        )

    valid = (
        max_gap
        <= tolerance
    )

    return {
        "valid": bool(
            valid
        ),
        "max_gap": float(
            max_gap
        ),
        "num_boundaries": len(
            gaps
        ),
        "gaps": gaps,
    }


def compute_volume_summary(
    elements,
):
    """
    Summarize target formed volume across physical elements.

    Useful for checking:

        sum(V_element) ~= V_whole_profile
    """

    if not elements:
        return {
            "total_element_volume": 0.0,
            "num_elements": 0,
        }

    total_volume = float(
        np.sum(
            [
                e[
                    "target_volume"
                ]
                for e in elements
            ]
        )
    )

    return {
        "total_element_volume":
            total_volume,

        "num_elements":
            len(elements),
    }


# ======================================================================
# 17. MAIN SOLVER
# ======================================================================

def solve_thickness(
    r_profile,
    z_profile,
    t_final,
    angle_step_deg=2.0,
    const_tol_deg=1e-6,
    samples_per_constant_region=10,
    min_arc_length_factor=3.0,
):
    """
    Main thickness solver.

    Parameters
    ----------
    r_profile : array-like
        Axisymmetric radial profile coordinates.

    z_profile : array-like
        Axisymmetric axial profile coordinates.

    t_final : float
        Desired final formed thickness.

    angle_step_deg : float, default=2.0
        Alpha discretization for sufficiently long varying regions.

    const_tol_deg : float, default=1e-6
        Angular tolerance for constant-angle detection.

        For noisy tessellated CAD, this may need to be increased,
        e.g. 0.05 or 0.1 degrees, after examining the actual
        sampling noise.

    samples_per_constant_region : int, default=10
        Visualization/output samples for constant elements.

    min_arc_length_factor : float, default=3.0
        Short-transition threshold in multiples of median profile
        point spacing.

    Returns
    -------
    dict
        Complete geometry, region, element, and output information.
    """

    # --------------------------------------------------------------
    # Convert inputs
    # --------------------------------------------------------------

    r = np.asarray(
        r_profile,
        dtype=float,
    ).copy()

    z = np.asarray(
        z_profile,
        dtype=float,
    ).copy()

    # --------------------------------------------------------------
    # Validate input
    # --------------------------------------------------------------

    if r.ndim != 1:
        raise ValueError(
            "r_profile must be 1-D."
        )

    if z.ndim != 1:
        raise ValueError(
            "z_profile must be 1-D."
        )

    if len(r) != len(z):
        raise ValueError(
            "r_profile and z_profile must have the same length."
        )

    if len(r) < 2:
        raise ValueError(
            "At least two profile points are required."
        )

    if not np.all(
        np.isfinite(r)
    ):
        raise ValueError(
            "r_profile contains non-finite values."
        )

    if not np.all(
        np.isfinite(z)
    ):
        raise ValueError(
            "z_profile contains non-finite values."
        )

    if t_final <= 0:
        raise ValueError(
            "t_final must be positive."
        )

    if angle_step_deg <= 0:
        raise ValueError(
            "angle_step_deg must be positive."
        )

    if angle_step_deg > 180:
        raise ValueError(
            "angle_step_deg must be <= 180."
        )

    if const_tol_deg < 0:
        raise ValueError(
            "const_tol_deg cannot be negative."
        )

    if samples_per_constant_region < 2:
        raise ValueError(
            "samples_per_constant_region must be >= 2."
        )

    if min_arc_length_factor <= 0:
        raise ValueError(
            "min_arc_length_factor must be positive."
        )

    # --------------------------------------------------------------
    # Orient profile FIRST
    # --------------------------------------------------------------

    if z[-1] < z[0]:

        r = r[::-1]
        z = z[::-1]

    # --------------------------------------------------------------
    # Geometry -> theta
    # --------------------------------------------------------------

    theta = (
        compute_tangent_angles(
            r,
            z,
        )
    )

    # --------------------------------------------------------------
    # theta -> alpha
    # --------------------------------------------------------------

    alpha = (
        tangent_to_sine_law_angle(
            theta
        )
    )

    # --------------------------------------------------------------
    # Detect physical regions
    # --------------------------------------------------------------

    physical_regions = (
        detect_physical_regions(
            theta,
            const_tol_deg=const_tol_deg,
        )
    )

    # --------------------------------------------------------------
    # Solve physical elements
    # --------------------------------------------------------------

    elements = (
        solve_physical_elements(
            r=r,
            z=z,
            theta=theta,
            alpha=alpha,
            physical_regions=physical_regions,
            t_final=t_final,
            angle_step_deg=angle_step_deg,
            min_arc_length_factor=(
                min_arc_length_factor
            ),
        )
    )

    # --------------------------------------------------------------
    # Resample only for output
    # --------------------------------------------------------------

    rows = (
        resample_solution_for_output(
            elements,
            samples_per_constant_region=(
                samples_per_constant_region
            ),
        )
    )

    # --------------------------------------------------------------
    # Validation information
    # --------------------------------------------------------------

    chain_validation = (
        validate_element_chain(
            elements
        )
    )

    volume_summary = (
        compute_volume_summary(
            elements
        )
    )

    # --------------------------------------------------------------
    # Return
    # --------------------------------------------------------------

    return {
        "r_profile": r,
        "z_profile": z,

        "theta": theta,
        "alpha": alpha,

        "physical_regions":
            physical_regions,

        "elements":
            elements,

        "rows":
            rows,

        "chain_validation":
            chain_validation,

        "volume_summary":
            volume_summary,

        "parameters": {
            "angle_step_deg":
                float(
                    angle_step_deg
                ),

            "const_tol_deg":
                float(
                    const_tol_deg
                ),

            "samples_per_constant_region":
                int(
                    samples_per_constant_region
                ),

            "min_arc_length_factor":
                float(
                    min_arc_length_factor
                ),
        },
    }