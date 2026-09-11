"""
Alpha-bin thickness solver -- reporting layer over the validated
Pappus volume-conservation solve.

WHAT CHANGED FROM THE PREVIOUS VERSION
---------------------------------------
The previous version never called Pappus / backward_solve at all: the
"spherical forming surface" branch computed t0 = tf*ds/|dr|, which is
algebraically just the closed-form sine law (tf/cos(beta)), and the
generic fallback branch had the angle conversion inverted on top of
that. Confirmed from your actual run: every printed T0 matched
1/cos(theta) to 4 decimal places for a genuinely curved sphere, which
should not happen once volume conservation is doing real work.

This version has ONE physics path: backward_solve() (Pappus + global
least-squares, from backward_solve.py), for ANY axisymmetric profile,
sphere or not -- no special-cased sphere formula needed. Angle helpers
below are for labeling/reporting ONLY; they never feed the thickness
math.

ANGLE CONVENTION (fixed)
-------------------------
compute_tangent_angles() returns the angle from VERTICAL -- this
already IS the paper's original sine-law angle (Eq.1: tf=t0*sin(alpha)),
directly usable in sin(), with no extra conversion. It is SMALL near a
vertical wall (rim of a bowl) and LARGE near a flat wall (pole/base).

angle_from_horizontal() is its complement (90 - that), kept only for
the printed "Theta" column, matching the convention your own working
sphere branch already used (theta_mid = 90 - alpha_mid): LARGE near a
vertical wall, SMALL near a flat wall.

Do not re-derive a "sine law angle" from theta via another 90-x flip
anywhere downstream -- that flip is exactly the bug this fixes.
"""

import numpy as np

from .backward_solve import backward_solve, constant_radius_mapping

EPS = 1.0e-12


def compute_tangent_angles(r_profile, z_profile):
    """
    Angle from VERTICAL, in degrees, clipped to [0, 90].

    This is the paper's original sine-law angle alpha -- use it
    directly in sin(alpha) for any classical-sine-law comparison.
    Small near a vertical wall, large near a flat wall.
    """
    r = np.asarray(r_profile, dtype=float)
    z = np.asarray(z_profile, dtype=float)
    dr = np.gradient(r)
    dz = np.gradient(z)
    alpha_from_vertical = np.degrees(np.arctan2(np.abs(dr), np.abs(dz) + EPS))
    return np.clip(alpha_from_vertical, 0.0, 90.0)


def angle_from_horizontal(alpha_from_vertical_deg):
    """
    Display-only complement of compute_tangent_angles(): the paper's
    beta convention (Fig. 3b/8c), angle from horizontal. NEVER feed
    this into sin() for the sine law -- that's the inversion bug from
    the previous version. Large near a vertical wall, small near flat.
    """
    return 90.0 - np.asarray(alpha_from_vertical_deg, dtype=float)


def solve_thickness(r_profile, z_profile, t_final, angle_step_deg=2.0, **_unused_kwargs):
    """
    Physics: backward_solve() on the full profile (Pappus volume
    conservation + global least-squares -- see backward_solve.py for
    why sequential marching was rejected).

    Reporting: group the per-node result into angle_step_deg-wide
    labels for the existing table/heatmap, one "element" per profile
    SEGMENT (matches backward_solve's N-1 elements exactly, no
    resampling or rebinning of the physics).

    **_unused_kwargs absorbs const_tol_deg / samples_per_constant_region
    if still passed from an older call site -- no-ops here, since
    there's no separate constant-region collapsing step: backward_solve
    already solves every segment on the actual profile resolution, and
    a straight/flat run naturally gets a near-uniform t_initial across
    its segments without needing special-casing.

    Returns
    -------
    elements : list[dict]  keys: index, alpha_start, alpha_end,
               alpha_mid, theta_mid, r_start, z_start, r_end, z_end,
               t_initial  -- same shape run_pipeline.py already expects
    rows     : (N-1, 5) ndarray  columns: r_mid, z_mid, t0, theta_mid,
               alpha_mid
    """
    r_profile = np.asarray(r_profile, dtype=float)
    z_profile = np.asarray(z_profile, dtype=float)

    if t_final <= 0:
        raise ValueError("t_final must be positive.")
    if angle_step_deg <= 0:
        raise ValueError("angle_step_deg must be positive.")
    if len(r_profile) != len(z_profile):
        raise ValueError("r_profile and z_profile must have the same length.")
    if len(r_profile) < 2:
        raise ValueError("At least two profile points are required.")

    r_flat, t_initial = backward_solve(
        r_profile, z_profile, t_final, position_mapping=constant_radius_mapping
    )

    alpha = compute_tangent_angles(r_profile, z_profile)   # sine-law ready, from vertical
    theta = angle_from_horizontal(alpha)                    # display only, from horizontal

    elements = []
    rows = []
    for i in range(len(r_profile) - 1):
        t0 = 0.5 * (t_initial[i] + t_initial[i + 1])
        alpha_mid = 0.5 * (alpha[i] + alpha[i + 1])
        theta_mid = 0.5 * (theta[i] + theta[i + 1])
        alpha_start = angle_step_deg * np.floor(alpha_mid / angle_step_deg)
        alpha_end = alpha_start + angle_step_deg

        elements.append({
            "index": i,
            "alpha_start": float(alpha_start),
            "alpha_end": float(alpha_end),
            "alpha_mid": float(alpha_mid),
            "theta_mid": float(theta_mid),
            "r_start": float(r_profile[i]), "z_start": float(z_profile[i]),
            "r_end": float(r_profile[i + 1]), "z_end": float(z_profile[i + 1]),
            "t_initial": float(t0),
        })

        r_mid = 0.5 * (r_profile[i] + r_profile[i + 1])
        z_mid = 0.5 * (z_profile[i] + z_profile[i + 1])
        rows.append((r_mid, z_mid, t0, theta_mid, alpha_mid))

    return elements, np.asarray(rows, dtype=float)


# Kept as a name for backward compatibility with anything that still
# imports it, but it is now the identity in every meaningful sense --
# see the module docstring for why the old 90-x conversion was wrong.
def tangent_to_sine_law_angle(theta_deg):
    """DEPRECATED alias. `compute_tangent_angles()`'s output is already
    the sine-law angle -- pass it straight to sin(), no conversion."""
    return np.asarray(theta_deg, dtype=float)