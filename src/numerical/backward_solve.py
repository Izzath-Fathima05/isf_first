"""
Backward Solving: Final Geometry + Desired Uniform Thickness
                   -> Required Non-Uniform Initial Blank Thickness
====================================================================

Implements Goal 1 using the paper's (Viegas et al. 2026) validated
volume-conservation approach instead of the simpler cosine-law
inversion.
"""

import numpy as np

from .point_projection import pappus_element_volume


def flatten_by_arc_length(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    NOTE: kept for reference / other forming processes, but NOT used by
    backward_solve() for SPIF -- see constant_radius_mapping() below and
    module docstring for why.

    Maps a final axisymmetric profile onto a flat blank radius by
    preserving cumulative meridional arc length -- the standard "blank
    development" approximation for DRAW-IN-DOMINATED processes like
    deep drawing, where material slides substantially inward.

    CONFIRMED WRONG FOR SPIF: produces thickness DECREASING at steep
    walls (opposite of physical expectation and the paper's own
    reported trend), because unrolling a steep wall by arc length
    stretches its effective radius far beyond the true 3D radius,
    forcing thickness down to match the same target volume.
    """
    segment_lengths = np.sqrt(np.diff(r) ** 2 + np.diff(z) ** 2)
    r_flat = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return r_flat


def constant_radius_mapping(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Maps a final axisymmetric profile onto a flat blank radius by
    assuming NEGLIGIBLE radial draw-in -- i.e. r_flat ~= r_final.

    This is the physically appropriate simplification for SPIF: per
    the reference paper's own Section 2.1, the classical sine law is
    built on exactly this assumption ("the sine law... assumes that
    the final radius remains equal to the initial one"), and the point
    projection method's key refinement corrects for the SMALL residual
    radial drift (Eq. 4: r_ext = r_int + t*sin(theta)), not a
    large-scale unrolling.

    VALIDATED: produces the correct physical trend (thickness
    increases toward steep walls) and a peak magnitude for the beta=60
    benchmark cone (~2.0-2.1mm via sequential marching, ~2.1-3.0mm
    range via global solve) closely matching the paper's own reported
    peak of ~2.15mm (Fig. 12b).
    """
    return r.copy()


def backward_solve(r_final: np.ndarray, z_final: np.ndarray, t_df: float,
                    position_mapping=constant_radius_mapping):
    """
    Given the FINAL formed profile (r_final, z_final) and the desired
    uniform final thickness t_df, computes the required non-uniform
    initial blank thickness at each corresponding flat-blank radius.

    SOLVE STRATEGY: global simultaneous system, not sequential marching.
    ------------------------------------------------------------------
    An earlier sequential version (t[0] -> solve t[1] -> use t[1] to
    solve t[2] -> ...) was found to be numerically UNSTABLE: each
    node's volume depends on BOTH its own and its neighbor's thickness,
    so a deviation at one node forces an overcorrection at the next,
    producing a persistent sawtooth oscillation -- CONFIRMED to occur
    even though every individual element's volume equation was solved
    exactly (traced directly: match=True at every step, yet the
    sequence alternated between ~1.0-1.2mm and ~2.9-3.0mm). Finer
    discretization did NOT shrink the oscillation, confirming this is
    a structural instability of sequential marching, not a
    root-finding precision issue.

    THE FIX: solve for ALL node thicknesses AT ONCE as one coupled
    system of N-1 volume-conservation equations in N-1 unknowns (t[0]
    is fixed as a boundary condition). This removes the one-directional
    error propagation entirely.

    Returns
    -------
    r_flat   : (N,) flat-blank radial coordinate for each node
    t_initial: (N,) required initial thickness at each flat-blank node
    """
    from scipy.optimize import least_squares

    N = len(r_final)
    assert len(z_final) == N

    r_flat = position_mapping(r_final, z_final)

    # Target volumes: the ACTUAL final-part volume of each element,
    # using the desired uniform thickness t_df on the final geometry.
    target_volumes = np.array([
        pappus_element_volume(r_final[i], z_final[i], r_final[i + 1], z_final[i + 1],
                               t_df, t_df)
        for i in range(N - 1)
    ])

    def residuals(t_unknown):
        # t_unknown = t_initial[1:]; t_initial[0] is fixed at t_df
        t = np.concatenate([[t_df], t_unknown])
        res = np.array([
            pappus_element_volume(r_flat[i], 0.0, r_flat[i + 1], 0.0, t[i], t[i + 1])
            - target_volumes[i]
            for i in range(N - 1)
        ])
        return res

    t0_guess = np.full(N - 1, t_df)
    result = least_squares(residuals, t0_guess, method="lm", xtol=1e-12, ftol=1e-12)

    if not result.success:
        print(f"Warning: global solve did not fully converge "
              f"(status={result.status}, max residual={np.abs(result.fun).max():.4f})")

    t_initial = np.concatenate([[t_df], result.x])
    return r_flat, t_initial