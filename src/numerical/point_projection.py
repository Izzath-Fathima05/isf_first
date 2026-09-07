"""
Point Projection Method — Pappus Centroid Volume + Bisection Thickness Solver
================================================================================

Direct implementation of Algorithm 1 and Algorithm 2 from:

    Viegas, J.M.A., Sampaio, R.F.V., Pragana, J.P.M., Braganca, I.M.F.,
    Silva, C.M.A., Martins, P.A.F. "Hybridization of single point
    incremental forming and machining to produce constant wall-thickness
    parts." Int. J. of Lightweight Materials and Manufacture (2026).
    https://doi.org/10.1016/j.ijlmm.2026.06.003

SCOPING NOTE
-----------------------
The paper's pseudocode (Algorithms 1 and 2) is reproduced here as
faithfully as the published pseudocode allows. Two details are NOT
fully specified in the paper text and required a documented, defensible
choice:

1. The exact construction of auxiliary points C and D (the
   "quadrilateral ABCD" for the Pappus centroid calculation) is
   implemented as: C and D are A and B offset along the LOCAL NORMAL
   to segment AB by the element's thickness -- matching the thin
   trapezoidal shell cross-section shown schematically in the paper's
   Fig. 3c/4.

2. The FULL incremental point-projection kinematic simulation (Fig.
   3a-b) is a larger undertaking than the two algorithms alone. This
   module implements the volume-conservation core (fully specified)
   plus a single-pass backward-solving procedure, NOT the full
   multi-stage incremental simulation of Section 2.2's software
   workflow (Fig. 6).
"""

import numpy as np


# ---------------------------------------------------------------------
# Algorithm 1: Pappus_CMethod
# ---------------------------------------------------------------------
def pappus_element_volume(rA: float, zA: float, rB: float, zB: float,
                           tA: float, tB: float) -> float:
    """
    Computes the volume of one revolved (axisymmetric) shell element
    bounded by profile points A=(rA,zA), B=(rB,zB), with thickness tA
    at A and tB at B, using Pappus's centroid theorem: V = 2*pi*A*r_bar.

    VALIDATED: for a vertical wall (rA=rB=R), this reduces EXACTLY to
    the closed-form hollow-cylinder volume pi*h*t*(2R+t). Confirmed to
    machine precision in validate_point_projection.py.
    """
    # --- CALCULATE AlphaBetweenAB ---
    alpha = np.arctan2(zB - zA, rB - rA)

    # local outward normal (perpendicular to AB, pointing toward
    # increasing r -- i.e. away from the axis of revolution)
    normal = np.array([np.sin(alpha), -np.cos(alpha)])

    A = np.array([rA, zA])
    B = np.array([rB, zB])

    # --- CALCULATE CoordinatesC and CoordinatesD ---
    D = A + tA * normal
    C = B + tB * normal

    # --- CALCULATE AreaQuadrilateralABCD --- (shoelace formula, order A-B-C-D)
    pts = np.array([A, B, C, D])
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # --- CALCULATE CoordinatesCentroid --- (proper polygon centroid, not
    # a naive vertex average -- matters when tA != tB)
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * np.sum(cross)
    if abs(signed_area) < 1e-12:
        cx, cy = x.mean(), y.mean()
    else:
        cx = np.sum((x + x_next) * cross) / (6 * signed_area)
        cy = np.sum((y + y_next) * cross) / (6 * signed_area)

    # --- CALCULATE DistanceBetweenCentroidAndAxis ---
    r_bar = abs(cx)

    # --- CALCULATE ElementVolume (equation 3) ---
    volume = 2 * np.pi * area * r_bar
    return volume


# ---------------------------------------------------------------------
# Algorithm 2: BissectionMethod (paper's original fixed-factor version)
# ---------------------------------------------------------------------
def bisection_thickness(rA: float, zA: float, rB: float, zB: float,
                         tA: float, target_volume: float,
                         tolerance: float = 1e-3,
                         max_iter: int = 100) -> float:
    """
    Solves for the unknown thickness tB at point B such that the
    element volume matches target_volume, given known thickness tA
    at point A. Bisection search with adaptive bracket expansion,
    following Algorithm 2's pseudocode AS PUBLISHED (fixed 0.75/1.25
    bracket-widening factors).

    KNOWN LIMITATION (found via testing on a marched sequence of many
    elements, not a single isolated element): this fixed-factor
    bracket expansion does not always guarantee the root is bracketed
    before bisecting, which caused a sawtooth oscillation artifact
    when marching sequentially (error from one step propagates into
    the next step's tA). Kept here as a faithful reproduction of the
    published pseudocode; use `robust_bisection_thickness()` and, for
    multi-element problems, the GLOBAL solve in backward_solve.py.
    """
    t_guess = tA
    vol_guess = pappus_element_volume(rA, zA, rB, zB, tA, t_guess)
    if abs(vol_guess - target_volume) < tolerance:
        return t_guess

    guess_min, guess_max = t_guess, t_guess
    if vol_guess > target_volume:
        guess_min = guess_max * 0.75
    else:
        guess_max = guess_min * 1.25

    t_guess = (guess_max + guess_min) / 2
    vol_guess = pappus_element_volume(rA, zA, rB, zB, tA, t_guess)

    n_iter = 0
    while abs(vol_guess - target_volume) > tolerance and n_iter < max_iter:
        if vol_guess < target_volume:
            guess_min = t_guess
            t_guess = (guess_max + t_guess) / 2
        else:
            guess_max = t_guess
            t_guess = (t_guess + guess_min) / 2
        vol_guess = pappus_element_volume(rA, zA, rB, zB, tA, t_guess)
        n_iter += 1

    if n_iter >= max_iter:
        print(f"Warning: bisection did not converge within {max_iter} "
              f"iterations (residual = {abs(vol_guess - target_volume):.6f})")

    return t_guess


# ---------------------------------------------------------------------
# Robust replacement: guaranteed-bracket root-finding (recommended)
# ---------------------------------------------------------------------
def robust_bisection_thickness(rA: float, zA: float, rB: float, zB: float,
                                tA: float, target_volume: float,
                                tolerance: float = 1e-6) -> float:
    """
    Same problem as bisection_thickness() above, but with a bracket
    that is EXPANDED UNTIL A SIGN CHANGE IS CONFIRMED before calling
    scipy's brentq -- guaranteeing convergence for a single element.
    (For multi-element sequences, still prefer the global solve in
    backward_solve.py -- see its docstring for why.)
    """
    from scipy.optimize import brentq

    def residual(tB):
        return pappus_element_volume(rA, zA, rB, zB, tA, tB) - target_volume

    lo, hi = max(tA * 0.1, 1e-6), tA * 2.0
    r_lo, r_hi = residual(lo), residual(hi)
    expand_iter = 0
    while r_lo * r_hi > 0 and expand_iter < 50:
        lo *= 0.5
        hi *= 2.0
        r_lo, r_hi = residual(lo), residual(hi)
        expand_iter += 1

    if r_lo * r_hi > 0:
        raise RuntimeError(
            f"Could not bracket root after {expand_iter} expansions "
            f"(rA={rA}, zA={zA}, rB={rB}, zB={zB}, tA={tA}, "
            f"target_volume={target_volume}) -- check inputs for a "
            f"degenerate (near-zero-length or near-axis) element."
        )

    return brentq(residual, lo, hi, xtol=tolerance)