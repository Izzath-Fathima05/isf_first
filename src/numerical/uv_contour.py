"""
UV Parametrization + 2D Contour Map
=====================================

Unrolls the 3D axisymmetric surface onto a flat (u,v) plane and
interpolates thickness onto a regular grid for a true 2D contour plot
-- the actual "UV thickness map" format, as opposed to a 3D colored mesh.

Assumption: the vessel is (approximately) a body of revolution. For
such shapes: u = height along the axis, v = angle around the axis.
"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def compute_uv_parametrization(vertices: np.ndarray,
                                axis: np.ndarray = np.array([0, 0, 1.0]),
                                axis_origin: np.ndarray = np.array([0, 0, 0])):
    """
    Unrolls 3D surface points onto a 2D (u, v) plane, assuming
    (approximate) rotational symmetry about `axis`.

        u = position along the axis (height)
        v = angle around the axis (radians, -pi to pi)

    Returns u, v, radius -- radius lets you sanity-check the
    axisymmetry assumption (should be ~constant at fixed u).
    """
    axis = axis / np.linalg.norm(axis)
    rel = vertices - axis_origin

    u = rel @ axis
    perp = rel - np.outer(u, axis)
    radius = np.linalg.norm(perp, axis=1)

    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - (ref @ axis) * axis
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    x_local = perp @ e1
    y_local = perp @ e2
    v = np.arctan2(y_local, x_local)

    return u, v, radius


def plot_uv_contour(u: np.ndarray, v: np.ndarray, T0: np.ndarray,
                     grid_res: int = 200, title: str = "Initial Sheet Thickness T\u2080(u,v)"):
    """
    Interpolates scattered (u, v, T0) samples onto a regular grid and
    draws a true 2D filled contour plot.
    """
    u_grid = np.linspace(u.min(), u.max(), grid_res)
    v_grid = np.linspace(v.min(), v.max(), grid_res)
    UU, VV = np.meshgrid(u_grid, v_grid)

    TT = griddata((u, v), T0, (UU, VV), method="linear")

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(np.degrees(VV), UU, TT, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="T\u2080 (mm)")
    ax.set_xlabel("v \u2014 angle around axis (deg)")
    ax.set_ylabel("u \u2014 height along axis (mm)")
    ax.set_title(title)
    plt.tight_layout()
    return fig