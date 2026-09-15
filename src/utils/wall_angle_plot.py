"""
Diagnostic plot for ISF wall angle alpha versus radius r.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_wall_angle_vs_radius(
    r_profile,
    alpha_profile,
    save_path=None,
):
    """
    Plot wall angle alpha versus radius r.

    Parameters
    ----------
    r_profile : array-like
        Radius values [mm].

    alpha_profile : array-like
        Wall angle alpha values [degrees].

    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    r = np.asarray(
        r_profile,
        dtype=float,
    )

    alpha = np.asarray(
        alpha_profile,
        dtype=float,
    )

    if len(r) != len(alpha):
        raise ValueError(
            "r_profile and alpha_profile must have "
            "the same length."
        )

    valid = (
        np.isfinite(r)
        & np.isfinite(alpha)
    )

    r = r[valid]
    alpha = alpha[valid]

    if len(r) < 2:
        raise ValueError(
            "Not enough valid points to generate "
            "wall-angle diagnostic plot."
        )

    # Sort by radius so the curve is visually ordered.
    order = np.argsort(r)

    r = r[order]
    alpha = alpha[order]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        r,
        alpha,
        linewidth=1.8,
    )

    ax.scatter(
        r,
        alpha,
        s=12,
    )

    ax.set_xlabel(
        "Radius r (mm)"
    )

    ax.set_ylabel(
        "Wall angle α (degrees)"
    )

    ax.set_title(
        "Wall Angle α vs Radius r"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    return fig