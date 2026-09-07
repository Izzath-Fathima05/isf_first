"""
Tests for the flattening/unrolling math
(src/cad_native/cad_native_patches.py, src/cad_native/general_flatten.py,
 src/utils/geometry_utils.py).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from utils.geometry_utils import cone_unroll_params, cylinder_unroll_params


def test_cone_unroll_sector_angle_bounds():
    """A cone's unrolled sector angle must be strictly between 0 and
    2*pi (a full cylinder-like limit approaches 2*pi as half_angle->90deg,
    a very sharp cone approaches 0 as half_angle->0deg)."""
    for half_angle_deg in [10, 30, 45, 60, 80]:
        params = cone_unroll_params(np.radians(half_angle_deg), slant_inner=10, slant_outer=20)
        assert 0 < params["sector_angle_rad"] < 2 * np.pi


def test_cone_unroll_matches_manual_formula():
    half_angle = np.radians(41.0)
    params = cone_unroll_params(half_angle, slant_inner=15, slant_outer=25)
    expected = 2 * np.pi * np.sin(half_angle)
    assert params["sector_angle_rad"] == pytest.approx(expected)


def test_cylinder_unroll_dimensions():
    """A cylinder unrolled over a full 2*pi revolution and height h
    should produce a rectangle of width 2*pi*R and height h."""
    R, h = 10.0, 5.0
    params = cylinder_unroll_params(R, u_range=(0, 2 * np.pi), v_range=(0, h))
    assert params["width"] == pytest.approx(2 * np.pi * R)
    assert params["height"] == pytest.approx(h)


def test_cylinder_unroll_zero_height_gives_zero_area():
    params = cylinder_unroll_params(10.0, u_range=(0, np.pi), v_range=(3, 3))
    assert params["height"] == 0.0