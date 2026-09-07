"""
Tests for patch extraction and segmentation
(src/numerical/patch_extraction.py, src/cad_native/cad_native_patches.py).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from numerical.patch_extraction import segment_into_patches, check_axisymmetry
from cad_native.cad_native_patches import make_piecewise_linear_profile, build_revolved_shell, split_into_patches


def test_patch_arc_lengths_sum_to_total():
    """Segmenting a profile into N patches should not lose or duplicate
    any arc length."""
    z = np.linspace(0, 25, 30)
    r = 20 + 10 * (z / 25) ** 1.5
    patches = segment_into_patches(z, r, n_patches=5)
    total = sum(p["arc_length"] for p in patches)
    full_length = np.sum(np.sqrt(np.diff(r) ** 2 + np.diff(z) ** 2))
    assert total == pytest.approx(full_length, rel=1e-6)


def test_patch_count_matches_request():
    z = np.linspace(0, 25, 30)
    r = 20 + 10 * (z / 25)
    patches = segment_into_patches(z, r, n_patches=7)
    assert len(patches) == 7


def test_synthetic_frustum_splits_into_requested_patch_count():
    """The CAD-native split should produce exactly N patch faces for
    an N-segment piecewise-linear profile."""
    r, z = make_piecewise_linear_profile(n_segments=5)
    shell = build_revolved_shell(r, z)
    patches = split_into_patches(shell, z)
    assert len(patches) == 5


def test_axisymmetric_mesh_passes_check():
    """A synthetic, genuinely axisymmetric point set should pass the
    axisymmetry diagnostic."""
    import trimesh
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    z_vals = np.linspace(0, 10, 10)
    verts = np.array([[10 * np.cos(t), 10 * np.sin(t), z] for z in z_vals for t in theta])
    faces = []  # not needed for this check, only vertices are used
    mesh = trimesh.Trimesh(vertices=verts, faces=np.zeros((0, 3), dtype=int), process=False)
    result = check_axisymmetry(mesh)
    assert result["likely_axisymmetric"]