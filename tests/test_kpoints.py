"""Tests for k-point generation."""

import jax.numpy as jnp
import numpy as np
import pytest

from pwdft.kpoints import monkhorst_pack, gamma_point, reduce_kpoints_time_reversal
from pwdft.lattice import reciprocal_lattice
from pwdft.constants import TWO_PI


def test_gamma_point():
    """Test Gamma point generation."""
    b = jnp.eye(3) * TWO_PI / 10.0
    kpts, wts = gamma_point(b)
    assert kpts.shape == (1, 3)
    np.testing.assert_allclose(kpts[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(wts[0], 1.0)


def test_mp_grid_size():
    """Test that MP grid has correct number of points."""
    b = jnp.eye(3) * TWO_PI / 10.0
    nk = (3, 3, 3)
    kpts, wts = monkhorst_pack(nk, b)
    assert kpts.shape == (27, 3)
    np.testing.assert_allclose(jnp.sum(wts), 1.0, atol=1e-12)


def test_mp_grid_inversion():
    """Test that unshifted MP grid has inversion symmetry."""
    b = jnp.eye(3) * TWO_PI / 10.0
    nk = (4, 4, 4)
    kpts, wts = monkhorst_pack(nk, b)

    # For each k, -k should also be in the set (modulo G)
    b_inv = jnp.linalg.inv(b)
    frac = np.array(kpts @ b_inv)
    for i in range(len(frac)):
        neg_k = -frac[i]
        diff = neg_k[None, :] - frac
        diff = diff - np.round(diff)
        min_dist = np.min(np.linalg.norm(diff, axis=-1))
        assert min_dist < 1e-10, f"k-point {i} has no inversion partner"


def test_mp_weights_uniform():
    """Test that MP weights are uniform."""
    b = jnp.eye(3) * TWO_PI / 10.0
    kpts, wts = monkhorst_pack((4, 4, 4), b)
    expected_weight = 1.0 / 64
    np.testing.assert_allclose(wts, expected_weight, atol=1e-12)


def test_time_reversal_reduction():
    """Test time-reversal reduction of k-points."""
    b = jnp.eye(3) * TWO_PI / 10.0
    kpts, wts = monkhorst_pack((4, 4, 4), b)
    kpts_red, wts_red = reduce_kpoints_time_reversal(kpts, wts, b)
    # Should have fewer k-points but same total weight
    assert len(kpts_red) <= len(kpts)
    np.testing.assert_allclose(jnp.sum(wts_red), 1.0, atol=1e-12)
