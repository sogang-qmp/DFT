"""Tests for lattice utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from pwdft.lattice import (
    reciprocal_lattice, cell_volume, metric_tensor,
    fractional_to_cartesian, cartesian_to_fractional,
)
from pwdft.constants import TWO_PI


def test_reciprocal_lattice_cubic():
    """Test reciprocal lattice of a simple cubic cell."""
    a_lat = 5.0  # Bohr
    a = jnp.eye(3) * a_lat
    b = reciprocal_lattice(a)
    # b should be (2*pi/a) * I
    expected = jnp.eye(3) * TWO_PI / a_lat
    np.testing.assert_allclose(b, expected, atol=1e-12)


def test_reciprocal_lattice_orthogonality():
    """Test a_i . b_j = 2*pi * delta_ij."""
    a = jnp.array([[5.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0],
                    [0.0, 0.0, 7.0]])
    b = reciprocal_lattice(a)
    product = a @ b.T
    np.testing.assert_allclose(product, TWO_PI * jnp.eye(3), atol=1e-12)


def test_reciprocal_lattice_fcc():
    """Test reciprocal lattice of an FCC cell."""
    a0 = 5.0
    a = jnp.array([[0, a0/2, a0/2],
                    [a0/2, 0, a0/2],
                    [a0/2, a0/2, 0]], dtype=jnp.float64)
    b = reciprocal_lattice(a)
    product = a @ b.T
    np.testing.assert_allclose(product, TWO_PI * jnp.eye(3), atol=1e-10)


def test_cell_volume_cubic():
    """Test cell volume for cubic cell."""
    a = jnp.eye(3) * 10.0
    vol = cell_volume(a)
    np.testing.assert_allclose(vol, 1000.0, atol=1e-10)


def test_cell_volume_fcc():
    """Test cell volume for FCC cell."""
    a0 = 10.0
    a = jnp.array([[0, a0/2, a0/2],
                    [a0/2, 0, a0/2],
                    [a0/2, a0/2, 0]], dtype=jnp.float64)
    vol = cell_volume(a)
    np.testing.assert_allclose(vol, a0**3 / 4.0, atol=1e-10)


def test_fractional_roundtrip():
    """Test fractional<->Cartesian conversion roundtrip."""
    a = jnp.array([[5.0, 0.0, 0.0],
                    [1.0, 6.0, 0.0],
                    [0.5, 0.5, 7.0]])
    frac = jnp.array([[0.1, 0.2, 0.3],
                       [0.5, 0.5, 0.5]])
    cart = fractional_to_cartesian(frac, a)
    frac_back = cartesian_to_fractional(cart, a)
    np.testing.assert_allclose(frac_back, frac, atol=1e-12)
