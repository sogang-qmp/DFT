"""Tests for pseudopotential module."""

import jax.numpy as jnp
import numpy as np
import pytest

from pwdft.pseudopotential import (
    get_hgh_params, vloc_reciprocal, projector_fourier,
    real_spherical_harmonics,
)
from pwdft.constants import FOUR_PI, PI


def test_get_hgh_params():
    """Test that HGH parameters are available for common elements."""
    for symbol in ["H", "He", "C", "N", "O", "Si", "Al"]:
        params = get_hgh_params(symbol)
        assert params.Z_ion > 0
        assert params.r_loc > 0
        assert params.symbol == symbol


def test_get_hgh_params_unknown():
    """Test that unknown element raises error."""
    with pytest.raises(ValueError, match="not available"):
        get_hgh_params("Unobtainium")


def test_vloc_g0_limit():
    """Test local potential at G=0 is finite."""
    params = get_hgh_params("Si")
    omega = 270.0  # Approximate Si cell volume in Bohr^3
    g2 = jnp.array([0.0])
    vloc = vloc_reciprocal(g2, params, omega)
    assert jnp.isfinite(vloc[0])


def test_vloc_decay():
    """Test that V_loc decays for large G."""
    params = get_hgh_params("Si")
    omega = 270.0
    g2 = jnp.array([1.0, 10.0, 100.0, 1000.0])
    vloc = vloc_reciprocal(g2, params, omega)
    # Should decay due to Gaussian factor
    assert jnp.abs(vloc[-1]) < jnp.abs(vloc[0])


def test_projector_normalization():
    """Test that projectors have correct behavior at G=0."""
    r_l = 0.4
    g_zero = jnp.array([[0.0, 0.0, 0.0]])

    # l=0, i=0 projector should be nonzero at G=0
    p00 = projector_fourier(g_zero, 0, 0, r_l)
    assert float(jnp.abs(p00[0])) > 0

    # l=1 projectors should be zero at G=0 (proportional to |G|)
    p10 = projector_fourier(g_zero, 1, 0, r_l)
    assert float(jnp.abs(p10[0])) < 1e-10


def test_spherical_harmonics_orthogonality():
    """Test that spherical harmonics satisfy normalization on unit sphere."""
    # Generate random unit vectors
    key = __import__('jax').random.PRNGKey(42)
    n = 10000
    vecs = __import__('jax').random.normal(key, (n, 3))
    vecs = vecs / jnp.linalg.norm(vecs, axis=-1, keepdims=True)

    # Y_00 should be constant
    y00 = real_spherical_harmonics(vecs, 0)  # (1, n)
    expected = 0.5 / jnp.sqrt(PI)
    np.testing.assert_allclose(y00[0], expected, atol=1e-5)

    # l=1 harmonics should be orthogonal (approximately, via MC integration)
    y1 = real_spherical_harmonics(vecs, 1)  # (3, n)
    # <Y1m|Y1m'> ≈ delta_mm' * 4*pi / (2*1+1) / (4*pi) = 1/3 for same m
    # The integral of Y_lm * Y_l'm' over unit sphere = delta_{ll'} delta_{mm'}
    for m in range(3):
        integral = 4.0 * PI * jnp.mean(y1[m]**2)
        np.testing.assert_allclose(float(integral), 1.0, atol=0.1)  # MC approximation
