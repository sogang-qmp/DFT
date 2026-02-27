"""Tests for plane-wave basis set generation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pwdft.basis import (
    get_g_vectors, pw_to_real, real_to_pw, _next_fft_size,
    get_g_vectors_fft,
)
from pwdft.lattice import reciprocal_lattice


def test_next_fft_size():
    """Test FFT size selection."""
    assert _next_fft_size(1) == 1
    assert _next_fft_size(7) == 8
    assert _next_fft_size(11) == 12
    assert _next_fft_size(16) == 16
    assert _next_fft_size(17) == 18


def test_g_vectors_cubic():
    """Test G-vector generation for cubic cell."""
    a = jnp.eye(3) * 10.0  # 10 Bohr cubic cell
    ecut = 5.0  # Hartree

    g_vecs, g_idx, fft_grid = get_g_vectors(a, ecut)

    # All G-vectors should satisfy |G|^2/2 <= ecut
    g2 = jnp.sum(g_vecs**2, axis=-1)
    assert jnp.all(g2 <= 2.0 * ecut * 1.01)

    # G=0 should be present
    g_norms = jnp.linalg.norm(g_vecs, axis=-1)
    assert jnp.any(g_norms < 1e-10)


def test_g_vectors_count():
    """Check that number of G-vectors scales correctly with cutoff."""
    a = jnp.eye(3) * 10.0
    _, g1, _ = get_g_vectors(a, 2.0)
    _, g2, _ = get_g_vectors(a, 5.0)
    # Higher cutoff should give more G-vectors
    assert len(g2) > len(g1)


def test_pw_real_roundtrip():
    """Test plane-wave <-> real-space roundtrip."""
    a = jnp.eye(3) * 10.0
    ecut = 3.0
    g_vecs, g_idx, fft_grid = get_g_vectors(a, ecut)

    # Create some PW coefficients
    npw = len(g_vecs)
    key = jax.random.PRNGKey(42)
    coeffs = jax.random.normal(key, (npw,)) + 1j * jax.random.normal(jax.random.PRNGKey(43), (npw,))

    # Forward and back
    real_field = pw_to_real(coeffs, g_idx, fft_grid)
    coeffs_back = real_to_pw(real_field, g_idx, fft_grid)

    np.testing.assert_allclose(coeffs_back, coeffs, atol=1e-10)


def test_g_vectors_fft():
    """Test full FFT grid G-vector generation."""
    a = jnp.eye(3) * 10.0
    b = reciprocal_lattice(a)
    fft_grid = (8, 8, 8)
    g_fft = get_g_vectors_fft(fft_grid, b)
    assert g_fft.shape == (8, 8, 8, 3)
    # G=0 should be at index [0,0,0]
    np.testing.assert_allclose(g_fft[0, 0, 0], 0.0, atol=1e-15)
