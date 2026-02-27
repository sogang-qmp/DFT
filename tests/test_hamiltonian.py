"""Tests for Hamiltonian construction and application."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pwdft.crystal import Crystal
from pwdft.hamiltonian import (
    setup_basis, setup_nonlocal, apply_hamiltonian,
    compute_structure_factors, compute_vloc_total,
    compute_hartree_potential_g, build_local_potential_real,
)
from pwdft.basis import get_g_vectors_fft
from pwdft.lattice import reciprocal_lattice


def _make_si_crystal():
    """Create a silicon diamond crystal for testing."""
    a0 = 10.263  # Si lattice constant in Bohr
    a = jnp.array([
        [0.0, a0/2, a0/2],
        [a0/2, 0.0, a0/2],
        [a0/2, a0/2, 0.0],
    ], dtype=jnp.float64)
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [a0/4, a0/4, a0/4],
    ], dtype=jnp.float64)
    return Crystal(
        a=a,
        species=["Si", "Si"],
        positions=positions,
        Z_vals=jnp.array([4.0, 4.0]),
    )


def test_setup_basis():
    """Test basis setup gives reasonable number of plane waves."""
    crystal = _make_si_crystal()
    k = jnp.zeros(3)
    basis = setup_basis(crystal, ecut=5.0, k=k)

    assert len(basis.kinetic) > 0
    assert basis.g_vectors.shape[1] == 3
    assert len(basis.g_vectors) == len(basis.kinetic)
    # All kinetic energies should be <= ecut
    assert jnp.all(basis.kinetic <= 5.0 * 1.01)


def test_hamiltonian_hermitian():
    """Test that the Hamiltonian is Hermitian."""
    crystal = _make_si_crystal()
    k = jnp.zeros(3)
    basis = setup_basis(crystal, ecut=3.0, k=k)
    nl_data = setup_nonlocal(crystal, basis)

    # Build a simple potential
    b = crystal.b
    fft_grid = basis.fft_grid
    g_fft = get_g_vectors_fft(fft_grid, b)
    rho_r = jnp.ones(fft_grid) * crystal.nelec / float(crystal.volume)
    vloc_r, _, _, _ = build_local_potential_real(crystal, rho_r, g_fft, fft_grid)

    npw = len(basis.kinetic)
    # Build full Hamiltonian
    eye = jnp.eye(npw, dtype=jnp.complex128)
    H = jnp.stack([apply_hamiltonian(eye[:, i], basis, vloc_r, nl_data) for i in range(npw)], axis=1)

    # Check Hermiticity
    diff = jnp.max(jnp.abs(H - H.conj().T))
    assert float(diff) < 1e-8, f"Hamiltonian not Hermitian: max diff = {float(diff)}"


def test_structure_factor():
    """Test structure factor for a single atom at origin."""
    g_vecs = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions = jnp.array([[0.0, 0.0, 0.0]])
    species = ["Si"]

    sf = compute_structure_factors(positions, species, g_vecs)
    # Single atom at origin: S(G) = 1 for all G
    np.testing.assert_allclose(sf["Si"], 1.0, atol=1e-12)


def test_hartree_g0_zero():
    """Test that Hartree potential at G=0 is zero."""
    rho_g = jnp.array([1.0, 0.5, 0.3, 0.1]) + 0j
    g2 = jnp.array([0.0, 1.0, 4.0, 9.0])
    vh_g = compute_hartree_potential_g(rho_g, g2)
    np.testing.assert_allclose(float(jnp.abs(vh_g[0])), 0.0, atol=1e-15)
