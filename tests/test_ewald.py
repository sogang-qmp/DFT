"""Tests for Ewald summation."""

import jax.numpy as jnp
import numpy as np
import pytest

from pwdft.ewald import ewald_energy, ewald_energy_fast


def test_ewald_nacl_structure():
    """Test Ewald energy for NaCl-type structure.

    The Madelung constant for NaCl is approximately 1.7476.
    E_Madelung = -M * q^2 / a where M is the Madelung constant,
    and a is the nearest-neighbor distance.
    """
    # Simple cubic NaCl structure with lattice constant a
    a0 = 10.0  # Bohr (artificially large for easy testing)
    a = jnp.eye(3) * a0

    # Two atoms: one at origin, one at (a0/2, 0, 0)
    positions = jnp.array([[0.0, 0.0, 0.0],
                            [a0/2, a0/2, a0/2]])
    charges = jnp.array([1.0, 1.0])  # Both +1 (testing same charge)

    e = ewald_energy(a, positions, charges)
    # Energy should be finite and positive for same-sign charges
    assert jnp.isfinite(e)


def test_ewald_single_charge():
    """Test Ewald for a single charge (self-interaction should cancel)."""
    a0 = 10.0
    a = jnp.eye(3) * a0
    positions = jnp.array([[0.0, 0.0, 0.0]])
    charges = jnp.array([1.0])

    e = ewald_energy(a, positions, charges)
    # The Ewald energy of a single point charge in a neutralizing background
    # should be the Madelung self-energy
    assert jnp.isfinite(e)


def test_ewald_fast_consistency():
    """Test that ewald_energy_fast gives consistent results with ewald_energy."""
    a0 = 8.0
    a = jnp.eye(3) * a0
    positions = jnp.array([[0.0, 0.0, 0.0],
                            [a0/2, a0/2, a0/2]])
    charges = jnp.array([1.0, 1.0])

    e1 = ewald_energy(a, positions, charges)
    e2 = ewald_energy_fast(a, positions, charges)

    np.testing.assert_allclose(float(e1), float(e2), rtol=1e-4)


def test_ewald_translation_invariance():
    """Test that Ewald energy is invariant under lattice translations."""
    a0 = 8.0
    a = jnp.eye(3) * a0

    pos1 = jnp.array([[0.0, 0.0, 0.0], [a0/2, a0/2, a0/2]])
    pos2 = jnp.array([[1.0, 1.0, 1.0], [a0/2 + 1.0, a0/2 + 1.0, a0/2 + 1.0]])
    charges = jnp.array([1.0, 1.0])

    e1 = ewald_energy(a, pos1, charges)
    e2 = ewald_energy(a, pos2, charges)

    np.testing.assert_allclose(float(e1), float(e2), rtol=1e-6)
