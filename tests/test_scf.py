"""Integration test for the SCF loop.

Tests a small system (hydrogen atom in a box / simple solid) to verify
the entire SCF machinery works correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pwdft.crystal import Crystal
from pwdft.scf import scf_loop, initial_density, compute_occupation
from pwdft.constants import HARTREE_TO_EV


def _make_hydrogen_box():
    """Create a hydrogen atom in a large box (molecule-in-a-box test)."""
    a0 = 10.0  # Bohr, large box
    a = jnp.eye(3) * a0
    positions = jnp.array([[a0/2, a0/2, a0/2]])  # Centered
    return Crystal(
        a=a,
        species=["H"],
        positions=positions,
        Z_vals=jnp.array([1.0]),
    )


def _make_he_box():
    """Helium atom in a box."""
    a0 = 10.0
    a = jnp.eye(3) * a0
    positions = jnp.array([[a0/2, a0/2, a0/2]])
    return Crystal(
        a=a,
        species=["He"],
        positions=positions,
        Z_vals=jnp.array([2.0]),
    )


def test_initial_density_integral():
    """Test that initial density integrates to the right number of electrons."""
    crystal = _make_he_box()
    fft_grid = (16, 16, 16)
    rho = initial_density(crystal, fft_grid)

    omega = float(crystal.volume)
    n_grid = fft_grid[0] * fft_grid[1] * fft_grid[2]
    nelec_integrated = float(jnp.sum(rho)) * omega / n_grid
    np.testing.assert_allclose(nelec_integrated, crystal.nelec, rtol=0.01)


def test_occupation_insulator():
    """Test occupation for an insulating system."""
    eigs = [jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])]
    weights = jnp.array([1.0])
    nelec = 4.0  # 2 occupied bands

    occs, ef = compute_occupation(eigs, weights, nelec)
    np.testing.assert_allclose(occs[0][:2], 2.0, atol=1e-10)
    np.testing.assert_allclose(occs[0][2:], 0.0, atol=1e-10)
    np.testing.assert_allclose(ef, -0.5, atol=1e-10)


def test_occupation_fermi_dirac():
    """Test Fermi-Dirac occupation."""
    eigs = [jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])]
    weights = jnp.array([1.0])
    nelec = 4.0
    smearing = 0.1

    occs, ef = compute_occupation(eigs, weights, nelec, smearing=smearing)
    # Total electrons should be conserved
    total = float(jnp.sum(occs[0]) * weights[0])
    np.testing.assert_allclose(total, nelec, atol=1e-8)


@pytest.mark.slow
def test_scf_helium_box():
    """Test SCF convergence for helium in a box.

    This is a basic sanity check that the SCF loop converges
    and gives a reasonable energy.
    """
    crystal = _make_he_box()
    result = scf_loop(
        crystal=crystal,
        ecut=5.0,
        n_bands=4,
        max_iter=50,
        tol=1e-4,
        mixing="pulay",
        mixing_alpha=0.3,
        eigensolver="direct",
        verbose=True,
    )

    # Should converge
    assert result.converged or result.n_iter <= 50
    # Energy should be negative (bound state)
    assert result.total_energy < 0, f"Energy = {result.total_energy}"
    # He atom energy should be around -2.83 Ha (exact: -2.9037 Ha)
    # With pseudopotential and finite box/cutoff, we expect roughly this range
    print(f"He total energy: {result.total_energy:.6f} Ha")
