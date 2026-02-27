"""Tests for exchange-correlation functionals."""

import jax.numpy as jnp
import numpy as np
import pytest

from pwdft.xc import (
    exchange_energy_density, exchange_potential,
    correlation_energy_density, correlation_potential,
    xc_energy_density, xc_potential,
)


def test_exchange_energy_density_values():
    """Test exchange energy density against known values."""
    # For uniform electron gas with r_s = 1:
    # rho = 3/(4*pi*r_s^3) = 3/(4*pi)
    rs = 1.0
    rho = jnp.array(3.0 / (4.0 * jnp.pi * rs**3))
    ex = exchange_energy_density(rho)
    # Known: ex = -(3/4)*(3*rho/pi)^(1/3) = -(3/4)*(9/(4*pi^2))^(1/3)
    expected = -0.75 * (3.0 * float(rho) / float(jnp.pi)) ** (1.0 / 3.0)
    np.testing.assert_allclose(ex, expected, rtol=1e-10)


def test_exchange_potential_relation():
    """Test that V_x = (4/3) * epsilon_x."""
    rho = jnp.linspace(0.001, 1.0, 100)
    ex = exchange_energy_density(rho)
    vx = exchange_potential(rho)
    np.testing.assert_allclose(vx, 4.0 / 3.0 * ex, rtol=1e-10)


def test_correlation_continuity():
    """Test that correlation is continuous at r_s = 1."""
    # Values just below and above r_s = 1
    rs_low = 0.999
    rs_high = 1.001
    rho_low = 3.0 / (4.0 * jnp.pi * rs_low**3)
    rho_high = 3.0 / (4.0 * jnp.pi * rs_high**3)

    ec_low = correlation_energy_density(jnp.array(rho_low))
    ec_high = correlation_energy_density(jnp.array(rho_high))
    # Should be approximately equal
    np.testing.assert_allclose(ec_low, ec_high, atol=1e-4)


def test_correlation_potential_continuity():
    """Test that correlation potential is continuous at r_s = 1."""
    rs_low = 0.999
    rs_high = 1.001
    rho_low = 3.0 / (4.0 * jnp.pi * rs_low**3)
    rho_high = 3.0 / (4.0 * jnp.pi * rs_high**3)

    vc_low = correlation_potential(jnp.array(rho_low))
    vc_high = correlation_potential(jnp.array(rho_high))
    np.testing.assert_allclose(vc_low, vc_high, atol=1e-3)


def test_xc_negative():
    """Test that XC energy density is negative for positive densities."""
    rho = jnp.linspace(0.001, 10.0, 100)
    exc = xc_energy_density(rho)
    assert jnp.all(exc < 0)


def test_correlation_known_values():
    """Test PZ correlation against known values at specific r_s."""
    # At r_s = 2 (low density regime):
    # ec should be approximately -0.0480 (known value close to this)
    rs = 2.0
    rho = jnp.array(3.0 / (4.0 * jnp.pi * rs**3))
    ec = float(correlation_energy_density(rho))
    # PZ parameterization at r_s=2: gamma/(1 + beta1*sqrt(2) + beta2*2)
    expected = -0.1423 / (1.0 + 1.0529 * np.sqrt(2.0) + 0.3334 * 2.0)
    np.testing.assert_allclose(ec, expected, rtol=1e-6)
