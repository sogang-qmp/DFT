"""Exchange-correlation functionals.

Implements LDA exchange-correlation using the Perdew-Zunger (1981)
parameterization of the Ceperley-Alder (1980) quantum Monte Carlo data.

References:
    J. P. Perdew, A. Zunger, Phys. Rev. B 23, 5048 (1981).
    D. M. Ceperley, B. J. Alder, Phys. Rev. Lett. 45, 566 (1980).
"""

import jax
import jax.numpy as jnp

from pwdft.constants import PI


# ============================================================================
# LDA Exchange
# ============================================================================

def exchange_energy_density(rho: jnp.ndarray) -> jnp.ndarray:
    """LDA exchange energy density per particle epsilon_x(rho).

    epsilon_x = -(3/4) * (3*rho/pi)^(1/3)

    Args:
        rho: Electron density (positive values).

    Returns:
        Exchange energy density per particle.
    """
    rho_safe = jnp.maximum(rho, 1e-30)
    return -0.75 * (3.0 * rho_safe / PI) ** (1.0 / 3.0)


def exchange_potential(rho: jnp.ndarray) -> jnp.ndarray:
    """LDA exchange potential V_x = d(rho * epsilon_x)/d(rho).

    V_x = -(3/pi)^(1/3) * rho^(1/3) = (4/3) * epsilon_x

    Args:
        rho: Electron density.

    Returns:
        Exchange potential.
    """
    rho_safe = jnp.maximum(rho, 1e-30)
    return -(3.0 / PI * rho_safe) ** (1.0 / 3.0)


# ============================================================================
# LDA Correlation: Perdew-Zunger parameterization
# ============================================================================

# PZ81 parameters for unpolarized (paramagnetic) case
_GAMMA_PZ = -0.1423
_BETA1_PZ = 1.0529
_BETA2_PZ = 0.3334
_A_PZ = 0.0311
_B_PZ = -0.0480
_C_PZ = 0.0020
_D_PZ = -0.0116


def correlation_energy_density(rho: jnp.ndarray) -> jnp.ndarray:
    """Perdew-Zunger LDA correlation energy density per particle.

    Two regimes based on r_s = (3/(4*pi*rho))^(1/3):
        r_s >= 1: epsilon_c = gamma / (1 + beta1*sqrt(r_s) + beta2*r_s)
        r_s < 1:  epsilon_c = A*ln(r_s) + B + C*r_s*ln(r_s) + D*r_s

    Args:
        rho: Electron density.

    Returns:
        Correlation energy density per particle.
    """
    rho_safe = jnp.maximum(rho, 1e-30)
    rs = (3.0 / (4.0 * PI * rho_safe)) ** (1.0 / 3.0)

    # High-density regime (r_s < 1)
    ec_high = _A_PZ * jnp.log(rs) + _B_PZ + _C_PZ * rs * jnp.log(rs) + _D_PZ * rs

    # Low-density regime (r_s >= 1)
    sqrt_rs = jnp.sqrt(rs)
    ec_low = _GAMMA_PZ / (1.0 + _BETA1_PZ * sqrt_rs + _BETA2_PZ * rs)

    return jnp.where(rs < 1.0, ec_high, ec_low)


def correlation_potential(rho: jnp.ndarray) -> jnp.ndarray:
    """Perdew-Zunger LDA correlation potential.

    V_c = epsilon_c + rho * d(epsilon_c)/d(rho)
        = epsilon_c - (r_s/3) * d(epsilon_c)/d(r_s)

    Args:
        rho: Electron density.

    Returns:
        Correlation potential.
    """
    rho_safe = jnp.maximum(rho, 1e-30)
    rs = (3.0 / (4.0 * PI * rho_safe)) ** (1.0 / 3.0)

    # High-density regime
    ec_high = _A_PZ * jnp.log(rs) + _B_PZ + _C_PZ * rs * jnp.log(rs) + _D_PZ * rs
    dec_drs_high = _A_PZ / rs + _C_PZ * (jnp.log(rs) + 1.0) + _D_PZ
    vc_high = ec_high - (rs / 3.0) * dec_drs_high

    # Low-density regime
    sqrt_rs = jnp.sqrt(rs)
    denom = 1.0 + _BETA1_PZ * sqrt_rs + _BETA2_PZ * rs
    ec_low = _GAMMA_PZ / denom
    dec_drs_low = -_GAMMA_PZ * (_BETA1_PZ / (2.0 * sqrt_rs) + _BETA2_PZ) / denom**2
    vc_low = ec_low - (rs / 3.0) * dec_drs_low

    return jnp.where(rs < 1.0, vc_high, vc_low)


# ============================================================================
# Combined XC
# ============================================================================

def xc_energy_density(rho: jnp.ndarray) -> jnp.ndarray:
    """Total LDA XC energy density per particle.

    Args:
        rho: Electron density.

    Returns:
        XC energy density per particle (exchange + correlation).
    """
    return exchange_energy_density(rho) + correlation_energy_density(rho)


def xc_potential(rho: jnp.ndarray) -> jnp.ndarray:
    """Total LDA XC potential.

    Args:
        rho: Electron density.

    Returns:
        XC potential V_xc(r).
    """
    return exchange_potential(rho) + correlation_potential(rho)


def xc_energy(rho: jnp.ndarray, volume_element: float) -> jnp.ndarray:
    """Total XC energy integrated over the cell.

    E_xc = integral rho(r) * epsilon_xc(rho(r)) dr

    Args:
        rho: Electron density on real-space grid.
        volume_element: Volume per grid point (Omega / N_grid).

    Returns:
        Scalar XC energy.
    """
    exc = xc_energy_density(rho)
    return jnp.sum(rho * exc) * volume_element
