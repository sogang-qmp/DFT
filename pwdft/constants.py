"""Physical constants in atomic units (Hartree)."""

import jax.numpy as jnp

# In atomic units: hbar = m_e = e = 4*pi*eps_0 = 1
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# Pi
PI = jnp.pi
TWO_PI = 2.0 * jnp.pi
FOUR_PI = 4.0 * jnp.pi
