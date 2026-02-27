"""Example: Silicon diamond structure DFT calculation.

Performs a self-consistent DFT calculation on bulk silicon
in the diamond structure using plane-wave basis with HGH pseudopotentials.
"""

import jax
import jax.numpy as jnp

# Enable 64-bit precision (essential for DFT)
jax.config.update("jax_enable_x64", True)

from pwdft import Crystal, DFTCalculator
from pwdft.constants import ANGSTROM_TO_BOHR, HARTREE_TO_EV

# Silicon diamond structure
# Lattice constant: 5.43 Angstrom = 10.263 Bohr
a0_ang = 5.43  # Angstrom
a0 = a0_ang * ANGSTROM_TO_BOHR  # Bohr

# FCC lattice vectors
lattice = jnp.array([
    [0.0, a0/2, a0/2],
    [a0/2, 0.0, a0/2],
    [a0/2, a0/2, 0.0],
], dtype=jnp.float64)

# Two Si atoms in diamond basis (fractional coordinates)
positions_frac = jnp.array([
    [0.00, 0.00, 0.00],
    [0.25, 0.25, 0.25],
])

# Create crystal (positions in fractional coords, lattice in Bohr)
positions_cart = positions_frac @ lattice
crystal = Crystal(
    a=lattice,
    species=["Si", "Si"],
    positions=positions_cart,
    Z_vals=jnp.array([4.0, 4.0]),
)

print(f"Lattice constant: {a0_ang:.4f} Å ({a0:.4f} Bohr)")
print(f"Cell volume: {float(crystal.volume):.4f} Bohr^3")
print(f"Number of electrons: {crystal.nelec}")
print()

# Run DFT calculation
# Using a modest cutoff and k-grid for demonstration
calc = DFTCalculator(
    ecut=5.0,           # 5 Ha ~ 136 eV (modest for production, fine for demo)
    kgrid=(2, 2, 2),    # 2x2x2 Monkhorst-Pack grid
    n_bands=8,           # 4 occupied + 4 empty
    max_scf=100,
    scf_tol=1e-5,
    mixing="pulay",
    mixing_alpha=0.5,
    smearing=0.01,       # Small Fermi-Dirac smearing for robust convergence
    eigensolver="davidson",
    verbose=True,
)

result = calc.run(crystal)
calc.print_summary(result)
