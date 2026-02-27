"""Example: Hydrogen molecule in a box (molecule-in-a-box test).

This tests the code on the simplest possible system: H2 in a large
periodic box, effectively isolating the molecule.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pwdft import Crystal, DFTCalculator
from pwdft.constants import ANGSTROM_TO_BOHR, HARTREE_TO_EV

# Large cubic box
box_size = 12.0  # Bohr

# H2 molecule: bond length ~ 0.74 Angstrom = 1.40 Bohr
d = 1.40  # Bohr
center = box_size / 2.0

lattice = jnp.eye(3) * box_size

positions = jnp.array([
    [center - d/2, center, center],
    [center + d/2, center, center],
], dtype=jnp.float64)

crystal = Crystal(
    a=lattice,
    species=["H", "H"],
    positions=positions,
    Z_vals=jnp.array([1.0, 1.0]),
)

print(f"H2 molecule in {box_size:.1f} Bohr box")
print(f"Bond length: {d:.2f} Bohr ({d / ANGSTROM_TO_BOHR:.2f} Å)")
print()

# Run DFT
calc = DFTCalculator(
    ecut=10.0,     # Higher cutoff needed for H
    kgrid=None,    # Gamma-only (molecule)
    n_bands=4,
    max_scf=60,
    scf_tol=1e-5,
    mixing="pulay",
    mixing_alpha=0.3,
    eigensolver="direct",
    verbose=True,
)

result = calc.run(crystal)
calc.print_summary(result)

# Compare with known value
# H2 total energy (LDA): approximately -1.17 Ha
print(f"\nH2 energy: {result.total_energy:.6f} Ha ({result.total_energy * HARTREE_TO_EV:.4f} eV)")
