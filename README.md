# PWDFT

Plane-wave pseudopotential density functional theory for solids, written in JAX.

PWDFT implements a self-consistent Kohn-Sham DFT solver with a plane-wave basis set, norm-conserving pseudopotentials, and full Brillouin zone sampling. JAX provides automatic differentiation and GPU acceleration.

## Features

- **Plane-wave basis** with kinetic energy cutoff and automatic FFT grid sizing
- **HGH pseudopotentials** (Hartwigsen-Goedecker-Hutter norm-conserving) with built-in parameters for H, C, N, O, Si, and more
- **LDA exchange-correlation** using Perdew-Zunger parameterization of Ceperley-Alder data
- **Monkhorst-Pack k-point sampling** with time-reversal symmetry reduction
- **Eigensolvers**: Davidson iterative diagonalization, LOBPCG, and direct diagonalization
- **Density mixing**: simple linear, Pulay DIIS, and Anderson mixing with optional Kerker preconditioning
- **Ewald summation** for periodic ion-ion electrostatics
- **Hellmann-Feynman forces** (local, nonlocal, and Ewald contributions)
- **Fermi-Dirac smearing** for metallic systems or robust convergence

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

Dependencies: `jax`, `jaxlib`, `numpy`, `scipy`.

## Quick start

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # Required for DFT accuracy

from pwdft import Crystal, DFTCalculator
from pwdft.constants import ANGSTROM_TO_BOHR

# Silicon diamond structure
a0 = 5.43 * ANGSTROM_TO_BOHR
lattice = jnp.array([
    [0.0, a0/2, a0/2],
    [a0/2, 0.0, a0/2],
    [a0/2, a0/2, 0.0],
], dtype=jnp.float64)

positions_frac = jnp.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
positions_cart = positions_frac @ lattice

crystal = Crystal(
    a=lattice,
    species=["Si", "Si"],
    positions=positions_cart,
    Z_vals=jnp.array([4.0, 4.0]),
)

calc = DFTCalculator(
    ecut=5.0,
    kgrid=(2, 2, 2),
    n_bands=8,
    scf_tol=1e-5,
    mixing="pulay",
    eigensolver="davidson",
    verbose=True,
)

result = calc.run(crystal)
calc.print_summary(result)
```

See `examples/` for complete runnable scripts (Si diamond, H2 molecule).

## Project structure

```
pwdft/
  calculator.py      High-level DFTCalculator interface
  scf.py             Self-consistent field loop
  hamiltonian.py     Kohn-Sham Hamiltonian construction and application
  pseudopotential.py HGH pseudopotential parameters and evaluation
  eigensolver.py     Davidson, LOBPCG, and direct eigensolvers
  basis.py           Plane-wave basis and FFT utilities
  xc.py              LDA exchange-correlation functionals
  ewald.py           Ewald ion-ion energy summation
  mixing.py          Density mixing (simple, Pulay DIIS, Anderson)
  forces.py          Hellmann-Feynman atomic forces
  crystal.py         Crystal structure representation
  kpoints.py         Monkhorst-Pack k-point generation
  lattice.py         Lattice vector utilities
  constants.py       Physical constants (atomic units)
examples/
  silicon_diamond.py Bulk Si calculation
  hydrogen_molecule.py H2 molecule-in-a-box test
tests/
  test_*.py          Unit tests for each module
```

## Running tests

```bash
pip install -e ".[test]"
pytest
```

## How it works

The code solves the Kohn-Sham equations self-consistently:

1. **Initialization** &mdash; Build plane-wave basis sets and precompute Kleinman-Bylander nonlocal projectors for each k-point.
2. **Initial density** &mdash; Superposition of atomic Gaussian densities.
3. **SCF loop** &mdash; Construct the effective potential (Hartree + XC + local pseudopotential), diagonalize the Hamiltonian at each k-point, compute new electron density, and mix with the previous density until convergence.
4. **Total energy** &mdash; Band energy with Harris-Foulkes double-counting corrections plus Ewald ion-ion energy.

All quantities are in Hartree atomic units internally.

## References

- Hartwigsen, Goedecker, Hutter, *Phys. Rev. B* **58**, 3641 (1998) &mdash; HGH pseudopotentials
- Perdew, Zunger, *Phys. Rev. B* **23**, 5048 (1981) &mdash; LDA parameterization
- Ceperley, Alder, *Phys. Rev. Lett.* **45**, 566 (1980) &mdash; QMC data for homogeneous electron gas

## License

MIT
