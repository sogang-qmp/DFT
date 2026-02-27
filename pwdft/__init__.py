"""
Plane-wave pseudopotential density functional theory for solids in JAX.

This package implements a self-consistent Kohn-Sham DFT solver using:
- Plane-wave basis set expansion
- Norm-conserving Hartwigsen-Goedecker-Hutter (HGH) pseudopotentials
- LDA exchange-correlation (Perdew-Zunger parameterization of Ceperley-Alder)
- Monkhorst-Pack k-point sampling
- Davidson iterative diagonalization
- Pulay (DIIS) density mixing
- Ewald summation for ion-ion interactions
"""

from pwdft.crystal import Crystal
from pwdft.scf import scf_loop
from pwdft.calculator import DFTCalculator

__version__ = "0.1.0"
__all__ = ["Crystal", "scf_loop", "DFTCalculator"]
