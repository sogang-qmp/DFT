"""High-level DFT calculator interface."""

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from pwdft.crystal import Crystal
from pwdft.scf import scf_loop, SCFResult
from pwdft.kpoints import monkhorst_pack
from pwdft.constants import HARTREE_TO_EV, BOHR_TO_ANGSTROM


@dataclass
class DFTCalculator:
    """High-level interface for plane-wave DFT calculations.

    Example usage:
        crystal = Crystal.from_angstrom(lattice, species, positions, Z_vals,
                                         coords_are_fractional=True)
        calc = DFTCalculator(ecut=10.0, kgrid=(2,2,2))
        result = calc.run(crystal)
        print(f"Total energy: {result.total_energy} Ha")
    """
    ecut: float = 10.0          # Energy cutoff in Hartree
    kgrid: tuple[int, int, int] | None = None  # K-point grid (None for gamma-only)
    n_bands: int | None = None  # Number of bands (None for auto)
    max_scf: int = 100          # Max SCF iterations
    scf_tol: float = 1e-6       # SCF convergence tolerance
    mixing: str = "pulay"       # Mixing scheme
    mixing_alpha: float = 0.3   # Mixing parameter
    smearing: float = 0.0       # Smearing width
    eigensolver: str = "davidson"  # Eigensolver
    verbose: bool = True        # Print info

    def run(self, crystal: Crystal) -> SCFResult:
        """Run a DFT calculation.

        Args:
            crystal: Crystal structure.

        Returns:
            SCFResult with all computed quantities.
        """
        kpoints = None
        kweights = None

        if self.kgrid is not None:
            kpoints, kweights = monkhorst_pack(self.kgrid, crystal.b)

        return scf_loop(
            crystal=crystal,
            ecut=self.ecut,
            kpoints=kpoints,
            kweights=kweights,
            n_bands=self.n_bands,
            max_iter=self.max_scf,
            tol=self.scf_tol,
            mixing=self.mixing,
            mixing_alpha=self.mixing_alpha,
            smearing=self.smearing,
            eigensolver=self.eigensolver,
            verbose=self.verbose,
        )

    @staticmethod
    def print_summary(result: SCFResult):
        """Print a summary of the calculation results."""
        print("\n" + "=" * 50)
        print("  Calculation Summary")
        print("=" * 50)
        print(f"  Converged: {result.converged}")
        print(f"  SCF iterations: {result.n_iter}")
        print(f"  Total energy: {result.total_energy:.10f} Ha")
        print(f"               {result.total_energy * HARTREE_TO_EV:.8f} eV")
        print(f"  Fermi energy: {result.fermi_energy:.6f} Ha")
        print(f"               {result.fermi_energy * HARTREE_TO_EV:.4f} eV")

        # Band gap (if insulator)
        all_eigs = []
        all_occs = []
        for ik in result.eigenvalues:
            all_eigs.extend(np.array(result.eigenvalues[ik]))
            # We can't easily get occupations from SCFResult, but we can
            # estimate from eigenvalues relative to Fermi level
        eigs_arr = np.array(all_eigs)
        occ_eigs = eigs_arr[eigs_arr <= result.fermi_energy + 1e-6]
        unocc_eigs = eigs_arr[eigs_arr > result.fermi_energy + 1e-6]
        if len(occ_eigs) > 0 and len(unocc_eigs) > 0:
            vbm = np.max(occ_eigs)
            cbm = np.min(unocc_eigs)
            gap = cbm - vbm
            if gap > 0:
                print(f"  Band gap: {gap:.6f} Ha ({gap * HARTREE_TO_EV:.4f} eV)")
        print("=" * 50)
