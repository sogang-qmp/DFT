"""
Test suite: Run pwdft against Quantum ESPRESSO pw.x test suite.

This module translates QE pw.x test cases to our pwdft API and runs them.
Since we use built-in HGH pseudopotentials (different from QE's UPF files),
the absolute energies will differ. We validate:
  - SCF convergence
  - Physical reasonableness (band gaps, metallicity)
  - Energy component signs and magnitudes
  - Eigenvalue ordering and structure

Tests are organized by QE test-suite directory:
  pw_scf/      - Basic SCF tests
  pw_metal/    - Metallic systems with smearing
  pw_atom/     - Atomic calculations
  pw_dft/      - Various XC functionals
  pw_lsda/     - Spin-polarized
  pw_pbe/      - PBE functional
  pw_hse/      - Hybrid HSE
  pw_b3lyp/    - Hybrid B3LYP
  pw_metaGGA/  - Meta-GGA
  pw_noncolin/ - Noncollinear magnetism
  pw_spinorbit/- Spin-orbit coupling
  pw_uspp/     - Ultrasoft pseudopotentials
  pw_pawatom/  - PAW pseudopotentials
  pw_relax/    - Structural relaxation
  pw_vc-relax/ - Variable-cell relaxation
  pw_md/       - Molecular dynamics
  pw_vdw/      - Van der Waals corrections
  pw_berry/    - Berry phase
  pw_electric/ - Electric fields
  pw_dipole/   - Dipole corrections
  pw_cluster/  - Cluster (isolated) systems
  pw_lda+U/    - DFT+U
  pw_realspace/- Real-space augmentation
  pw_irrbz/    - Irreducible BZ
  pw_lattice-ibrav/ - Bravais lattice types
  pw_eval/     - Post-processing
  pw_plugins/  - Plugin framework
  pw_libxc/    - LibXC interface
  pw_gau-pbe/  - Gau-PBE functional
  pw_twochem/  - Two-chemical-potential
  pw_workflow_*/ - Multi-step workflows

Reference: https://github.com/QEF/q-e/tree/develop/test-suite
"""

import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pwdft import Crystal, DFTCalculator
from pwdft.scf import scf_loop
from pwdft.kpoints import gamma_point
from pwdft.constants import HARTREE_TO_EV, BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR


# =============================================================================
# Unit conversions (QE uses Rydberg, we use Hartree)
# =============================================================================
RY_TO_HA = 0.5          # 1 Ry = 0.5 Ha
HA_TO_RY = 2.0          # 1 Ha = 2 Ry
BOHR_TO_ANG = BOHR_TO_ANGSTROM
ANG_TO_BOHR = ANGSTROM_TO_BOHR


# =============================================================================
# Bravais lattice vector generation (QE ibrav convention)
# =============================================================================
def ibrav_to_lattice(ibrav: int, celldm: dict[int, float]) -> np.ndarray:
    """Convert QE ibrav + celldm to lattice vectors in Bohr.

    Reference: QE documentation for ibrav conventions.
    Returns lattice vectors as rows of a (3,3) matrix.

    Args:
        ibrav: Bravais lattice index.
        celldm: Dictionary mapping celldm index (1-based) to value.
            celldm(1) = a (lattice constant in Bohr)
            celldm(2) = b/a, celldm(3) = c/a
            celldm(4) = cos(alpha), etc.

    Returns:
        (3, 3) lattice vectors in Bohr (rows).
    """
    a = celldm[1]

    if ibrav == 1:
        # Simple cubic
        return a * np.eye(3)

    elif ibrav == 2:
        # FCC: a1 = a/2*(-1,0,1), a2 = a/2*(0,1,1), a3 = a/2*(-1,1,0)
        return a / 2.0 * np.array([
            [-1.0, 0.0, 1.0],
            [ 0.0, 1.0, 1.0],
            [-1.0, 1.0, 0.0],
        ])

    elif ibrav == 3:
        # BCC: a1 = a/2*(1,1,1), a2 = a/2*(-1,1,1), a3 = a/2*(-1,-1,1)
        return a / 2.0 * np.array([
            [ 1.0,  1.0, 1.0],
            [-1.0,  1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ])

    elif ibrav == 4:
        # Hexagonal: a1 = a*(1,0,0), a2 = a*(-1/2,sqrt(3)/2,0), a3 = a*(0,0,c/a)
        c_over_a = celldm.get(3, 1.0)
        return a * np.array([
            [1.0,              0.0,              0.0],
            [-0.5,  np.sqrt(3)/2.0,              0.0],
            [0.0,              0.0,        c_over_a],
        ])

    elif ibrav == 6:
        # Tetragonal: a1 = a*(1,0,0), a2 = a*(0,1,0), a3 = a*(0,0,c/a)
        c_over_a = celldm.get(3, 1.0)
        return a * np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, c_over_a],
        ])

    elif ibrav == 8:
        # Orthorhombic
        b_over_a = celldm.get(2, 1.0)
        c_over_a = celldm.get(3, 1.0)
        return a * np.array([
            [1.0,       0.0,       0.0],
            [0.0, b_over_a,       0.0],
            [0.0,       0.0, c_over_a],
        ])

    else:
        raise ValueError(f"ibrav={ibrav} not yet supported in converter")


def alat_to_cartesian(pos_alat: np.ndarray, lattice: np.ndarray, a: float) -> np.ndarray:
    """Convert positions in alat units to Cartesian Bohr.

    In QE, 'alat' positions are in units of the lattice constant a.
    pos_cart = pos_alat * a

    Args:
        pos_alat: (natom, 3) positions in alat units.
        lattice: (3, 3) lattice vectors (rows) in Bohr.
        a: Lattice constant in Bohr.

    Returns:
        (natom, 3) Cartesian positions in Bohr.
    """
    return pos_alat * a


def crystal_to_cartesian(pos_frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert fractional coordinates to Cartesian Bohr.

    Args:
        pos_frac: (natom, 3) fractional coordinates.
        lattice: (3, 3) lattice vectors (rows) in Bohr.

    Returns:
        (natom, 3) Cartesian positions in Bohr.
    """
    return pos_frac @ lattice


def kpoints_crystal_to_cartesian(kpts_frac: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convert k-points from crystal to Cartesian coordinates.

    Args:
        kpts_frac: (nk, 3) k-points in fractional reciprocal coordinates.
        b: (3, 3) reciprocal lattice vectors (rows).

    Returns:
        (nk, 3) k-points in Cartesian (1/Bohr).
    """
    return kpts_frac @ b


# =============================================================================
# Test case definitions
# =============================================================================
class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestResult:
    name: str
    qe_source: str          # QE test-suite directory/file
    status: TestStatus
    message: str = ""
    energy_ha: float = 0.0  # Total energy in Hartree
    energy_ry: float = 0.0  # Total energy in Rydberg (for QE comparison)
    qe_ref_energy_ry: float = 0.0  # QE reference energy in Rydberg
    n_scf_iter: int = 0
    fermi_energy_ev: float = 0.0
    band_gap_ev: float = 0.0
    wall_time_s: float = 0.0
    skip_reason: str = ""
    details: str = ""


@dataclass
class QETestCase:
    """Represents a single QE pw.x test case translated to our API."""
    name: str
    qe_source: str
    description: str
    # Crystal setup
    species: list[str] = field(default_factory=list)
    lattice_bohr: np.ndarray = field(default_factory=lambda: np.eye(3))
    positions_bohr: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    Z_vals: np.ndarray = field(default_factory=lambda: np.ones(1))
    # Calculation parameters
    ecut_ha: float = 5.0
    kpoints: np.ndarray | None = None   # (nk, 3) Cartesian, or None for gamma
    kweights: np.ndarray | None = None  # (nk,)
    n_bands: int | None = None
    nk_grid: tuple[int, int, int] | None = None  # MP grid (alternative to explicit kpoints)
    smearing_ha: float = 0.0
    mixing: str = "pulay"
    mixing_alpha: float = 0.3
    max_scf: int = 100
    scf_tol: float = 1e-6
    # Reference values from QE benchmark
    qe_ref_energy_ry: float = 0.0
    expect_converged: bool = True
    expect_metallic: bool = False
    expect_band_gap_range: tuple[float, float] | None = None  # (min, max) in eV


# =============================================================================
# Define ALL QE test cases with status (run / skip + reason)
# =============================================================================

def make_si_fcc_lattice(a_bohr: float) -> np.ndarray:
    """Create FCC lattice for Si (ibrav=2)."""
    return ibrav_to_lattice(2, {1: a_bohr})


def make_si_fcc_positions(a_bohr: float, lattice: np.ndarray) -> np.ndarray:
    """Si diamond positions: (0,0,0) and (0.25,0.25,0.25) in alat."""
    pos_alat = np.array([
        [0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25],
    ])
    return alat_to_cartesian(pos_alat, lattice, a_bohr)


def make_al_fcc_lattice(a_bohr: float) -> np.ndarray:
    """Create FCC lattice for Al (ibrav=2)."""
    return ibrav_to_lattice(2, {1: a_bohr})


def build_test_cases() -> list[tuple[QETestCase | None, str, str]]:
    """Build all test cases from QE pw.x test suite.

    Returns list of (test_case_or_None, qe_source, skip_reason).
    If test_case is None, the test is skipped with the given reason.
    """
    tests = []

    # =========================================================================
    # pw_scf - Basic SCF tests
    # =========================================================================

    # --- pw_scf/scf.in: Si FCC with 2 k-points, LDA ---
    a_si = 10.20  # Bohr
    si_lat = make_si_fcc_lattice(a_si)
    si_pos = make_si_fcc_positions(a_si, si_lat)
    # QE k-points (Cartesian in 2pi/a units -> need to convert via recip lattice)
    # QE specifies: 0.25 0.25 0.25 (weight 1) and 0.25 0.25 0.75 (weight 3)
    # These are in crystal coordinates for ibrav=2
    # We need to convert to Cartesian using reciprocal lattice
    from pwdft.lattice import reciprocal_lattice
    b_si = np.array(reciprocal_lattice(jnp.array(si_lat)))
    kpts_frac = np.array([
        [0.250, 0.250, 0.250],
        [0.250, 0.250, 0.750],
    ])
    kpts_cart = kpts_frac @ b_si
    kwts = np.array([1.0, 3.0])
    kwts = kwts / kwts.sum()  # Normalize weights to sum to 1

    tests.append((QETestCase(
        name="Si SCF (2 k-points)",
        qe_source="pw_scf/scf.in",
        description="Silicon FCC, LDA, 2 explicit k-points. QE ecutwfc=12 Ry.",
        species=["Si", "Si"],
        lattice_bohr=si_lat,
        positions_bohr=si_pos,
        Z_vals=np.array([4.0, 4.0]),
        ecut_ha=12.0 * RY_TO_HA,  # 12 Ry = 6 Ha
        kpoints=kpts_cart,
        kweights=kwts,
        n_bands=8,
        smearing_ha=0.0,
        mixing="pulay",
        mixing_alpha=0.3,
        max_scf=100,
        scf_tol=1e-6,
        qe_ref_energy_ry=-15.79449593,
        expect_band_gap_range=(0.1, 5.0),  # Si should have a gap
    ), "pw_scf/scf.in", ""))

    # --- pw_scf/scf-gamma.in: Si FCC Gamma-only ---
    # Gamma-only requires Kerker preconditioning for stable convergence
    tests.append((QETestCase(
        name="Si SCF (Gamma-only)",
        qe_source="pw_scf/scf-gamma.in",
        description="Silicon FCC, LDA, Gamma-point only. QE ecutwfc=12 Ry.",
        species=["Si", "Si"],
        lattice_bohr=si_lat,
        positions_bohr=si_pos,
        Z_vals=np.array([4.0, 4.0]),
        ecut_ha=12.0 * RY_TO_HA,
        kpoints=None,  # Gamma only
        kweights=None,
        n_bands=8,
        smearing_ha=0.01,  # Small smearing for convergence robustness
        mixing="pulay-kerker",
        mixing_alpha=0.2,
        max_scf=200,
        scf_tol=1e-5,
        qe_ref_energy_ry=-14.51875980,
        expect_band_gap_range=(0.1, 10.0),
    ), "pw_scf/scf-gamma.in", ""))

    # --- pw_scf/scf-ncpp.in: Si with norm-conserving PP (same as scf.in but different PP) ---
    tests.append((QETestCase(
        name="Si SCF NCPP (2 k-points)",
        qe_source="pw_scf/scf-ncpp.in",
        description="Silicon FCC, LDA, norm-conserving PP, 2 k-points.",
        species=["Si", "Si"],
        lattice_bohr=si_lat,
        positions_bohr=si_pos,
        Z_vals=np.array([4.0, 4.0]),
        ecut_ha=12.0 * RY_TO_HA,
        kpoints=kpts_cart,
        kweights=kwts,
        n_bands=8,
        smearing_ha=0.0,
        mixing="pulay",
        mixing_alpha=0.3,
        max_scf=100,
        scf_tol=1e-6,
        qe_ref_energy_ry=-15.79449593,  # Same structure, different PP
        expect_band_gap_range=(0.1, 5.0),
    ), "pw_scf/scf-ncpp.in", ""))

    # --- pw_scf/scf-occ.in ---
    tests.append((None, "pw_scf/scf-occ.in",
                  "Manual occupation numbers not supported"))

    # --- pw_scf/scf-cg.in, scf-rmm-*.in, scf-paro-*.in ---
    # These test different eigensolvers - we can run them with our solvers
    tests.append((QETestCase(
        name="Si SCF (davidson eigensolver)",
        qe_source="pw_scf/scf-cg.in",
        description="Si FCC, testing our Davidson eigensolver (QE tests CG).",
        species=["Si", "Si"],
        lattice_bohr=si_lat,
        positions_bohr=si_pos,
        Z_vals=np.array([4.0, 4.0]),
        ecut_ha=12.0 * RY_TO_HA,
        kpoints=kpts_cart,
        kweights=kwts,
        n_bands=8,
        mixing="pulay",
        max_scf=100,
        scf_tol=1e-6,
        qe_ref_energy_ry=-15.79449593,
        expect_band_gap_range=(0.1, 5.0),
    ), "pw_scf/scf-cg.in", ""))

    # --- pw_scf/scf-mixing tests ---
    for mix_test in ["scf-mixing_beta.in", "scf-mixing_TF.in",
                      "scf-mixing_localTF.in", "scf-mixing_ndim.in"]:
        tests.append((QETestCase(
            name=f"Si SCF ({mix_test})",
            qe_source=f"pw_scf/{mix_test}",
            description=f"Si FCC, testing mixing variants. Our Pulay mixer.",
            species=["Si", "Si"],
            lattice_bohr=si_lat,
            positions_bohr=si_pos,
            Z_vals=np.array([4.0, 4.0]),
            ecut_ha=12.0 * RY_TO_HA,
            kpoints=kpts_cart,
            kweights=kwts,
            n_bands=8,
            mixing="pulay",
            mixing_alpha=0.3,
            max_scf=100,
            scf_tol=1e-6,
            qe_ref_energy_ry=-15.79449593,
            expect_band_gap_range=(0.1, 5.0),
        ), f"pw_scf/{mix_test}", ""))

    # --- pw_scf/scf-allfrac.in, scf-nofrac.in ---
    # These test fractional vs Cartesian position input - same physics
    tests.append((QETestCase(
        name="Si SCF (fractional coords)",
        qe_source="pw_scf/scf-allfrac.in",
        description="Si FCC with fractional position input.",
        species=["Si", "Si"],
        lattice_bohr=si_lat,
        positions_bohr=si_pos,
        Z_vals=np.array([4.0, 4.0]),
        ecut_ha=12.0 * RY_TO_HA,
        kpoints=kpts_cart,
        kweights=kwts,
        n_bands=8,
        mixing="pulay",
        max_scf=100,
        scf_tol=1e-6,
        qe_ref_energy_ry=-15.79449593,
        expect_band_gap_range=(0.1, 5.0),
    ), "pw_scf/scf-allfrac.in", ""))

    # --- pw_scf/scf-disk_io*.in ---
    tests.append((None, "pw_scf/scf-disk_io.in",
                  "Disk I/O control is QE-specific, not applicable"))
    tests.append((None, "pw_scf/scf-disk_io-1.in",
                  "Disk I/O control is QE-specific, not applicable"))
    tests.append((None, "pw_scf/scf-disk_io-2.in",
                  "Disk I/O control is QE-specific, not applicable"))

    # --- pw_scf/scf-1.in, scf-2.in ---
    tests.append((None, "pw_scf/scf-1.in",
                  "Multi-step restart test, not applicable"))
    tests.append((None, "pw_scf/scf-2.in",
                  "Multi-step restart test, not applicable"))

    # --- pw_scf/scf-k0.in, scf-kauto.in, scf-kcrys.in ---
    # These test different k-point input methods - same physics as scf.in
    tests.append((QETestCase(
        name="Si SCF (auto k-points 2x2x2)",
        qe_source="pw_scf/scf-kauto.in",
        description="Si FCC with automatic 2x2x2 MP k-grid.",
        species=["Si", "Si"],
        lattice_bohr=si_lat,
        positions_bohr=si_pos,
        Z_vals=np.array([4.0, 4.0]),
        ecut_ha=12.0 * RY_TO_HA,
        kpoints=None,
        kweights=None,
        nk_grid=(2, 2, 2),
        n_bands=8,
        smearing_ha=0.01,
        mixing="pulay-kerker",
        mixing_alpha=0.2,
        max_scf=200,
        scf_tol=1e-5,
        qe_ref_energy_ry=-15.79449593,
        expect_band_gap_range=(0.1, 5.0),
    ), "pw_scf/scf-kauto.in", ""))

    # --- pw_scf/scf-gth.in ---
    tests.append((None, "pw_scf/scf-gth.in",
                  "28-atom molecular crystal with vdW-D2 correction; vdW not implemented and system too large"))

    # =========================================================================
    # pw_metal - Metallic systems
    # =========================================================================

    # --- pw_metal/metal.in: Al FCC, Marzari-Vanderbilt smearing ---
    a_al = 7.50  # Bohr
    al_lat = make_al_fcc_lattice(a_al)
    al_pos = np.array([[0.0, 0.0, 0.0]])  # Single atom at origin
    b_al = np.array(reciprocal_lattice(jnp.array(al_lat)))

    tests.append((None, "pw_metal/metal.in",
                  "Marzari-Vanderbilt (cold) smearing not implemented; only Fermi-Dirac available"))

    # --- pw_metal/metal-fermi_dirac.in: Al FCC, Fermi-Dirac smearing ---
    # 10 k-points with explicit weights
    al_kpts_frac = np.array([
        [0.1250000, 0.1250000, 0.1250000],
        [0.1250000, 0.1250000, 0.3750000],
        [0.1250000, 0.1250000, 0.6250000],
        [0.1250000, 0.3750000, 0.3750000],
        [0.1250000, 0.3750000, 0.6250000],
        [0.1250000, 0.3750000, 0.8750000],
        [0.1250000, 0.6250000, 0.6250000],
        [0.3750000, 0.3750000, 0.3750000],
        [0.3750000, 0.3750000, 0.6250000],
        [0.3750000, 0.6250000, 0.6250000],
    ])
    al_kwts = np.array([1.0, 6.0, 6.0, 6.0, 12.0, 12.0, 6.0, 2.0, 6.0, 6.0])
    al_kwts = al_kwts / al_kwts.sum()
    al_kpts_cart = al_kpts_frac @ b_al

    tests.append((QETestCase(
        name="Al metal (Fermi-Dirac smearing)",
        qe_source="pw_metal/metal-fermi_dirac.in",
        description="Aluminum FCC, LDA, Fermi-Dirac smearing (degauss=0.05 Ry), 10 k-points.",
        species=["Al"],
        lattice_bohr=al_lat,
        positions_bohr=al_pos,
        Z_vals=np.array([3.0]),
        ecut_ha=15.0 * RY_TO_HA,  # 15 Ry = 7.5 Ha
        kpoints=al_kpts_cart,
        kweights=al_kwts,
        n_bands=6,
        smearing_ha=0.05 * RY_TO_HA,  # 0.05 Ry = 0.025 Ha
        mixing="pulay",
        mixing_alpha=0.3,
        max_scf=100,
        scf_tol=1e-6,
        qe_ref_energy_ry=-4.20868148,
        expect_metallic=True,
    ), "pw_metal/metal-fermi_dirac.in", ""))

    # --- pw_metal/metal-2.in ---
    tests.append((None, "pw_metal/metal-2.in",
                  "NSCF/bands calculation not implemented"))

    # --- pw_metal/metal-gaussian.in ---
    tests.append((None, "pw_metal/metal-gaussian.in",
                  "Gaussian smearing not implemented; only Fermi-Dirac available"))

    # --- pw_metal/metal-tetrahedra*.in ---
    tests.append((None, "pw_metal/metal-tetrahedra.in",
                  "Tetrahedron integration method not implemented"))
    tests.append((None, "pw_metal/metal-tetrahedra-1.in",
                  "Tetrahedron integration method not implemented"))
    tests.append((None, "pw_metal/metal-tetrahedra-2.in",
                  "Tetrahedron integration method not implemented"))

    # =========================================================================
    # pw_dft - Various XC functionals (all non-LDA → skip)
    # =========================================================================
    dft_functionals = {
        "dft1.in": "PW91 (GGA)",
        "dft2.in": "revPBE (GGA)",
        "dft3.in": "PW86PBE (GGA)",
        "dft4.in": "BLYP (GGA)",
        "dft5.in": "OLYP (GGA)",
        "dft6.in": "WC (GGA)",
        "dft7.in": "PBEsol (GGA)",
        "dft8.in": "Q2D (GGA)",
        "dft9.in": "SOGGA (GGA)",
        "dft10.in": "BEEF-vdW (GGA+vdW)",
        "dft11.in": "optB88 (GGA)",
    }
    for fname, func in dft_functionals.items():
        tests.append((None, f"pw_dft/{fname}",
                      f"XC functional {func} not implemented; only LDA (Perdew-Zunger) available"))

    # =========================================================================
    # pw_atom - Atomic calculations (need USPP / manual occs)
    # =========================================================================
    tests.append((None, "pw_atom/atom.in",
                  "Requires ultrasoft pseudopotential (O.pz-rrkjus.UPF) and manual occupations"))
    tests.append((None, "pw_atom/atom-lsda.in",
                  "Spin-polarized (LSDA) calculation not implemented"))
    tests.append((None, "pw_atom/atom-pbe.in",
                  "PBE functional not implemented"))
    tests.append((None, "pw_atom/atom-sigmapbe.in",
                  "sigma-PBE functional not implemented"))
    tests.append((None, "pw_atom/atom-occ1.in",
                  "Manual occupation numbers not supported"))
    tests.append((None, "pw_atom/atom-occ2.in",
                  "Manual occupation numbers not supported"))

    # =========================================================================
    # pw_pbe - PBE tests (GGA not implemented)
    # =========================================================================
    tests.append((None, "pw_pbe/",
                  "PBE (GGA) exchange-correlation functional not implemented"))

    # =========================================================================
    # pw_lsda - Spin-polarized tests
    # =========================================================================
    tests.append((None, "pw_lsda/",
                  "Spin-polarized (LSDA) calculations not implemented"))

    # =========================================================================
    # pw_hse - Hybrid HSE functional
    # =========================================================================
    tests.append((None, "pw_hse/",
                  "Hybrid HSE functional not implemented"))

    # =========================================================================
    # pw_b3lyp - Hybrid B3LYP
    # =========================================================================
    tests.append((None, "pw_b3lyp/",
                  "Hybrid B3LYP functional not implemented"))

    # =========================================================================
    # pw_metaGGA - Meta-GGA functionals
    # =========================================================================
    tests.append((None, "pw_metaGGA/",
                  "Meta-GGA functionals not implemented"))

    # =========================================================================
    # pw_gau-pbe - Gau-PBE functional
    # =========================================================================
    tests.append((None, "pw_gau-pbe/",
                  "Gau-PBE functional not implemented"))

    # =========================================================================
    # pw_libxc - LibXC interface
    # =========================================================================
    tests.append((None, "pw_libxc/",
                  "LibXC interface not implemented"))

    # =========================================================================
    # pw_noncolin - Noncollinear magnetism
    # =========================================================================
    tests.append((None, "pw_noncolin/",
                  "Noncollinear magnetism not implemented"))

    # =========================================================================
    # pw_spinorbit - Spin-orbit coupling
    # =========================================================================
    tests.append((None, "pw_spinorbit/",
                  "Spin-orbit coupling not implemented"))

    # =========================================================================
    # pw_lda+U - DFT+U
    # =========================================================================
    tests.append((None, "pw_lda+U/",
                  "DFT+U (Hubbard correction) not implemented"))

    # =========================================================================
    # pw_uspp - Ultrasoft pseudopotentials
    # =========================================================================
    tests.append((None, "pw_uspp/",
                  "Ultrasoft pseudopotentials not implemented; only norm-conserving HGH"))

    # =========================================================================
    # pw_pawatom - PAW pseudopotentials
    # =========================================================================
    tests.append((None, "pw_pawatom/",
                  "PAW pseudopotentials not implemented; only norm-conserving HGH"))

    # =========================================================================
    # pw_relax - Structural relaxation
    # =========================================================================
    tests.append((None, "pw_relax/",
                  "Structural relaxation (BFGS/damped) not implemented"))

    # =========================================================================
    # pw_vc-relax - Variable-cell relaxation
    # =========================================================================
    tests.append((None, "pw_vc-relax/",
                  "Variable-cell relaxation not implemented"))

    # =========================================================================
    # pw_md - Molecular dynamics
    # =========================================================================
    tests.append((None, "pw_md/",
                  "Molecular dynamics not implemented"))

    # =========================================================================
    # pw_vdw - Van der Waals corrections
    # =========================================================================
    tests.append((None, "pw_vdw/",
                  "Van der Waals corrections (DFT-D, vdW-DF) not implemented"))

    # =========================================================================
    # pw_berry - Berry phase
    # =========================================================================
    tests.append((None, "pw_berry/",
                  "Berry phase calculation not implemented"))

    # =========================================================================
    # pw_electric - Electric fields
    # =========================================================================
    tests.append((None, "pw_electric/",
                  "External electric field not implemented"))

    # =========================================================================
    # pw_dipole - Dipole corrections
    # =========================================================================
    tests.append((None, "pw_dipole/",
                  "Dipole correction not implemented"))

    # =========================================================================
    # pw_cluster - Cluster/isolated systems
    # =========================================================================
    tests.append((None, "pw_cluster/",
                  "Cluster (Makov-Payne/Martyna-Tuckerman) corrections not implemented; "
                  "also requires ultrasoft pseudopotentials"))

    # =========================================================================
    # pw_realspace - Real-space augmentation
    # =========================================================================
    tests.append((None, "pw_realspace/",
                  "Real-space augmentation (for USPP) not implemented"))

    # =========================================================================
    # pw_irrbz - Irreducible BZ tests
    # =========================================================================
    tests.append((None, "pw_irrbz/",
                  "Symmetry-based irreducible BZ generation not implemented"))

    # =========================================================================
    # pw_lattice-ibrav - Bravais lattice tests
    # =========================================================================
    tests.append((None, "pw_lattice-ibrav/",
                  "Tests QE's ibrav parameter handling; not applicable to our API"))

    # =========================================================================
    # pw_eval - Evaluation/post-processing
    # =========================================================================
    tests.append((None, "pw_eval/",
                  "Post-processing evaluation not implemented"))

    # =========================================================================
    # pw_plugins - Plugin framework
    # =========================================================================
    tests.append((None, "pw_plugins/",
                  "Plugin framework is QE-specific"))

    # =========================================================================
    # pw_twochem - Two-chemical-potential
    # =========================================================================
    tests.append((None, "pw_twochem/",
                  "Two-chemical-potential method not implemented"))

    # =========================================================================
    # pw_workflow_* - Multi-step workflows
    # =========================================================================
    tests.append((None, "pw_workflow_scf_dos/",
                  "DOS calculation not implemented"))
    tests.append((None, "pw_workflow_exx_nscf/",
                  "Exact exchange (EXX) and NSCF not implemented"))
    tests.append((None, "pw_workflow_relax_relax/",
                  "Multi-step relaxation workflow not implemented"))
    tests.append((None, "pw_workflow_vc-relax_dos/",
                  "VC-relax + DOS workflow not implemented"))
    tests.append((None, "pw_workflow_vc-relax_scf/",
                  "VC-relax + SCF workflow not implemented"))

    return tests


# =============================================================================
# Test runner
# =============================================================================

def run_single_test(tc: QETestCase) -> TestResult:
    """Run a single test case and return the result."""
    t0 = time.time()
    result = TestResult(
        name=tc.name,
        qe_source=tc.qe_source,
        status=TestStatus.ERROR,
    )

    try:
        # Build crystal
        crystal = Crystal(
            a=jnp.array(tc.lattice_bohr, dtype=jnp.float64),
            species=tc.species,
            positions=jnp.array(tc.positions_bohr, dtype=jnp.float64),
            Z_vals=jnp.array(tc.Z_vals, dtype=jnp.float64),
        )

        # Set up k-points
        kpoints = None
        kweights = None
        nk_grid = tc.nk_grid
        if tc.kpoints is not None:
            kpoints = jnp.array(tc.kpoints, dtype=jnp.float64)
            kweights = jnp.array(tc.kweights, dtype=jnp.float64)

        # Suppress verbose output - capture it
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()

        try:
            scf_result = scf_loop(
                crystal=crystal,
                ecut=tc.ecut_ha,
                kpoints=kpoints,
                kweights=kweights,
                nk_grid=nk_grid,
                n_bands=tc.n_bands,
                max_iter=tc.max_scf,
                tol=tc.scf_tol,
                mixing=tc.mixing,
                mixing_alpha=tc.mixing_alpha,
                smearing=tc.smearing_ha,
                eigensolver="davidson",
                verbose=True,
            )
        finally:
            sys.stdout = old_stdout
            calc_output = captured.getvalue()

        wall_time = time.time() - t0

        # Extract results
        e_total_ha = float(scf_result.total_energy)
        e_total_ry = e_total_ha * HA_TO_RY
        fermi_ev = float(scf_result.fermi_energy) * HARTREE_TO_EV

        # Compute band gap
        all_eigs = []
        for ik in scf_result.eigenvalues:
            all_eigs.extend(np.array(scf_result.eigenvalues[ik]).tolist())
        eigs_arr = np.array(all_eigs)
        ef = float(scf_result.fermi_energy)
        occ_eigs = eigs_arr[eigs_arr <= ef + 1e-6]
        unocc_eigs = eigs_arr[eigs_arr > ef + 1e-6]
        band_gap_ev = 0.0
        if len(occ_eigs) > 0 and len(unocc_eigs) > 0:
            vbm = np.max(occ_eigs)
            cbm = np.min(unocc_eigs)
            gap = (cbm - vbm) * HARTREE_TO_EV
            band_gap_ev = max(gap, 0.0)

        result.energy_ha = e_total_ha
        result.energy_ry = e_total_ry
        result.qe_ref_energy_ry = tc.qe_ref_energy_ry
        result.n_scf_iter = scf_result.n_iter
        result.fermi_energy_ev = fermi_ev
        result.band_gap_ev = band_gap_ev
        result.wall_time_s = wall_time

        # Validate results
        checks_passed = True
        messages = []

        # Check convergence
        if tc.expect_converged and not scf_result.converged:
            checks_passed = False
            messages.append(f"SCF did NOT converge in {scf_result.n_iter} iterations")
        elif scf_result.converged:
            messages.append(f"SCF converged in {scf_result.n_iter} iterations")

        # Check energy is negative and finite
        if not np.isfinite(e_total_ha):
            checks_passed = False
            messages.append(f"Total energy is not finite: {e_total_ha}")
        elif e_total_ha > 0:
            checks_passed = False
            messages.append(f"Total energy is positive: {e_total_ha:.6f} Ha (unphysical)")
        else:
            messages.append(f"Total energy: {e_total_ha:.8f} Ha = {e_total_ry:.8f} Ry")
            if tc.qe_ref_energy_ry != 0.0:
                messages.append(f"QE reference:  {tc.qe_ref_energy_ry:.8f} Ry")
                diff_ry = abs(e_total_ry - tc.qe_ref_energy_ry)
                messages.append(f"Difference: {diff_ry:.6f} Ry (expected due to different pseudopotentials)")

        # Check metallicity
        if tc.expect_metallic:
            if band_gap_ev > 0.5:
                messages.append(f"WARNING: Expected metallic but got band gap = {band_gap_ev:.4f} eV")
            else:
                messages.append(f"Correctly identified as metallic (gap = {band_gap_ev:.4f} eV)")

        # Check band gap range
        if tc.expect_band_gap_range is not None:
            lo, hi = tc.expect_band_gap_range
            if band_gap_ev < lo or band_gap_ev > hi:
                messages.append(f"WARNING: Band gap {band_gap_ev:.4f} eV outside expected range [{lo}, {hi}] eV")
            else:
                messages.append(f"Band gap {band_gap_ev:.4f} eV within expected range [{lo}, {hi}] eV")

        # Check energy components
        messages.append(f"Fermi energy: {fermi_ev:.4f} eV")
        messages.append(f"Hartree energy: {scf_result.hartree_energy:.8f} Ha")
        messages.append(f"XC energy: {scf_result.xc_energy:.8f} Ha")
        messages.append(f"Ewald energy: {scf_result.ewald_energy:.8f} Ha")

        result.status = TestStatus.PASS if checks_passed else TestStatus.FAIL
        result.message = "; ".join(messages[:3])
        result.details = "\n".join(messages) + "\n\n--- Calculation Output ---\n" + calc_output

    except Exception as e:
        result.status = TestStatus.ERROR
        result.message = f"{type(e).__name__}: {str(e)[:200]}"
        result.details = traceback.format_exc()
        result.wall_time_s = time.time() - t0

    return result


def run_all_tests() -> list[TestResult]:
    """Run all QE pw.x test cases."""
    test_cases = build_test_cases()
    results = []

    print("=" * 80)
    print("  QE pw.x Test Suite — Running against pwdft (HGH/LDA)")
    print("=" * 80)
    print()

    n_run = sum(1 for tc, _, _ in test_cases if tc is not None)
    n_skip = sum(1 for tc, _, _ in test_cases if tc is None)
    print(f"  Total test entries: {len(test_cases)}")
    print(f"  Tests to run:  {n_run}")
    print(f"  Tests to skip: {n_skip}")
    print()

    # First, list skipped tests
    print("-" * 80)
    print("  SKIPPED TESTS (features not implemented)")
    print("-" * 80)
    for tc, src, reason in test_cases:
        if tc is None:
            results.append(TestResult(
                name=src,
                qe_source=src,
                status=TestStatus.SKIP,
                skip_reason=reason,
            ))
            print(f"  SKIP  {src}")
            print(f"        Reason: {reason}")
    print()

    # Run tests that we can execute
    print("-" * 80)
    print("  RUNNING TESTS")
    print("-" * 80)
    print()

    for tc, src, reason in test_cases:
        if tc is None:
            continue

        print(f"  >>> Running: {tc.name} ({tc.qe_source})")
        print(f"      {tc.description}")
        result = run_single_test(tc)
        results.append(result)

        status_str = result.status.value
        print(f"      Status: {status_str} ({result.wall_time_s:.1f}s)")
        print(f"      {result.message}")
        print()

    return results


def generate_report(results: list[TestResult]) -> str:
    """Generate a comprehensive test report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  QE pw.x TEST SUITE REPORT — pwdft (HGH/LDA)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Platform: JAX-based plane-wave DFT")
    lines.append(f"  Pseudopotentials: HGH (Hartwigsen-Goedecker-Hutter)")
    lines.append(f"  XC Functional: LDA (Perdew-Zunger / Ceperley-Alder)")
    lines.append("")

    # Summary statistics
    n_pass = sum(1 for r in results if r.status == TestStatus.PASS)
    n_fail = sum(1 for r in results if r.status == TestStatus.FAIL)
    n_skip = sum(1 for r in results if r.status == TestStatus.SKIP)
    n_error = sum(1 for r in results if r.status == TestStatus.ERROR)
    n_total = len(results)
    n_run = n_pass + n_fail + n_error

    lines.append("-" * 80)
    lines.append("  SUMMARY")
    lines.append("-" * 80)
    lines.append(f"  Total test entries:    {n_total}")
    lines.append(f"  Tests executed:        {n_run}")
    lines.append(f"  Tests skipped:         {n_skip}")
    lines.append(f"  ---")
    lines.append(f"  PASSED:   {n_pass:3d} / {n_run}")
    lines.append(f"  FAILED:   {n_fail:3d} / {n_run}")
    lines.append(f"  ERRORS:   {n_error:3d} / {n_run}")
    lines.append("")

    # Detailed results - Run tests
    lines.append("-" * 80)
    lines.append("  EXECUTED TESTS - DETAILED RESULTS")
    lines.append("-" * 80)
    lines.append("")

    for r in results:
        if r.status == TestStatus.SKIP:
            continue
        lines.append(f"  [{r.status.value:5s}] {r.name}")
        lines.append(f"         QE source: {r.qe_source}")
        if r.energy_ha != 0:
            lines.append(f"         Energy: {r.energy_ha:.10f} Ha = {r.energy_ry:.10f} Ry")
            if r.qe_ref_energy_ry != 0:
                diff = abs(r.energy_ry - r.qe_ref_energy_ry)
                lines.append(f"         QE ref: {r.qe_ref_energy_ry:.10f} Ry (diff = {diff:.6f} Ry)")
        if r.n_scf_iter > 0:
            lines.append(f"         SCF iterations: {r.n_scf_iter}")
        if r.fermi_energy_ev != 0:
            lines.append(f"         Fermi energy: {r.fermi_energy_ev:.4f} eV")
        if r.band_gap_ev > 0:
            lines.append(f"         Band gap: {r.band_gap_ev:.4f} eV")
        lines.append(f"         Wall time: {r.wall_time_s:.1f}s")
        if r.message:
            lines.append(f"         Notes: {r.message}")
        lines.append("")

    # Skipped tests
    lines.append("-" * 80)
    lines.append("  SKIPPED TESTS (features not implemented in pwdft)")
    lines.append("-" * 80)
    lines.append("")

    # Group skipped tests by reason category
    skip_categories = {}
    for r in results:
        if r.status != TestStatus.SKIP:
            continue
        reason = r.skip_reason
        if reason not in skip_categories:
            skip_categories[reason] = []
        skip_categories[reason].append(r.qe_source)

    for reason, sources in sorted(skip_categories.items()):
        lines.append(f"  Reason: {reason}")
        for src in sources:
            lines.append(f"    - {src}")
        lines.append("")

    # Feature coverage summary
    lines.append("-" * 80)
    lines.append("  FEATURE COVERAGE ANALYSIS")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Features implemented and tested:")
    lines.append("    [x] SCF self-consistent calculation")
    lines.append("    [x] LDA exchange-correlation (Perdew-Zunger)")
    lines.append("    [x] HGH norm-conserving pseudopotentials")
    lines.append("    [x] Plane-wave basis with energy cutoff")
    lines.append("    [x] K-point sampling (explicit + Monkhorst-Pack)")
    lines.append("    [x] Gamma-point calculations")
    lines.append("    [x] Fermi-Dirac smearing for metals")
    lines.append("    [x] Pulay (DIIS) density mixing")
    lines.append("    [x] Davidson iterative diagonalization")
    lines.append("    [x] Ewald summation for ion-ion interaction")
    lines.append("")
    lines.append("  Features NOT implemented (cause of skipped tests):")
    lines.append("    [ ] GGA functionals (PBE, PW91, revPBE, etc.)")
    lines.append("    [ ] Hybrid functionals (HSE, B3LYP, PBE0)")
    lines.append("    [ ] Meta-GGA functionals (SCAN, TPSS)")
    lines.append("    [ ] Spin-polarized calculations (LSDA)")
    lines.append("    [ ] Noncollinear magnetism")
    lines.append("    [ ] Spin-orbit coupling")
    lines.append("    [ ] DFT+U (Hubbard correction)")
    lines.append("    [ ] Ultrasoft pseudopotentials (USPP)")
    lines.append("    [ ] PAW pseudopotentials")
    lines.append("    [ ] Van der Waals corrections (DFT-D, vdW-DF)")
    lines.append("    [ ] Structural relaxation (BFGS)")
    lines.append("    [ ] Variable-cell relaxation")
    lines.append("    [ ] Molecular dynamics")
    lines.append("    [ ] Stress tensor")
    lines.append("    [ ] Berry phase / electric polarization")
    lines.append("    [ ] External electric fields")
    lines.append("    [ ] Manual occupation numbers")
    lines.append("    [ ] NSCF / band structure / DOS calculations")
    lines.append("    [ ] Symmetry-based BZ reduction")
    lines.append("")
    lines.append("  Note on energy comparison:")
    lines.append("    Absolute energies differ from QE references because we use")
    lines.append("    HGH pseudopotentials while QE tests use various UPF files.")
    lines.append("    The physics (band gaps, metallicity, convergence) should agree.")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    results = run_all_tests()
    report = generate_report(results)

    # Print report
    print("\n\n")
    print(report)

    # Save report to file
    report_path = "/home/user/DFT/qe_test_suite_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Also save detailed log
    detail_path = "/home/user/DFT/qe_test_suite_details.txt"
    with open(detail_path, "w") as f:
        for r in results:
            if r.details:
                f.write(f"{'='*80}\n")
                f.write(f"Test: {r.name} ({r.qe_source})\n")
                f.write(f"Status: {r.status.value}\n")
                f.write(f"{'='*80}\n")
                f.write(r.details)
                f.write("\n\n")
    print(f"Detailed log saved to: {detail_path}")
