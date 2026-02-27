"""Self-consistent field (SCF) loop for Kohn-Sham DFT.

The SCF procedure:
1. Start with initial guess for electron density rho(r)
2. Construct effective potential V_eff = V_loc + V_H[rho] + V_xc[rho]
3. Solve Kohn-Sham equations: (-nabla^2/2 + V_eff)|psi_i> = epsilon_i |psi_i>
4. Compute new density: rho(r) = sum_i f_i |psi_i(r)|^2
5. Mix input and output density
6. Check convergence; if not converged, go to 2
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pwdft.constants import PI, FOUR_PI
from pwdft.crystal import Crystal
from pwdft.basis import (
    get_g_vectors, get_g_vectors_fft, pw_to_real, real_to_pw,
)
from pwdft.lattice import reciprocal_lattice, cell_volume
from pwdft.hamiltonian import (
    setup_basis, setup_nonlocal, build_local_potential_real,
    compute_hartree_energy, PWBasisData, NonlocalData,
)
from pwdft.eigensolver import davidson, direct_diagonalize
from pwdft.mixing import PulayMixer, simple_mixing
from pwdft.xc import xc_energy, xc_energy_density, xc_potential
from pwdft.kpoints import monkhorst_pack, gamma_point
from pwdft.ewald import ewald_energy_fast
from pwdft.pseudopotential import get_hgh_params, vloc_reciprocal


class SCFResult(NamedTuple):
    """Results of an SCF calculation."""
    converged: bool
    n_iter: int
    total_energy: float
    eigenvalues: dict  # {k_index: (n_bands,) eigenvalues}
    wavefunctions: dict  # {k_index: (npw, n_bands) wavefunctions}
    rho_r: jnp.ndarray  # (n1, n2, n3) final density
    fermi_energy: float
    band_energy: float
    hartree_energy: float
    xc_energy: float
    kinetic_energy: float
    local_pp_energy: float
    nonlocal_pp_energy: float
    ewald_energy: float


def compute_occupation(eigenvalues_all: list[jnp.ndarray],
                        weights: jnp.ndarray,
                        nelec: float,
                        smearing: float = 0.0) -> tuple[list[jnp.ndarray], float]:
    """Compute occupation numbers and Fermi energy.

    For insulators (smearing=0), simply fill the lowest nelec/2 bands
    with occupation 2 (spin-unpolarized).

    For metals (smearing > 0), use Fermi-Dirac smearing.

    Args:
        eigenvalues_all: List of eigenvalue arrays, one per k-point.
        weights: (nk,) k-point weights.
        nelec: Number of electrons.
        smearing: Smearing width in Hartree (0 for insulators).

    Returns:
        occupations: List of occupation arrays (matching eigenvalues).
        fermi_energy: Fermi energy.
    """
    # Collect all eigenvalues with k-point weights
    all_eigs = []
    all_weights = []
    all_indices = []
    for ik, eigs in enumerate(eigenvalues_all):
        for ib, e in enumerate(np.array(eigs)):
            all_eigs.append(float(e))
            all_weights.append(float(weights[ik]))
            all_indices.append((ik, ib))

    all_eigs = np.array(all_eigs)
    all_weights = np.array(all_weights)

    if smearing <= 0.0:
        # Zero-temperature occupation: fill lowest states
        # Sort by eigenvalue
        sorted_idx = np.argsort(all_eigs)

        # Fill bands (2 electrons per band for spin-unpolarized)
        n_fill = 0.0
        fermi_energy = 0.0
        occ_dict = {}
        for idx in sorted_idx:
            ik, ib = all_indices[idx]
            key = (ik, ib)
            if n_fill + 2.0 * all_weights[idx] <= nelec + 1e-10:
                occ_dict[key] = 2.0
                n_fill += 2.0 * all_weights[idx]
                fermi_energy = all_eigs[idx]
            elif n_fill < nelec - 1e-10:
                # Partial occupation
                remaining = nelec - n_fill
                occ_dict[key] = remaining / all_weights[idx]
                n_fill += remaining
                fermi_energy = all_eigs[idx]
            else:
                occ_dict[key] = 0.0

        occupations = []
        for ik, eigs in enumerate(eigenvalues_all):
            occ = jnp.array([occ_dict.get((ik, ib), 0.0) for ib in range(len(eigs))])
            occupations.append(occ)

        return occupations, fermi_energy

    else:
        # Fermi-Dirac smearing
        # Find Fermi energy by bisection
        emin = float(np.min(all_eigs)) - 10 * smearing
        emax = float(np.max(all_eigs)) + 10 * smearing

        for _ in range(100):
            ef = 0.5 * (emin + emax)
            x = (all_eigs - ef) / smearing
            fd = 1.0 / (1.0 + np.exp(np.clip(x, -500, 500)))
            n_elec_test = 2.0 * np.sum(all_weights * fd)
            if n_elec_test > nelec:
                emax = ef
            else:
                emin = ef
            if abs(n_elec_test - nelec) < 1e-12:
                break

        fermi_energy = ef

        occupations = []
        for ik, eigs in enumerate(eigenvalues_all):
            x = (np.array(eigs) - fermi_energy) / smearing
            fd = 1.0 / (1.0 + np.exp(np.clip(x, -500, 500)))
            occupations.append(jnp.array(2.0 * fd))

        return occupations, fermi_energy


def compute_density(
    wavefunctions: dict[int, jnp.ndarray],
    occupations: list[jnp.ndarray],
    bases: dict[int, PWBasisData],
    weights: jnp.ndarray,
    fft_grid: tuple[int, int, int],
    omega: float,
) -> jnp.ndarray:
    """Compute electron density from wavefunctions.

    rho(r) = sum_{k,n} w_k * f_{n,k} * |psi_{n,k}(r)|^2

    Normalized so that integral rho(r) dr = N_elec.

    Args:
        wavefunctions: {k_index: (npw_k, n_bands)} wavefunction coefficients.
        occupations: List of occupation arrays.
        bases: {k_index: PWBasisData} for each k-point.
        weights: (nk,) k-point weights.
        fft_grid: FFT grid dimensions.
        omega: Cell volume.

    Returns:
        (n1, n2, n3) electron density on real-space grid.
    """
    n1, n2, n3 = fft_grid
    n_grid = n1 * n2 * n3
    rho = jnp.zeros(fft_grid)

    for ik in wavefunctions:
        psi_k = wavefunctions[ik]  # (npw_k, n_bands)
        occ = occupations[ik]
        basis = bases[ik]
        wk = weights[ik]
        n_bands = psi_k.shape[1]

        for ib in range(n_bands):
            if float(occ[ib]) < 1e-15:
                continue
            # Transform to real space
            psi_r = pw_to_real(psi_k[:, ib], basis.g_indices, fft_grid)
            # |psi|^2 (normalize: integral |psi|^2 dr = 1 requires /omega factor)
            rho += wk * float(occ[ib]) * jnp.abs(psi_r)**2 / omega

    # At this point, rho should integrate to n_elec:
    # integral rho dr = sum_grid rho * (omega/n_grid) = n_elec
    # But our pw_to_real uses the convention that sum_r |psi_r|^2 = n_grid
    # (since IFFT normalization gives sum_r |psi_r|^2 = N * sum_G |c_G|^2)
    # With sum_G |c_G|^2 = 1, we get sum_r |psi_r|^2 = n_grid
    # So rho = sum wk * occ * |psi_r|^2 / omega has
    # integral rho dr = (omega/n_grid) * sum_r rho_r
    #                  = (omega/n_grid) * (1/omega) * sum wk*occ * n_grid
    #                  = sum wk*occ = n_elec ✓
    # (since sum_k w_k = 1 and sum_n occ_n = n_elec at each k)

    return rho


def compute_total_energy(
    crystal: Crystal,
    rho_r: jnp.ndarray,
    eigenvalues_all: list[jnp.ndarray],
    occupations: list[jnp.ndarray],
    weights: jnp.ndarray,
    fft_grid: tuple[int, int, int],
    g_vectors_fft: jnp.ndarray,
    e_ewald: float,
) -> dict[str, float]:
    """Compute the total energy and its components.

    E_total = E_band - E_H - E_xc_pot + E_xc + E_H + E_ewald
            = E_kinetic + E_loc + E_nl + E_H + E_xc + E_ewald

    We use the Harris-Foulkes expression:
        E_total = sum_{k,n} w_k * f_{nk} * epsilon_{nk}  (band energy)
                - E_H[rho]                                  (double-counting correction)
                + E_xc[rho] - integral V_xc * rho dr       (XC double-counting)
                + E_ewald

    Args:
        crystal: Crystal structure.
        rho_r: Electron density on real-space grid.
        eigenvalues_all: Eigenvalues for each k-point.
        occupations: Occupation numbers.
        weights: K-point weights.
        fft_grid: FFT grid dimensions.
        g_vectors_fft: G-vectors for full FFT grid.
        e_ewald: Ewald ion-ion energy.

    Returns:
        Dictionary of energy components.
    """
    n1, n2, n3 = fft_grid
    n_grid = n1 * n2 * n3
    omega = float(crystal.volume)
    dvol = omega / n_grid  # Volume element per grid point

    # Band energy: sum w_k * f_nk * epsilon_nk
    e_band = 0.0
    for ik, eigs in enumerate(eigenvalues_all):
        e_band += float(weights[ik]) * float(jnp.sum(occupations[ik] * eigs))

    # Hartree energy
    rho_g_fft = jnp.fft.fftn(rho_r) / n_grid
    g2_fft = jnp.sum(g_vectors_fft**2, axis=-1)
    g2_safe = jnp.where(g2_fft == 0.0, 1.0, g2_fft)
    e_hartree = 0.5 * omega * float(jnp.real(jnp.sum(
        jnp.where(g2_fft == 0.0, 0.0, FOUR_PI * jnp.abs(rho_g_fft)**2 / g2_safe)
    )))

    # XC energy and potential integral
    rho_real = jnp.maximum(jnp.real(rho_r), 1e-30)
    e_xc_val = float(xc_energy(rho_real, dvol))
    vxc_r = xc_potential(rho_real)
    e_vxc_int = float(jnp.sum(vxc_r * rho_real) * dvol)

    # Total energy using double-counting correction
    # E_total = E_band - E_H - int V_xc * rho dr + E_xc + E_ewald
    e_total = e_band - e_hartree + e_xc_val - e_vxc_int + e_ewald

    return {
        'total': e_total,
        'band': e_band,
        'hartree': e_hartree,
        'xc': e_xc_val,
        'vxc_integral': e_vxc_int,
        'ewald': e_ewald,
    }


def initial_density(crystal: Crystal, fft_grid: tuple[int, int, int]) -> jnp.ndarray:
    """Generate initial guess for electron density.

    Uses superposition of atomic densities (approximated by Gaussians).

    Args:
        crystal: Crystal structure.
        fft_grid: FFT grid dimensions.

    Returns:
        (n1, n2, n3) initial electron density.
    """
    n1, n2, n3 = fft_grid
    n_grid = n1 * n2 * n3
    omega = float(crystal.volume)
    a = crystal.a

    # Real-space grid points
    i1 = jnp.arange(n1) / n1
    i2 = jnp.arange(n2) / n2
    i3 = jnp.arange(n3) / n3
    g1, g2, g3 = jnp.meshgrid(i1, i2, i3, indexing='ij')
    frac_grid = jnp.stack([g1, g2, g3], axis=-1)  # (n1, n2, n3, 3)
    r_grid = jnp.einsum('...i,ij->...j', frac_grid, a)  # Cartesian

    rho = jnp.zeros(fft_grid)

    for ia in range(crystal.natom):
        Z = float(crystal.Z_vals[ia])
        tau = crystal.positions[ia]
        sp = crystal.species[ia]
        params = get_hgh_params(sp)
        r_loc = params.r_loc

        # Gaussian charge: rho_atom = Z * (1/(2*pi*sigma^2))^(3/2) * exp(-|r-tau|^2/(2*sigma^2))
        # Use r_loc as the width parameter
        sigma = max(r_loc, 0.5)  # Ensure reasonable width

        # Handle periodic images
        for p1 in range(-1, 2):
            for p2 in range(-1, 2):
                for p3 in range(-1, 2):
                    shift = jnp.array(p1 * a[0] + p2 * a[1] + p3 * a[2])
                    dr = r_grid - (tau + shift)[None, None, None, :]
                    r2 = jnp.sum(dr**2, axis=-1)
                    rho_atom = Z * (1.0 / (2.0 * PI * sigma**2))**1.5 * jnp.exp(-r2 / (2.0 * sigma**2))
                    rho += rho_atom

    # Normalize so that integral rho dr = nelec
    current_integral = jnp.sum(rho) * omega / n_grid
    rho = rho * crystal.nelec / float(current_integral)

    # Ensure positive
    rho = jnp.maximum(rho, 1e-10)

    return rho


def scf_loop(
    crystal: Crystal,
    ecut: float,
    kpoints: jnp.ndarray | None = None,
    kweights: jnp.ndarray | None = None,
    nk_grid: tuple[int, int, int] | None = None,
    n_bands: int | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    mixing: str = "pulay",
    mixing_alpha: float = 0.3,
    mixing_history: int = 8,
    smearing: float = 0.0,
    eigensolver: str = "davidson",
    verbose: bool = True,
) -> SCFResult:
    """Run the SCF loop.

    Args:
        crystal: Crystal structure.
        ecut: Plane-wave energy cutoff in Hartree.
        kpoints: (nk, 3) k-points in Cartesian coords. If None, uses nk_grid or Gamma.
        kweights: (nk,) k-point weights. Must be provided with kpoints.
        nk_grid: Monkhorst-Pack grid dimensions. Used if kpoints is None.
        n_bands: Number of bands to compute. Default: nelec/2 + 4.
        max_iter: Maximum SCF iterations.
        tol: SCF convergence tolerance (on total energy change).
        mixing: Mixing scheme: "simple" or "pulay".
        mixing_alpha: Mixing parameter.
        mixing_history: History depth for Pulay mixing.
        smearing: Smearing width (0 for insulators).
        eigensolver: "davidson" or "direct".
        verbose: Print convergence info.

    Returns:
        SCFResult with all computed quantities.
    """
    if verbose:
        print("=" * 60)
        print("  Plane-Wave DFT Calculation (JAX)")
        print("=" * 60)
        print(f"  System: {crystal.natom} atoms, {crystal.nelec:.0f} electrons")
        print(f"  Ecut: {ecut:.1f} Ha")
        print(f"  Cell volume: {float(crystal.volume):.4f} Bohr^3")

    # Set up k-points
    b = crystal.b
    if kpoints is None:
        if nk_grid is not None:
            kpoints, kweights = monkhorst_pack(nk_grid, b)
        else:
            kpoints, kweights = gamma_point(b)
    nk = len(kpoints)

    if verbose:
        print(f"  K-points: {nk}")

    # Number of bands
    n_occ = int(crystal.nelec / 2)
    if n_bands is None:
        n_bands = n_occ + max(4, int(0.2 * n_occ))
    n_bands = max(n_bands, n_occ + 1)

    if verbose:
        print(f"  Bands: {n_bands} ({n_occ} occupied)")

    # Set up basis for each k-point
    bases = {}
    nl_datas = {}
    for ik in range(nk):
        k = kpoints[ik]
        basis = setup_basis(crystal, ecut, k)
        nl_data = setup_nonlocal(crystal, basis)
        bases[ik] = basis
        nl_datas[ik] = nl_data
        if verbose and ik == 0:
            print(f"  Plane waves (k=0): {len(basis.kinetic)}")
            print(f"  FFT grid: {basis.fft_grid}")

    fft_grid = bases[0].fft_grid  # Use same FFT grid for all k-points

    # G-vectors for FFT grid
    g_vectors_fft = get_g_vectors_fft(fft_grid, b)

    # Ewald energy (computed once)
    if verbose:
        print("  Computing Ewald energy...", end=" ", flush=True)
    e_ewald = float(ewald_energy_fast(crystal.a, crystal.positions, crystal.Z_vals))
    if verbose:
        print(f"{e_ewald:.8f} Ha")

    # Initial density
    rho_r = initial_density(crystal, fft_grid)

    # Compute G^2 on FFT grid for Kerker preconditioning
    g2_fft_np = np.array(jnp.sum(g_vectors_fft**2, axis=-1))

    # Set up mixer
    if mixing == "pulay":
        mixer = PulayMixer(
            max_hist=mixing_history, alpha=mixing_alpha,
            use_kerker=False, fft_grid=fft_grid, g2_fft=g2_fft_np,
        )
    elif mixing == "pulay-kerker":
        mixer = PulayMixer(
            max_hist=mixing_history, alpha=mixing_alpha,
            use_kerker=True, fft_grid=fft_grid, g2_fft=g2_fft_np,
        )
    else:
        mixer = None

    # SCF loop
    omega = float(crystal.volume)
    e_total_prev = 0.0
    wavefunctions = {}
    eigenvalues_all = []
    occupations_all = []

    if verbose:
        print()
        print(f"  {'Iter':>4s}  {'Total Energy':>16s}  {'Delta E':>12s}  {'Density Change':>14s}")
        print("  " + "-" * 52)

    for scf_iter in range(max_iter):
        # Build local potential
        vloc_r, vh_r, vxc_r, vps_loc_r = build_local_potential_real(
            crystal, rho_r, g_vectors_fft, fft_grid
        )

        # Solve KS equations for each k-point
        eigenvalues_all = []
        wavefunctions = {}

        for ik in range(nk):
            basis = bases[ik]
            nl_data = nl_datas[ik]

            # Use previous wavefunctions as initial guess
            psi_init = None
            if scf_iter > 0 and ik in wavefunctions:
                psi_init = wavefunctions[ik]

            if eigensolver == "direct" or len(basis.kinetic) < 500:
                evals, evecs = direct_diagonalize(basis, vloc_r, nl_data, n_bands)
            else:
                evals, evecs = davidson(
                    basis, vloc_r, nl_data, n_bands,
                    psi_init=psi_init,
                    max_iter=100,
                    tol=1e-10,
                )

            eigenvalues_all.append(evals)
            wavefunctions[ik] = evecs

        # Compute occupations
        occupations_all, fermi_energy = compute_occupation(
            eigenvalues_all, kweights, crystal.nelec, smearing
        )

        # Compute new density
        rho_new = compute_density(wavefunctions, occupations_all, bases, kweights, fft_grid, omega)
        rho_new = jnp.maximum(rho_new, 1e-10)

        # Density change
        n_grid = fft_grid[0] * fft_grid[1] * fft_grid[2]
        drho = float(jnp.sqrt(jnp.sum((rho_new - rho_r)**2) * omega / n_grid))

        # Compute total energy
        energies = compute_total_energy(
            crystal, rho_new, eigenvalues_all, occupations_all,
            kweights, fft_grid, g_vectors_fft, e_ewald
        )
        e_total = energies['total']
        de = e_total - e_total_prev

        if verbose:
            print(f"  {scf_iter + 1:4d}  {e_total:16.10f}  {de:12.2e}  {drho:14.2e}")

        # Check convergence
        if scf_iter > 0 and abs(de) < tol and drho < tol * 10:
            if verbose:
                print()
                print(f"  SCF converged in {scf_iter + 1} iterations!")
                _print_energies(energies)
                _print_eigenvalues(eigenvalues_all, occupations_all, kweights, fermi_energy)

            return SCFResult(
                converged=True,
                n_iter=scf_iter + 1,
                total_energy=e_total,
                eigenvalues={ik: eigenvalues_all[ik] for ik in range(nk)},
                wavefunctions=wavefunctions,
                rho_r=rho_new,
                fermi_energy=fermi_energy,
                band_energy=energies['band'],
                hartree_energy=energies['hartree'],
                xc_energy=energies['xc'],
                kinetic_energy=0.0,
                local_pp_energy=0.0,
                nonlocal_pp_energy=0.0,
                ewald_energy=e_ewald,
            )

        e_total_prev = e_total

        # Mix density
        if mixer is not None:
            rho_r = mixer.mix(rho_r, rho_new)
        else:
            rho_r = simple_mixing(rho_r, rho_new, mixing_alpha)
        rho_r = jnp.maximum(rho_r, 1e-10)

    # Not converged
    if verbose:
        print(f"\n  WARNING: SCF not converged after {max_iter} iterations")
        _print_energies(energies)

    return SCFResult(
        converged=False,
        n_iter=max_iter,
        total_energy=e_total,
        eigenvalues={ik: eigenvalues_all[ik] for ik in range(nk)},
        wavefunctions=wavefunctions,
        rho_r=rho_r,
        fermi_energy=fermi_energy,
        band_energy=energies['band'],
        hartree_energy=energies['hartree'],
        xc_energy=energies['xc'],
        kinetic_energy=0.0,
        local_pp_energy=0.0,
        nonlocal_pp_energy=0.0,
        ewald_energy=e_ewald,
    )


def _print_energies(energies: dict):
    """Print energy breakdown."""
    print()
    print("  Energy breakdown (Ha):")
    print(f"    Band energy:     {energies['band']:16.10f}")
    print(f"    Hartree energy:  {energies['hartree']:16.10f}")
    print(f"    XC energy:       {energies['xc']:16.10f}")
    print(f"    Ewald energy:    {energies['ewald']:16.10f}")
    print(f"    Total energy:    {energies['total']:16.10f}")


def _print_eigenvalues(eigenvalues_all, occupations_all, kweights, fermi_energy):
    """Print eigenvalues."""
    from pwdft.constants import HARTREE_TO_EV
    print()
    print(f"  Fermi energy: {fermi_energy:.6f} Ha ({fermi_energy * HARTREE_TO_EV:.4f} eV)")
    print()
    for ik, eigs in enumerate(eigenvalues_all):
        if len(eigenvalues_all) > 1:
            print(f"  K-point {ik + 1} (weight={float(kweights[ik]):.4f}):")
        for ib, e in enumerate(np.array(eigs)):
            occ = float(occupations_all[ik][ib])
            marker = "*" if occ > 0.5 else " "
            print(f"    {marker} Band {ib + 1:3d}: {e:12.6f} Ha  ({e * HARTREE_TO_EV:10.4f} eV)  occ={occ:.4f}")
