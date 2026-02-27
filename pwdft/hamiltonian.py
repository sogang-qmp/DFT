"""Kohn-Sham Hamiltonian construction and application.

The KS Hamiltonian in a plane-wave basis at k-point k is:
    H_{G,G'} = T_{G,G'} + V_loc(G-G') + V_H(G-G') + V_xc(G-G') + V_nl_{G,G'}

where:
    T_{G,G'} = (1/2)|k+G|^2 * delta_{G,G'}  (kinetic energy)
    V_loc(G-G') = local pseudopotential (diagonal in the same basis)
    V_H(G-G') = Hartree potential
    V_xc(G-G') = XC potential
    V_nl_{G,G'} = nonlocal pseudopotential (Kleinman-Bylander form)
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pwdft.constants import PI, FOUR_PI
from pwdft.crystal import Crystal
from pwdft.basis import (
    get_g_vectors, get_g_vectors_fft, kinetic_energies,
    pw_to_real, real_to_pw, g_vector_norms_sq,
)
from pwdft.lattice import reciprocal_lattice, cell_volume
from pwdft.pseudopotential import (
    HGHParams, vloc_reciprocal, projector_fourier,
    real_spherical_harmonics, get_hgh_params,
)
from pwdft.xc import xc_potential, xc_energy_density


class PWBasisData(NamedTuple):
    """Precomputed data for a plane-wave basis at a single k-point."""
    g_vectors: jnp.ndarray      # (npw, 3) G-vectors
    g_indices: jnp.ndarray      # (npw, 3) integer Miller indices
    kg_vectors: jnp.ndarray     # (npw, 3) k+G vectors
    kinetic: jnp.ndarray        # (npw,) kinetic energies
    fft_grid: tuple[int, int, int]


class NonlocalData(NamedTuple):
    """Precomputed nonlocal projector data for a k-point."""
    # For each (atom, l, m, i) channel:
    projectors: jnp.ndarray     # (n_proj, npw) projector values <G|p>
    hij_flat: jnp.ndarray       # (n_proj, n_proj) block-diagonal h matrix


def setup_basis(crystal: Crystal, ecut: float, k: jnp.ndarray) -> PWBasisData:
    """Set up the plane-wave basis for a given k-point.

    Selects G-vectors satisfying |k+G|^2/2 <= ecut.

    Args:
        crystal: Crystal structure.
        ecut: Energy cutoff in Hartree.
        k: (3,) k-point in Cartesian coordinates.

    Returns:
        PWBasisData for this k-point.
    """
    g_vecs, g_idx, fft_grid = get_g_vectors(crystal.a, ecut)

    # Filter: keep only G where |k+G|^2/2 <= ecut
    kg = g_vecs + k[None, :]
    ke = 0.5 * jnp.sum(kg**2, axis=-1)
    mask = ke <= ecut * (1.0 + 1e-8)

    # Apply mask using numpy for dynamic sizing
    mask_np = np.array(mask)
    g_vecs_k = jnp.array(np.array(g_vecs)[mask_np])
    g_idx_k = jnp.array(np.array(g_idx)[mask_np])
    kg_k = jnp.array(np.array(kg)[mask_np])
    ke_k = jnp.array(np.array(ke)[mask_np])

    return PWBasisData(
        g_vectors=g_vecs_k,
        g_indices=g_idx_k,
        kg_vectors=kg_k,
        kinetic=ke_k,
        fft_grid=fft_grid,
    )


def compute_structure_factors(
    positions: jnp.ndarray,
    species: list[str],
    g_vectors: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute structure factors S_alpha(G) = sum_i exp(-i G . tau_i) for each species.

    Args:
        positions: (natom, 3) atomic positions.
        species: List of element symbols.
        g_vectors: (npw, 3) G-vectors.

    Returns:
        Dict mapping species -> (npw,) complex structure factor.
    """
    unique_species = list(set(species))
    sf = {}
    for sp in unique_species:
        mask = jnp.array([1.0 if s == sp else 0.0 for s in species])
        # S(G) = sum_i mask_i * exp(-i G . tau_i)
        phases = jnp.exp(-1j * positions @ g_vectors.T)  # (natom, npw)
        sf[sp] = jnp.sum(mask[:, None] * phases, axis=0)
    return sf


def compute_vloc_total(
    crystal: Crystal,
    g_vectors: jnp.ndarray,
    fft_grid: tuple[int, int, int],
) -> jnp.ndarray:
    """Compute total local pseudopotential in reciprocal space.

    V_loc^tot(G) = sum_alpha S_alpha(G) * V_loc_alpha(G)

    where alpha runs over unique species.

    Args:
        crystal: Crystal structure.
        g_vectors: (npw, 3) G-vectors.
        fft_grid: FFT grid dimensions.

    Returns:
        (npw,) total local potential in reciprocal space (per unit cell volume).
    """
    omega = float(crystal.volume)
    sf = compute_structure_factors(crystal.positions, crystal.species, g_vectors)
    g2 = jnp.sum(g_vectors**2, axis=-1)

    vloc = jnp.zeros(len(g_vectors), dtype=jnp.complex128)
    unique_species = list(set(crystal.species))

    for sp in unique_species:
        params = get_hgh_params(sp)
        # vloc_reciprocal returns V(G)*Omega (includes 1/Omega normalization)
        vloc_sp = vloc_reciprocal(g2, params, omega)  # (npw,) real
        vloc = vloc + sf[sp] * vloc_sp

    return vloc


def setup_nonlocal(
    crystal: Crystal,
    basis: PWBasisData,
) -> NonlocalData:
    """Precompute nonlocal projector data for KB form.

    The nonlocal potential in KB form:
        V_nl = sum_{a,l,m,i,j} |p^l_{m,i}(a)> h^l_{ij} <p^l_{m,j}(a)|

    where |p^l_{m,i}(a)> = p^l_i(|k+G|) * Y_{lm}(k+G) * exp(-i(k+G).tau_a) / sqrt(Omega)

    Args:
        crystal: Crystal structure.
        basis: Plane-wave basis data.

    Returns:
        NonlocalData with precomputed projectors and h matrix.
    """
    omega = float(crystal.volume)
    kg = basis.kg_vectors  # (npw, 3)
    npw = len(kg)

    # Collect all projector channels
    projector_list = []  # List of (npw,) arrays
    # Build block-diagonal h matrix
    h_blocks = []
    block_indices = []  # (start, end, h_block) for assembling the matrix

    unique_species = list(set(crystal.species))
    proj_idx = 0

    for sp in unique_species:
        params = get_hgh_params(sp)
        atom_indices = [ia for ia, s in enumerate(crystal.species) if s == sp]

        for l_ch in range(len(params.r_nl)):
            r_l = params.r_nl[l_ch]
            h_l = jnp.array(params.h_nl[l_ch])

            # Determine number of projectors for this l
            # Count nonzero diagonal elements
            n_proj_l = 0
            for ii in range(h_l.shape[0]):
                if jnp.any(h_l[ii, :] != 0) or jnp.any(h_l[:, ii] != 0):
                    n_proj_l = ii + 1
            if n_proj_l == 0:
                continue

            # Compute radial projectors for this channel
            radial_projs = []
            for i_proj in range(n_proj_l):
                p_rad = projector_fourier(kg, l_ch, i_proj, r_l)  # (npw,)
                radial_projs.append(p_rad)

            # Spherical harmonics for this l
            ylm = real_spherical_harmonics(kg, l_ch)  # (2l+1, npw)

            # For each atom of this species
            for ia in atom_indices:
                tau = crystal.positions[ia]
                # Phase factor: exp(-i (k+G) . tau)
                phase = jnp.exp(-1j * kg @ tau)  # (npw,) complex

                # Build projectors: p^l_{m,i} = radial_i * Y_lm * phase / sqrt(Omega)
                block_start = proj_idx
                for i_proj in range(n_proj_l):
                    for m in range(2 * l_ch + 1):
                        proj = radial_projs[i_proj] * ylm[m] * phase / jnp.sqrt(omega)
                        projector_list.append(proj)
                        proj_idx += 1

                block_end = proj_idx
                n_block = n_proj_l * (2 * l_ch + 1)

                # Build the h-matrix block for this atom
                # h_{(i,m),(j,m')} = h^l_{ij} * delta_{m,m'}
                h_block = jnp.zeros((n_block, n_block))
                for i_proj in range(n_proj_l):
                    for j_proj in range(n_proj_l):
                        for m in range(2 * l_ch + 1):
                            row = i_proj * (2 * l_ch + 1) + m
                            col = j_proj * (2 * l_ch + 1) + m
                            h_block = h_block.at[row, col].set(h_l[i_proj, j_proj])

                h_blocks.append((block_start, block_end, h_block))

    if len(projector_list) == 0:
        # No nonlocal projectors
        return NonlocalData(
            projectors=jnp.zeros((0, npw), dtype=jnp.complex128),
            hij_flat=jnp.zeros((0, 0)),
        )

    # Stack projectors: (n_proj_total, npw)
    projectors = jnp.stack(projector_list, axis=0)
    n_proj_total = projectors.shape[0]

    # Assemble full block-diagonal h matrix
    hij = jnp.zeros((n_proj_total, n_proj_total))
    for block_start, block_end, h_block in h_blocks:
        hij = hij.at[block_start:block_end, block_start:block_end].set(h_block)

    return NonlocalData(projectors=projectors, hij_flat=hij)


def apply_hamiltonian(
    psi_g: jnp.ndarray,
    basis: PWBasisData,
    vloc_r: jnp.ndarray,
    nl_data: NonlocalData,
) -> jnp.ndarray:
    """Apply the Kohn-Sham Hamiltonian to a wavefunction in reciprocal space.

    H|psi> = T|psi> + V_loc|psi> + V_nl|psi>

    where V_loc includes local PP + Hartree + XC.

    The local potential is applied via FFT:
        (V_loc * psi)(G) = IFFT[ V_loc(r) * FFT[psi(G)](r) ](G)

    Args:
        psi_g: (npw,) wavefunction coefficients in reciprocal space.
        basis: PWBasisData.
        vloc_r: (n1, n2, n3) total local potential on real-space grid.
        nl_data: Precomputed nonlocal projector data.

    Returns:
        (npw,) H|psi> in reciprocal space.
    """
    npw = len(psi_g)
    fft_grid = basis.fft_grid

    # Kinetic: T|psi> = (1/2)|k+G|^2 * psi(G)
    h_psi = basis.kinetic * psi_g

    # Local potential via FFT
    # 1. Put psi on FFT grid and transform to real space
    psi_r = pw_to_real(psi_g, basis.g_indices, fft_grid)  # (n1,n2,n3) complex
    # 2. Multiply by local potential
    vpsi_r = vloc_r * psi_r
    # 3. Transform back to reciprocal space
    vpsi_g = real_to_pw(vpsi_r, basis.g_indices, fft_grid)
    h_psi = h_psi + vpsi_g

    # Nonlocal: V_nl|psi> = sum_{ij} |p_i> h_{ij} <p_j|psi>
    if nl_data.projectors.shape[0] > 0:
        # <p_j|psi> = sum_G p_j(G)^* psi(G)
        proj_psi = nl_data.projectors.conj() @ psi_g  # (n_proj,) complex
        # h @ <p|psi>
        h_proj_psi = nl_data.hij_flat @ proj_psi  # (n_proj,) complex
        # sum_i |p_i> (h @ <p|psi>)_i
        vnl_psi = nl_data.projectors.T @ h_proj_psi  # (npw,) complex
        h_psi = h_psi + vnl_psi

    return h_psi


def compute_hartree_potential_g(rho_g: jnp.ndarray, g2: jnp.ndarray) -> jnp.ndarray:
    """Compute the Hartree potential in reciprocal space.

    V_H(G) = 4*pi * rho(G) / |G|^2  for G != 0
    V_H(G=0) = 0  (absorbed into pseudopotential / neutralizing background)

    Args:
        rho_g: (npw,) electron density in reciprocal space.
        g2: (npw,) |G|^2 values.

    Returns:
        (npw,) Hartree potential in reciprocal space.
    """
    g2_safe = jnp.where(g2 == 0.0, 1.0, g2)
    vh_g = FOUR_PI * rho_g / g2_safe
    vh_g = jnp.where(g2 == 0.0, 0.0, vh_g)
    return vh_g


def compute_hartree_energy(rho_g: jnp.ndarray, g2: jnp.ndarray, omega: float) -> jnp.ndarray:
    """Compute Hartree energy.

    E_H = (omega/2) * sum_G 4*pi * |rho(G)|^2 / |G|^2  (G != 0)

    Note: rho(G) here uses the convention rho(G) = (1/Omega) integral rho(r) e^{-iGr} dr

    Args:
        rho_g: (npw,) density Fourier coefficients (rho_G = 1/N_grid * sum_r rho(r) e^{-iGr}).
        g2: (npw,) |G|^2.
        omega: Cell volume.

    Returns:
        Hartree energy.
    """
    g2_safe = jnp.where(g2 == 0.0, 1.0, g2)
    eh = 0.5 * omega * FOUR_PI * jnp.sum(
        jnp.where(g2 == 0.0, 0.0, jnp.abs(rho_g)**2 / g2_safe)
    )
    return jnp.real(eh)


def build_local_potential_real(
    crystal: Crystal,
    rho_r: jnp.ndarray,
    g_vectors_fft: jnp.ndarray,
    fft_grid: tuple[int, int, int],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the total local potential on the real-space grid.

    V_loc_total(r) = V_ps_loc(r) + V_H(r) + V_xc(r)

    Args:
        crystal: Crystal structure.
        rho_r: (n1, n2, n3) electron density on real-space grid.
        g_vectors_fft: (n1, n2, n3, 3) G-vectors for the full FFT grid.
        fft_grid: FFT grid dimensions.

    Returns:
        vloc_r: (n1, n2, n3) total local potential.
        vh_r: (n1, n2, n3) Hartree potential.
        vxc_r: (n1, n2, n3) XC potential.
        vps_loc_r: (n1, n2, n3) local pseudopotential.
    """
    n1, n2, n3 = fft_grid
    n_grid = n1 * n2 * n3
    omega = float(crystal.volume)

    # 1. Compute rho in G-space
    rho_g = jnp.fft.fftn(rho_r) / n_grid  # Convention: rho(G) = (1/N) sum_r rho(r) e^{-iGr}

    # 2. G^2 on full FFT grid
    g2_fft = jnp.sum(g_vectors_fft**2, axis=-1)  # (n1, n2, n3)

    # 3. Hartree potential in G-space
    g2_safe = jnp.where(g2_fft == 0.0, 1.0, g2_fft)
    vh_g = FOUR_PI * rho_g / g2_safe
    vh_g = jnp.where(g2_fft == 0.0, 0.0, vh_g)
    vh_r = jnp.real(jnp.fft.ifftn(vh_g * n_grid))

    # 4. Local pseudopotential in G-space
    b = crystal.b
    sf = {}
    unique_species = list(set(crystal.species))
    for sp in unique_species:
        mask_arr = jnp.array([1.0 if s == sp else 0.0 for s in crystal.species])
        # Structure factor on FFT grid
        # S(G) = sum_i Z_i * exp(-i G . tau_i)
        g_flat = g_vectors_fft.reshape(-1, 3)  # (N, 3)
        phases = jnp.exp(-1j * crystal.positions @ g_flat.T)  # (natom, N)
        s_g = jnp.sum(mask_arr[:, None] * phases, axis=0)  # (N,)
        sf[sp] = s_g.reshape(fft_grid)

    vps_g = jnp.zeros(fft_grid, dtype=jnp.complex128)
    for sp in unique_species:
        params = get_hgh_params(sp)
        g2_flat = g2_fft.ravel()
        vloc_sp = vloc_reciprocal(g2_flat, params, omega)
        vps_g = vps_g + sf[sp] * vloc_sp.reshape(fft_grid)

    # The local PP is stored as V(G) * Omega in our convention; convert to potential
    # Actually vloc_reciprocal already includes the 1/Omega factor in the formula,
    # and multiplied by structure factor gives us the total V_loc(G).
    # To get V_loc(r), we need to inverse FFT.
    # V_loc(r) = sum_G V_loc(G) * e^{iGr} = N * IFFT[V_loc(G)]
    vps_loc_r = jnp.real(jnp.fft.ifftn(vps_g * n_grid))

    # 5. XC potential
    rho_real = jnp.maximum(jnp.real(rho_r), 1e-30)
    vxc_r = xc_potential(rho_real)

    # 6. Total local potential
    vloc_r = vps_loc_r + vh_r + vxc_r

    return vloc_r, vh_r, vxc_r, vps_loc_r
