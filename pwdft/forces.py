"""Atomic forces via the Hellmann-Feynman theorem.

F_a = -dE/d(tau_a) = F_local + F_nonlocal + F_ewald

The local force arises from the local pseudopotential:
    F_a^loc = -sum_G i*G * V_loc^a(G) * S_a(G)^* * rho(G) * Omega

The nonlocal force arises from the Kleinman-Bylander projectors:
    F_a^nl = -sum_{k,n} w_k * f_{nk} * sum_{l,m,i,j}
             h^l_{ij} * 2*Re[ <psi|dp_i/dtau> <p_j|psi> ]

The Ewald force is the derivative of the Ewald energy.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pwdft.constants import PI, TWO_PI, FOUR_PI
from pwdft.crystal import Crystal
from pwdft.lattice import reciprocal_lattice, cell_volume
from pwdft.basis import get_g_vectors_fft
from pwdft.pseudopotential import get_hgh_params, vloc_reciprocal


def hellmann_feynman_local_forces(
    crystal: Crystal,
    rho_r: jnp.ndarray,
    fft_grid: tuple[int, int, int],
) -> jnp.ndarray:
    """Compute local pseudopotential contribution to atomic forces.

    F_a^loc_alpha = -Omega * sum_G i*G_alpha * V_loc^a(G) * exp(iG.tau_a) * rho(-G)

    Since rho is real, rho(-G) = rho(G)*.

    Args:
        crystal: Crystal structure.
        rho_r: (n1, n2, n3) electron density.
        fft_grid: FFT grid dimensions.

    Returns:
        (natom, 3) local forces in Hartree/Bohr.
    """
    n1, n2, n3 = fft_grid
    n_grid = n1 * n2 * n3
    omega = float(crystal.volume)
    b = crystal.b

    # Density in G-space
    rho_g = jnp.fft.fftn(rho_r) / n_grid  # (n1, n2, n3) complex

    # G-vectors on FFT grid
    g_fft = get_g_vectors_fft(fft_grid, b)  # (n1, n2, n3, 3)
    g2_fft = jnp.sum(g_fft**2, axis=-1)    # (n1, n2, n3)

    forces = jnp.zeros((crystal.natom, 3))

    for ia in range(crystal.natom):
        sp = crystal.species[ia]
        tau = crystal.positions[ia]
        params = get_hgh_params(sp)

        # V_loc(G) for this species
        vloc_g = vloc_reciprocal(g2_fft.ravel(), params, omega).reshape(fft_grid)

        # Phase factor: exp(i G . tau)
        phase = jnp.exp(1j * jnp.einsum('...i,i->...', g_fft, tau))

        # Force: -Omega * sum_G i*G * V_loc(G) * exp(iG.tau) * rho(-G)
        # rho(-G) = rho_g^* (since rho(r) is real)
        for alpha in range(3):
            f_alpha = -omega * jnp.sum(
                1j * g_fft[..., alpha] * vloc_g * phase * rho_g.conj()
            ) * n_grid
            forces = forces.at[ia, alpha].set(jnp.real(f_alpha))

    return forces


def ewald_forces(
    a: jnp.ndarray,
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    eta: float | None = None,
) -> jnp.ndarray:
    """Compute Ewald contribution to atomic forces.

    F_a = F_a^real + F_a^recip

    Args:
        a: (3, 3) lattice vectors.
        positions: (natom, 3) atomic positions.
        charges: (natom,) ionic charges.
        eta: Ewald parameter.

    Returns:
        (natom, 3) Ewald forces.
    """
    natom = len(charges)
    omega = cell_volume(a)
    b = reciprocal_lattice(a)

    if eta is None:
        eta = float((natom * PI**3 / omega**2) ** (1.0 / 3.0))
        eta = max(eta, 0.1)

    sqrt_eta = np.sqrt(eta)
    rcut = 4.5 / sqrt_eta

    a_np = np.array(a)
    a_lengths = np.linalg.norm(a_np, axis=1)
    nmax = np.ceil(rcut / a_lengths).astype(int) + 1

    forces = jnp.zeros((natom, 3))

    # Real-space forces
    ns = []
    for n1 in range(-nmax[0], nmax[0] + 1):
        for n2 in range(-nmax[1], nmax[1] + 1):
            for n3 in range(-nmax[2], nmax[2] + 1):
                ns.append([n1, n2, n3])
    ns = jnp.array(ns, dtype=jnp.float64)
    R_vecs = ns @ a

    for ia in range(natom):
        f_real = jnp.zeros(3)
        for ja in range(natom):
            d_ij = positions[ia] - positions[ja]
            d_all = d_ij[None, :] - R_vecs
            dists = jnp.linalg.norm(d_all, axis=-1)

            is_self = (ia == ja) & (jnp.sum(ns**2, axis=-1) == 0)
            dists_safe = jnp.where(is_self | (dists < 1e-15), 1.0, dists)

            # d/d(tau_i) erfc(eta*r)/r = -(d_ij/r) * (2*eta/sqrt(pi)*exp(-eta^2*r^2) + erfc(eta*r)/r) / r
            exp_term = jnp.exp(-eta * dists_safe**2)
            erfc_term = jax.scipy.special.erfc(sqrt_eta * dists_safe)
            factor = (2.0 * sqrt_eta / jnp.sqrt(PI) * exp_term + erfc_term / dists_safe) / dists_safe**2

            contrib = charges[ja] * jnp.sum(
                jnp.where((is_self | (dists > rcut))[:, None], 0.0,
                          d_all / dists_safe[:, None] * factor[:, None]),
                axis=0
            )
            f_real -= charges[ia] * contrib

        forces = forces.at[ia].set(forces[ia] + f_real)

    # Reciprocal-space forces
    b_np = np.array(b)
    b_lengths = np.linalg.norm(b_np, axis=1)
    gcut = 4.5 * sqrt_eta
    gmax = np.ceil(gcut / b_lengths).astype(int) + 1

    gs = []
    for m1 in range(-gmax[0], gmax[0] + 1):
        for m2 in range(-gmax[1], gmax[1] + 1):
            for m3 in range(-gmax[2], gmax[2] + 1):
                if m1 == 0 and m2 == 0 and m3 == 0:
                    continue
                gs.append([m1, m2, m3])

    if len(gs) > 0:
        gs = jnp.array(gs, dtype=jnp.float64)
        G_vecs = gs @ b
        G2 = jnp.sum(G_vecs**2, axis=-1)

        for ia in range(natom):
            f_recip = jnp.zeros(3)
            tau_i = positions[ia]

            for ja in range(natom):
                tau_j = positions[ja]
                d_ij = tau_i - tau_j
                # Reciprocal force contribution
                phase = jnp.sin(G_vecs @ d_ij)
                kernel = jnp.exp(-G2 / (4.0 * eta)) / G2
                f_recip += charges[ja] * jnp.sum(
                    G_vecs * (phase * kernel)[:, None], axis=0
                )

            forces = forces.at[ia].set(
                forces[ia] - charges[ia] * FOUR_PI / float(omega) * f_recip
            )

    return forces
