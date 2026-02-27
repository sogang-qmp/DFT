"""Ewald summation for the ion-ion electrostatic energy.

Computes the Madelung energy for a periodic system of point charges:
    E_ii = (1/2) sum_{i,j,R}' Z_i * Z_j / |tau_i - tau_j - R|

The prime means exclude i=j when R=0.

The Ewald method splits this into short-range (real-space) and
long-range (reciprocal-space) contributions plus a self-energy correction.
"""

import jax
import jax.numpy as jnp
import numpy as np

from pwdft.constants import PI, TWO_PI, FOUR_PI
from pwdft.lattice import reciprocal_lattice, cell_volume


def ewald_energy(
    a: jnp.ndarray,
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    eta: float | None = None,
    rcut_factor: float = 4.5,
    gcut_factor: float = 4.5,
) -> jnp.ndarray:
    """Compute Ewald ion-ion energy.

    E_ewald = E_real + E_recip + E_self + E_charged

    Args:
        a: (3, 3) lattice vectors (rows) in Bohr.
        positions: (natom, 3) Cartesian positions in Bohr.
        charges: (natom,) ionic charges (Z_ion).
        eta: Ewald splitting parameter. If None, chosen optimally.
        rcut_factor: Real-space cutoff in units of 1/sqrt(eta).
        gcut_factor: Reciprocal-space cutoff in units of sqrt(eta).

    Returns:
        Scalar Ewald energy in Hartree.
    """
    natom = len(charges)
    omega = cell_volume(a)
    b = reciprocal_lattice(a)

    # Choose optimal eta if not specified
    if eta is None:
        # Optimal: eta ~ (natom * pi^3 / omega^2)^(1/3)  (rough heuristic)
        eta = float((natom * PI**3 / omega**2) ** (1.0 / 3.0))
        eta = max(eta, 0.1)  # ensure not too small

    sqrt_eta = np.sqrt(eta)
    rcut = rcut_factor / sqrt_eta
    gcut = gcut_factor * sqrt_eta

    # ========================================
    # Real-space sum
    # ========================================
    # Find all lattice vectors R within rcut + max interatomic distance
    a_np = np.array(a)
    # Determine range of lattice vector indices
    a_lengths = np.linalg.norm(a_np, axis=1)
    nmax = np.ceil(rcut / a_lengths).astype(int) + 1

    e_real = 0.0
    for n1 in range(-nmax[0], nmax[0] + 1):
        for n2 in range(-nmax[1], nmax[1] + 1):
            for n3 in range(-nmax[2], nmax[2] + 1):
                R = jnp.array(n1 * a_np[0] + n2 * a_np[1] + n3 * a_np[2])
                for i in range(natom):
                    for j in range(natom):
                        if n1 == 0 and n2 == 0 and n3 == 0 and i == j:
                            continue
                        d = positions[i] - positions[j] - R
                        dist = jnp.linalg.norm(d)
                        if float(dist) < rcut:
                            e_real += charges[i] * charges[j] * jax.scipy.special.erfc(sqrt_eta * dist) / dist

    e_real = 0.5 * e_real

    # ========================================
    # Reciprocal-space sum
    # ========================================
    b_np = np.array(b)
    b_lengths = np.linalg.norm(b_np, axis=1)
    gmax = np.ceil(gcut / b_lengths).astype(int) + 1

    e_recip = 0.0
    for m1 in range(-gmax[0], gmax[0] + 1):
        for m2 in range(-gmax[1], gmax[1] + 1):
            for m3 in range(-gmax[2], gmax[2] + 1):
                if m1 == 0 and m2 == 0 and m3 == 0:
                    continue
                G = jnp.array(m1 * b_np[0] + m2 * b_np[1] + m3 * b_np[2])
                G2 = jnp.dot(G, G)
                if float(G2) > (2.0 * gcut)**2:
                    continue
                # Structure factor
                S_G = jnp.sum(charges * jnp.exp(-1j * jnp.dot(positions, G)))
                e_recip += float(jnp.abs(S_G)**2) * jnp.exp(-G2 / (4.0 * eta)) / G2

    e_recip = e_recip * TWO_PI / omega

    # ========================================
    # Self-energy correction
    # ========================================
    e_self = -jnp.sqrt(eta / PI) * jnp.sum(charges**2)

    # ========================================
    # Charged system correction (neutralizing background)
    # ========================================
    total_charge = jnp.sum(charges)
    e_charged = -PI * total_charge**2 / (2.0 * omega * eta)

    return e_real + e_recip + e_self + e_charged


def ewald_energy_fast(
    a: jnp.ndarray,
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    fft_grid: tuple[int, int, int] | None = None,
    eta: float | None = None,
) -> jnp.ndarray:
    """Compute Ewald energy using reciprocal-space FFT approach.

    This is faster for larger systems as it avoids explicit double loops.
    Uses the same decomposition but evaluates the reciprocal sum via FFT.

    Args:
        a: (3, 3) lattice vectors (rows).
        positions: (natom, 3) Cartesian positions.
        charges: (natom,) ionic charges.
        fft_grid: FFT grid dimensions. If None, auto-determined.
        eta: Ewald parameter.

    Returns:
        Ewald energy in Hartree.
    """
    natom = len(charges)
    omega = cell_volume(a)
    b = reciprocal_lattice(a)

    if eta is None:
        eta = float((natom * PI**3 / omega**2) ** (1.0 / 3.0))
        eta = max(eta, 0.1)

    if fft_grid is None:
        # Use a coarse grid sufficient for Ewald
        gcut = 5.0 * jnp.sqrt(eta)
        b_lengths = jnp.linalg.norm(b, axis=1)
        n_grid = np.array(2 * np.ceil(np.array(gcut / b_lengths)) + 1, dtype=int)
        fft_grid = tuple(int(max(n, 5)) for n in n_grid)

    n1, n2, n3 = fft_grid

    # Build G-vectors on FFT grid
    freq1 = jnp.fft.fftfreq(n1, d=1.0) * n1
    freq2 = jnp.fft.fftfreq(n2, d=1.0) * n2
    freq3 = jnp.fft.fftfreq(n3, d=1.0) * n3
    m1, m2, m3 = jnp.meshgrid(freq1, freq2, freq3, indexing='ij')
    miller = jnp.stack([m1, m2, m3], axis=-1)
    G_grid = jnp.einsum('...i,ij->...j', miller, b)
    G2 = jnp.sum(G_grid**2, axis=-1)

    # Structure factor on grid
    # S(G) = sum_i Z_i * exp(-i * G . tau_i)
    # positions: (natom, 3), G_grid: (n1, n2, n3, 3)
    phase = jnp.einsum('ai,klmi->aklm', positions, G_grid)
    S_G = jnp.sum(charges[:, None, None, None] * jnp.exp(-1j * phase), axis=0)

    # Reciprocal-space sum (exclude G=0)
    G2_safe = jnp.where(G2 == 0.0, 1.0, G2)
    kernel = FOUR_PI * jnp.exp(-G2_safe / (4.0 * eta)) / G2_safe / (2.0 * omega)
    kernel = jnp.where(G2 == 0.0, 0.0, kernel)

    e_recip = jnp.real(jnp.sum(jnp.abs(S_G)**2 * kernel))

    # Self energy
    e_self = -jnp.sqrt(eta / PI) * jnp.sum(charges**2)

    # Charged background
    total_charge = jnp.sum(charges)
    e_charged = -PI * total_charge**2 / (2.0 * omega * eta)

    # Real-space sum (for small eta, short-ranged)
    sqrt_eta = jnp.sqrt(eta)
    a_np = np.array(a)
    a_lengths = np.linalg.norm(a_np, axis=1)
    rcut = 4.5 / float(sqrt_eta)
    nmax = np.ceil(rcut / a_lengths).astype(int) + 1

    # Vectorized real-space sum
    ns = []
    for n1r in range(-nmax[0], nmax[0] + 1):
        for n2r in range(-nmax[1], nmax[1] + 1):
            for n3r in range(-nmax[2], nmax[2] + 1):
                ns.append([n1r, n2r, n3r])
    ns = jnp.array(ns, dtype=jnp.float64)  # (nR, 3)
    R_vecs = ns @ a  # (nR, 3)

    e_real = jnp.float64(0.0)
    for i in range(natom):
        for j in range(natom):
            d_ij = positions[i] - positions[j]  # (3,)
            d_all = d_ij[None, :] - R_vecs  # (nR, 3)
            dists = jnp.linalg.norm(d_all, axis=-1)  # (nR,)
            # Exclude self-interaction (i==j, R=0)
            is_self = (i == j) & (jnp.sum(ns**2, axis=-1) == 0)
            dists_safe = jnp.where(is_self, 1.0, dists)
            dists_safe = jnp.where(dists_safe < 1e-15, 1.0, dists_safe)
            contrib = jax.scipy.special.erfc(sqrt_eta * dists_safe) / dists_safe
            contrib = jnp.where(is_self, 0.0, contrib)
            contrib = jnp.where(dists > rcut, 0.0, contrib)
            e_real += charges[i] * charges[j] * jnp.sum(contrib)

    e_real = 0.5 * e_real

    return e_real + e_recip + e_self + e_charged
