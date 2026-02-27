"""K-point generation and symmetry utilities."""

import jax.numpy as jnp
import numpy as np

from pwdft.lattice import reciprocal_lattice


def monkhorst_pack(nk: tuple[int, int, int], b: jnp.ndarray,
                   shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
                   ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a Monkhorst-Pack k-point grid.

    k_{n1,n2,n3} = ((2*n_i - N_i - 1) / (2*N_i) + shift_i) * b_i

    Reference: H. J. Monkhorst, J. D. Pack, Phys. Rev. B 13, 5188 (1976).

    Args:
        nk: (nk1, nk2, nk3) k-point grid dimensions.
        b: (3, 3) reciprocal lattice vectors (rows).
        shift: (s1, s2, s3) optional shift in fractional reciprocal coordinates.

    Returns:
        kpoints: (nk_total, 3) k-points in Cartesian reciprocal coordinates.
        weights: (nk_total,) integration weights (sum to 1).
    """
    nk1, nk2, nk3 = nk
    total = nk1 * nk2 * nk3

    # Generate fractional k-point coordinates
    i1 = np.arange(nk1)
    i2 = np.arange(nk2)
    i3 = np.arange(nk3)

    f1 = (2.0 * i1 - nk1 + 1) / (2.0 * nk1) + shift[0]
    f2 = (2.0 * i2 - nk2 + 1) / (2.0 * nk2) + shift[1]
    f3 = (2.0 * i3 - nk3 + 1) / (2.0 * nk3) + shift[2]

    g1, g2, g3 = np.meshgrid(f1, f2, f3, indexing='ij')
    frac_kpts = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=-1)

    # Convert to Cartesian: k = frac @ b
    kpoints = jnp.array(frac_kpts) @ b
    weights = jnp.ones(total) / total

    return kpoints, weights


def gamma_point(b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return just the Gamma point.

    Args:
        b: (3, 3) reciprocal lattice vectors.

    Returns:
        kpoints: (1, 3) Gamma point.
        weights: (1,) weight = 1.0.
    """
    return jnp.zeros((1, 3)), jnp.ones(1)


def reduce_kpoints_time_reversal(kpoints: jnp.ndarray, weights: jnp.ndarray,
                                  b: jnp.ndarray, tol: float = 1e-8
                                  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Reduce k-point set using time-reversal symmetry: k ~ -k.

    For each pair (k, -k), keep one and double its weight.

    Args:
        kpoints: (nk, 3) k-points in Cartesian coords.
        weights: (nk,) weights.
        b: (3, 3) reciprocal lattice vectors.
        tol: Tolerance for comparing k-points.

    Returns:
        Reduced kpoints and weights.
    """
    # Work in fractional coordinates for cleaner comparison
    b_inv = jnp.linalg.inv(b)
    frac = np.array(kpoints @ b_inv)
    weights_np = np.array(weights)

    kept = []
    new_weights = []
    used = set()

    for i in range(len(frac)):
        if i in used:
            continue
        # Find if -k is in the set (modulo reciprocal lattice vector)
        neg_k = -frac[i]
        found_pair = False
        for j in range(i + 1, len(frac)):
            if j in used:
                continue
            diff = neg_k - frac[j]
            # Reduce to first BZ
            diff = diff - np.round(diff)
            if np.linalg.norm(diff) < tol:
                found_pair = True
                used.add(j)
                break
        kept.append(i)
        used.add(i)
        if found_pair:
            new_weights.append(weights_np[i] + weights_np[j])
        else:
            new_weights.append(weights_np[i])

    kept = np.array(kept)
    return jnp.array(np.array(kpoints)[kept]), jnp.array(new_weights)
