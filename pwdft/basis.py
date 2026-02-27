"""Plane-wave basis set generation and utilities."""

import jax
import jax.numpy as jnp
import numpy as np

from pwdft.lattice import reciprocal_lattice, cell_volume


def get_g_vectors(a: jnp.ndarray, ecut: float) -> tuple[jnp.ndarray, jnp.ndarray, tuple[int, int, int]]:
    """Generate the set of G-vectors satisfying |G|^2/2 <= ecut.

    The kinetic energy cutoff determines the plane-wave basis:
        (1/2)|k + G|^2 <= Ecut
    For G-vector generation (independent of k), we use
        |G|^2/2 <= Ecut (with some margin to accommodate all k-points).

    We use a regular FFT grid determined by the cutoff.

    Args:
        a: (3, 3) lattice vectors (rows) in Bohr.
        ecut: Plane-wave kinetic energy cutoff in Hartree.

    Returns:
        g_vectors: (npw, 3) G-vectors in Cartesian coords (1/Bohr).
        g_indices: (npw, 3) integer Miller indices of G-vectors.
        fft_grid: (n1, n2, n3) FFT grid dimensions.
    """
    b = reciprocal_lattice(a)

    # Determine FFT grid dimensions from cutoff.
    # |G_max| = sqrt(2*Ecut), and G = n1*b1 + n2*b2 + n3*b3
    # We need n_i >= G_max / |b_i| but we use a safer estimate.
    g_max = jnp.sqrt(2.0 * ecut)
    b_lengths = jnp.linalg.norm(b, axis=1)

    # Number of G-vectors along each direction (ensure odd for symmetry)
    n_grid = np.array(2 * np.ceil(float(g_max) / np.array(b_lengths)) + 1, dtype=int)
    # Make compatible with FFT: use next efficient FFT size
    n_grid = np.array([_next_fft_size(int(n)) for n in n_grid])
    fft_grid = tuple(int(n) for n in n_grid)

    # Generate all integer triplets
    n1, n2, n3 = fft_grid
    # Use numpy for index generation (static), then convert
    i1 = np.fft.fftfreq(n1, d=1.0) * n1
    i2 = np.fft.fftfreq(n2, d=1.0) * n2
    i3 = np.fft.fftfreq(n3, d=1.0) * n3
    m1, m2, m3 = np.meshgrid(i1, i2, i3, indexing='ij')
    miller = np.stack([m1.ravel(), m2.ravel(), m3.ravel()], axis=-1).astype(int)

    # Compute G = miller @ b  (each row of b is a reciprocal vector)
    b_np = np.array(b)
    g_cart = miller.astype(float) @ b_np

    # Filter by cutoff: |G|^2/2 <= ecut (with small tolerance)
    g2 = np.sum(g_cart**2, axis=1)
    mask = g2 <= 2.0 * ecut * (1.0 + 1e-8)

    g_indices_out = jnp.array(miller[mask])
    g_vectors_out = jnp.array(g_cart[mask])

    return g_vectors_out, g_indices_out, fft_grid


def get_g_vectors_fft(fft_grid: tuple[int, int, int], b: jnp.ndarray) -> jnp.ndarray:
    """Get G-vectors for the full FFT grid (used in real-space operations).

    Returns G-vectors indexed in standard FFT order.

    Args:
        fft_grid: (n1, n2, n3) FFT grid dimensions.
        b: (3, 3) reciprocal lattice vectors (rows).

    Returns:
        (n1, n2, n3, 3) array of G-vectors.
    """
    n1, n2, n3 = fft_grid
    freq1 = jnp.fft.fftfreq(n1, d=1.0) * n1
    freq2 = jnp.fft.fftfreq(n2, d=1.0) * n2
    freq3 = jnp.fft.fftfreq(n3, d=1.0) * n3
    m1, m2, m3 = jnp.meshgrid(freq1, freq2, freq3, indexing='ij')
    miller = jnp.stack([m1, m2, m3], axis=-1)  # (n1, n2, n3, 3)
    # G = miller @ b
    g_grid = jnp.einsum('...i,ij->...j', miller, b)
    return g_grid


def g_vector_norms_sq(g_vectors: jnp.ndarray) -> jnp.ndarray:
    """Compute |G|^2 for each G-vector.

    Args:
        g_vectors: (..., 3) G-vectors.

    Returns:
        (...,) squared norms.
    """
    return jnp.sum(g_vectors**2, axis=-1)


def kinetic_energies(g_vectors: jnp.ndarray, k: jnp.ndarray) -> jnp.ndarray:
    """Compute kinetic energies (1/2)|k + G|^2 for a given k-point.

    Args:
        g_vectors: (npw, 3) G-vectors in Cartesian coords.
        k: (3,) k-point in Cartesian coords.

    Returns:
        (npw,) kinetic energies.
    """
    kg = g_vectors + k[None, :]
    return 0.5 * jnp.sum(kg**2, axis=-1)


def _next_fft_size(n: int) -> int:
    """Find next integer >= n that factors only into 2, 3, 5 (efficient FFT size)."""
    if n <= 1:
        return 1
    while True:
        m = n
        for p in [2, 3, 5]:
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def pw_to_real(coeffs: jnp.ndarray, g_indices: jnp.ndarray, fft_grid: tuple[int, int, int]) -> jnp.ndarray:
    """Convert plane-wave coefficients to real-space on FFT grid.

    Places coefficients c_G at their FFT-grid positions and performs inverse FFT.

    Args:
        coeffs: (npw,) complex plane-wave coefficients.
        g_indices: (npw, 3) integer Miller indices.
        fft_grid: (n1, n2, n3) FFT grid dimensions.

    Returns:
        (n1, n2, n3) complex array on real-space grid.
    """
    n1, n2, n3 = fft_grid
    n_total = n1 * n2 * n3
    # Map indices to positive range for FFT grid
    idx = g_indices % jnp.array([n1, n2, n3])
    flat_idx = idx[:, 0] * (n2 * n3) + idx[:, 1] * n3 + idx[:, 2]
    grid_flat = jnp.zeros(n_total, dtype=jnp.complex128).at[flat_idx].set(coeffs)
    grid = grid_flat.reshape(fft_grid)
    # Inverse FFT with correct normalization: sum_G c_G e^{iG.r}
    # jnp.fft.ifftn includes 1/N factor, so multiply by N
    return jnp.fft.ifftn(grid) * n_total


def real_to_pw(field: jnp.ndarray, g_indices: jnp.ndarray, fft_grid: tuple[int, int, int]) -> jnp.ndarray:
    """Convert real-space field to plane-wave coefficients.

    Args:
        field: (n1, n2, n3) real-space field on FFT grid.
        g_indices: (npw, 3) integer Miller indices.
        fft_grid: (n1, n2, n3) FFT grid dimensions.

    Returns:
        (npw,) complex plane-wave coefficients.
    """
    n1, n2, n3 = fft_grid
    # Forward FFT
    field_g = jnp.fft.fftn(field) / (n1 * n2 * n3)
    # Extract coefficients at G-vector positions
    idx = g_indices % jnp.array([n1, n2, n3])
    return field_g[idx[:, 0], idx[:, 1], idx[:, 2]]
