"""Iterative eigensolvers for the Kohn-Sham equations.

Implements:
1. Davidson iterative diagonalization
2. LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)
3. Direct diagonalization (for small systems / reference)
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from pwdft.hamiltonian import apply_hamiltonian, PWBasisData, NonlocalData


def direct_diagonalize(
    basis: PWBasisData,
    vloc_r: jnp.ndarray,
    nl_data: NonlocalData,
    n_bands: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve KS equations by explicit Hamiltonian construction and diagonalization.

    Only feasible for small basis sets (npw < ~2000).

    Args:
        basis: Plane-wave basis data.
        vloc_r: (n1, n2, n3) total local potential on real-space grid.
        nl_data: Nonlocal projector data.
        n_bands: Number of eigenvalues/vectors to compute.

    Returns:
        eigenvalues: (n_bands,) sorted eigenvalues.
        eigenvectors: (npw, n_bands) eigenvectors as columns.
    """
    npw = len(basis.kinetic)

    # Build full Hamiltonian matrix by applying H to basis vectors
    # H_{GG'} = <G|H|G'>
    eye = jnp.eye(npw, dtype=jnp.complex128)
    H = jax.vmap(lambda col: apply_hamiltonian(col, basis, vloc_r, nl_data))(eye)
    # H is (npw, npw) where H[i,:] = H|e_i>
    # We want H[i,j] = <e_i|H|e_j>, which is just H.T since H is Hermitian
    H = H.T

    # Symmetrize (should already be symmetric up to numerical noise)
    H = 0.5 * (H + H.conj().T)

    # Diagonalize
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)

    return eigenvalues[:n_bands], eigenvectors[:, :n_bands]


def davidson(
    basis: PWBasisData,
    vloc_r: jnp.ndarray,
    nl_data: NonlocalData,
    n_bands: int,
    psi_init: jnp.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
    max_subspace: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Davidson iterative diagonalization.

    Finds the n_bands lowest eigenvalues/eigenvectors of H.

    The algorithm iteratively expands a subspace by adding preconditioned
    residual vectors, then solves the projected eigenvalue problem.

    Args:
        basis: Plane-wave basis data.
        vloc_r: Local potential on real-space grid.
        nl_data: Nonlocal data.
        n_bands: Number of bands to compute.
        psi_init: (npw, n_bands) initial guess for eigenvectors. Random if None.
        max_iter: Maximum Davidson iterations.
        tol: Convergence tolerance on residual norm.
        max_subspace: Maximum subspace dimension before restart.

    Returns:
        eigenvalues: (n_bands,) lowest eigenvalues.
        eigenvectors: (npw, n_bands) corresponding eigenvectors.
    """
    npw = len(basis.kinetic)
    if max_subspace is None:
        max_subspace = min(4 * n_bands, npw)

    # Diagonal preconditioner (kinetic energy)
    # P = 1 / (T_G - epsilon) where epsilon is approximate eigenvalue
    T_diag = basis.kinetic

    # Initial subspace
    if psi_init is not None:
        V = psi_init  # (npw, n_bands)
    else:
        # Use random initial vectors
        key = jax.random.PRNGKey(42)
        V = jax.random.normal(key, (npw, n_bands), dtype=jnp.float64)
        V = V + 0j  # Make complex

    # Orthonormalize
    V = _orthonormalize(V)

    # Apply H to all basis vectors
    HV = jnp.stack([apply_hamiltonian(V[:, i], basis, vloc_r, nl_data)
                     for i in range(V.shape[1])], axis=1)

    eigenvalues = jnp.zeros(n_bands)

    for iteration in range(max_iter):
        n_sub = V.shape[1]

        # Projected Hamiltonian: H_sub = V^H @ H @ V
        H_sub = V.conj().T @ HV  # (n_sub, n_sub)
        H_sub = 0.5 * (H_sub + H_sub.conj().T)  # Ensure Hermitian

        # Solve projected eigenproblem
        evals, evecs = jnp.linalg.eigh(H_sub)

        # Ritz vectors and values
        eigenvalues = evals[:n_bands]
        ritz = V @ evecs[:, :n_bands]  # (npw, n_bands)
        h_ritz = HV @ evecs[:, :n_bands]  # (npw, n_bands)

        # Compute residuals: r_i = H|ritz_i> - epsilon_i * |ritz_i>
        residuals = h_ritz - ritz * eigenvalues[None, :]

        # Check convergence
        res_norms = jnp.linalg.norm(residuals, axis=0)
        max_res = jnp.max(res_norms)

        if float(max_res) < tol:
            return eigenvalues, ritz

        # Precondition residuals
        new_vecs = []
        for i in range(n_bands):
            # Diagonal preconditioning
            denom = T_diag - eigenvalues[i]
            denom = jnp.where(jnp.abs(denom) < 0.1, jnp.sign(denom) * 0.1 + 0j, denom)
            t = residuals[:, i] / denom
            new_vecs.append(t)

        new_V = jnp.stack(new_vecs, axis=1)  # (npw, n_bands)

        # Check if subspace is getting too large -> restart
        if n_sub + n_bands > max_subspace:
            # Restart with current Ritz vectors
            V = ritz
            HV = h_ritz
            # Orthonormalize
            V = _orthonormalize(V)
            HV = jnp.stack([apply_hamiltonian(V[:, i], basis, vloc_r, nl_data)
                           for i in range(V.shape[1])], axis=1)
        else:
            # Expand subspace
            # Orthogonalize new vectors against existing subspace
            new_V = new_V - V @ (V.conj().T @ new_V)
            new_V = _orthonormalize(new_V)

            if new_V.shape[1] == 0:
                # No new directions, converged or stuck
                break

            # Apply H to new vectors
            H_new = jnp.stack([apply_hamiltonian(new_V[:, i], basis, vloc_r, nl_data)
                              for i in range(new_V.shape[1])], axis=1)

            V = jnp.concatenate([V, new_V], axis=1)
            HV = jnp.concatenate([HV, H_new], axis=1)

    return eigenvalues, ritz


def _orthonormalize(V: jnp.ndarray, tol: float = 1e-12) -> jnp.ndarray:
    """Orthonormalize columns of V using modified Gram-Schmidt.

    Drops linearly dependent vectors.

    Args:
        V: (n, m) matrix.
        tol: Tolerance for detecting linear dependence.

    Returns:
        (n, m') orthonormal matrix with m' <= m.
    """
    n, m = V.shape
    Q_list = []

    for i in range(m):
        v = V[:, i]
        # Subtract projections onto existing Q vectors
        for q in Q_list:
            v = v - jnp.dot(q.conj(), v) * q
        norm = jnp.linalg.norm(v)
        if float(norm) > tol:
            Q_list.append(v / norm)

    if len(Q_list) == 0:
        return jnp.zeros((n, 0), dtype=V.dtype)
    return jnp.stack(Q_list, axis=1)


def lobpcg(
    basis: PWBasisData,
    vloc_r: jnp.ndarray,
    nl_data: NonlocalData,
    n_bands: int,
    psi_init: jnp.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """LOBPCG eigensolver.

    Locally Optimal Block Preconditioned Conjugate Gradient method.
    Reference: A. V. Knyazev, SIAM J. Sci. Comput. 23, 517 (2001).

    Args:
        basis: Plane-wave basis data.
        vloc_r: Local potential on real-space grid.
        nl_data: Nonlocal data.
        n_bands: Number of bands.
        psi_init: Initial guess.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        eigenvalues: (n_bands,)
        eigenvectors: (npw, n_bands)
    """
    npw = len(basis.kinetic)
    T_diag = basis.kinetic

    # Initialize
    if psi_init is not None:
        X = psi_init
    else:
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (npw, n_bands), dtype=jnp.float64) + 0j

    X = _orthonormalize(X)

    # Apply H
    AX = jnp.stack([apply_hamiltonian(X[:, i], basis, vloc_r, nl_data)
                    for i in range(X.shape[1])], axis=1)

    # Rayleigh quotient
    eigenvalues = jnp.real(jnp.sum(X.conj() * AX, axis=0))

    P = None  # Previous search directions

    for iteration in range(max_iter):
        # Residuals
        R = AX - X * eigenvalues[None, :]
        res_norms = jnp.linalg.norm(R, axis=0)

        if float(jnp.max(res_norms)) < tol:
            return eigenvalues, X

        # Precondition: approximate inverse of (T - epsilon)
        W = jnp.zeros_like(R)
        for i in range(n_bands):
            denom = T_diag - eigenvalues[i]
            denom = jnp.where(jnp.abs(denom) < 0.1, jnp.sign(denom) * 0.1 + 0j, denom)
            W = W.at[:, i].set(R[:, i] / denom)

        # Apply H to preconditioned residuals
        AW = jnp.stack([apply_hamiltonian(W[:, i], basis, vloc_r, nl_data)
                        for i in range(W.shape[1])], axis=1)

        # Build trial subspace [X, W, P]
        if P is not None:
            S = jnp.concatenate([X, W, P], axis=1)
            AS = jnp.concatenate([AX, AW, AP], axis=1)
        else:
            S = jnp.concatenate([X, W], axis=1)
            AS = jnp.concatenate([AX, AW], axis=1)

        # Orthonormalize S
        S_orth = _orthonormalize(S)
        n_orth = S_orth.shape[1]
        if n_orth < n_bands:
            # Something went wrong, fall back
            return eigenvalues, X

        # Recompute AS for orthonormalized basis
        AS_orth = jnp.stack([apply_hamiltonian(S_orth[:, i], basis, vloc_r, nl_data)
                            for i in range(n_orth)], axis=1)

        # Projected eigenproblem
        H_sub = S_orth.conj().T @ AS_orth
        H_sub = 0.5 * (H_sub + H_sub.conj().T)
        evals, evecs = jnp.linalg.eigh(H_sub)

        eigenvalues = evals[:n_bands]
        X_new = S_orth @ evecs[:, :n_bands]
        AX_new = AS_orth @ evecs[:, :n_bands]

        # Update search directions
        P = X_new - X
        AP = AX_new - AX

        X = X_new
        AX = AX_new

    return eigenvalues, X
