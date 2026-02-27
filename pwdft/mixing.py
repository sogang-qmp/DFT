"""Density mixing schemes for SCF convergence.

Implements:
1. Simple (linear) mixing
2. Anderson/Pulay mixing (DIIS) with optional Kerker preconditioning
3. Kerker-preconditioned mixing
"""

import jax.numpy as jnp
import numpy as np


def simple_mixing(rho_in: jnp.ndarray, rho_out: jnp.ndarray,
                  alpha: float = 0.3) -> jnp.ndarray:
    """Simple linear mixing.

    rho_new = (1 - alpha) * rho_in + alpha * rho_out

    Args:
        rho_in: Input density.
        rho_out: Output density from diagonalization.
        alpha: Mixing parameter (0 < alpha <= 1).

    Returns:
        Mixed density.
    """
    return (1.0 - alpha) * rho_in + alpha * rho_out


class PulayMixer:
    """Pulay mixing (DIIS - Direct Inversion in the Iterative Subspace).

    Reference: P. Pulay, Chem. Phys. Lett. 73, 393 (1980).

    Stores history of input densities and residuals, and finds the
    optimal linear combination that minimizes the residual.
    Optionally applies Kerker preconditioning to suppress charge sloshing.
    """

    def __init__(self, max_hist: int = 8, alpha: float = 0.3,
                 use_kerker: bool = False, fft_grid: tuple[int, int, int] | None = None,
                 g2_fft: np.ndarray | None = None, kerker_q0: float = 1.5):
        """Initialize Pulay mixer.

        Args:
            max_hist: Maximum number of history entries.
            alpha: Linear mixing parameter.
            use_kerker: Whether to use Kerker preconditioning.
            fft_grid: FFT grid dimensions (needed for Kerker).
            g2_fft: |G|^2 on FFT grid (needed for Kerker).
            kerker_q0: Kerker screening wavevector.
        """
        self.max_hist = max_hist
        self.alpha = alpha
        self.use_kerker = use_kerker
        self.fft_grid = fft_grid
        self.g2_fft = g2_fft
        self.kerker_q0 = kerker_q0
        self.rho_history: list[np.ndarray] = []
        self.res_history: list[np.ndarray] = []
        self._iteration = 0

    def reset(self):
        """Clear history."""
        self.rho_history = []
        self.res_history = []
        self._iteration = 0

    def _apply_kerker(self, residual_flat: np.ndarray) -> np.ndarray:
        """Apply Kerker preconditioning to a residual.

        Damps long-wavelength components: K(G) = |G|^2 / (|G|^2 + q0^2).
        K(G=0) = 1 to preserve total charge.
        """
        if not self.use_kerker or self.fft_grid is None or self.g2_fft is None:
            return residual_flat

        res_g = np.fft.fftn(residual_flat.reshape(self.fft_grid))
        g2 = self.g2_fft
        q02 = self.kerker_q0 ** 2
        kernel = np.where(g2 == 0.0, 1.0, g2 / (g2 + q02))
        prec_g = res_g * kernel
        return np.real(np.fft.ifftn(prec_g)).ravel()

    def mix(self, rho_in: jnp.ndarray, rho_out: jnp.ndarray) -> jnp.ndarray:
        """Compute Pulay-mixed density.

        The DIIS is performed on the raw (unpreconditioned) residuals.
        Kerker preconditioning is applied only to the final mixing step.

        Args:
            rho_in: Input density (shaped array).
            rho_out: Output density from diagonalization.

        Returns:
            Mixed density with same shape as input.
        """
        self._iteration += 1
        shape = rho_in.shape
        rho_in_flat = np.array(rho_in.ravel(), dtype=np.float64)
        rho_out_flat = np.array(rho_out.ravel(), dtype=np.float64)
        residual = rho_out_flat - rho_in_flat

        self.rho_history.append(rho_in_flat.copy())
        self.res_history.append(residual.copy())

        # Trim history
        if len(self.rho_history) > self.max_hist:
            self.rho_history.pop(0)
            self.res_history.pop(0)

        n = len(self.rho_history)

        # First few iterations: simple (Kerker-preconditioned) mixing
        if n < 3:
            prec_res = self._apply_kerker(residual)
            result = rho_in_flat + self.alpha * prec_res
            return jnp.array(result.reshape(shape))

        # Build DIIS error matrix B_{ij} = <R_i | R_j>
        R = np.array(self.res_history)  # (n, N)
        B = R @ R.T  # (n, n)

        # Add Tikhonov regularization for stability
        reg = 1e-10 * np.trace(B) / n
        B_reg = B + reg * np.eye(n)

        # Solve constrained optimization
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = B_reg
        A[:n, n] = 1.0
        A[n, :n] = 1.0
        rhs = np.zeros(n + 1)
        rhs[n] = 1.0

        use_diis = True
        try:
            sol = np.linalg.solve(A, rhs)
            coeffs = sol[:n]
            # Reject wild extrapolation
            if np.max(np.abs(coeffs)) > 5.0:
                use_diis = False
        except np.linalg.LinAlgError:
            use_diis = False

        if not use_diis:
            # Anderson (2-point) fallback
            if n >= 2:
                dR = self.res_history[-1] - self.res_history[-2]
                dR_dot = np.dot(dR, dR)
                if dR_dot > 1e-30:
                    beta = np.clip(np.dot(self.res_history[-1], dR) / dR_dot, 0.0, 1.0)
                    rho_bar = (1 - beta) * self.rho_history[-1] + beta * self.rho_history[-2]
                    res_bar = (1 - beta) * self.res_history[-1] + beta * self.res_history[-2]
                    prec_res = self._apply_kerker(res_bar)
                    result = rho_bar + self.alpha * prec_res
                    return jnp.array(result.reshape(shape))
            prec_res = self._apply_kerker(residual)
            result = rho_in_flat + self.alpha * prec_res
            return jnp.array(result.reshape(shape))

        # DIIS optimal density:
        # rho_opt = sum_i c_i * rho_i, then mix with DIIS residual
        rho_opt = np.zeros_like(rho_in_flat)
        res_opt = np.zeros_like(rho_in_flat)
        for i in range(n):
            rho_opt += coeffs[i] * self.rho_history[i]
            res_opt += coeffs[i] * self.res_history[i]

        # Apply Kerker to the optimal residual and mix
        prec_res = self._apply_kerker(res_opt)
        result = rho_opt + self.alpha * prec_res

        return jnp.array(result.reshape(shape))


class KerkerPreconditioner:
    """Kerker preconditioner for density mixing.

    Suppresses long-wavelength charge sloshing in metallic systems.
    K(G) = |G|^2 / (|G|^2 + q0^2)

    Reference: G. P. Kerker, Phys. Rev. B 23, 3082 (1981).
    """

    def __init__(self, q0: float = 1.5):
        """
        Args:
            q0: Screening wavevector in 1/Bohr.
        """
        self.q0 = q0

    def precondition(self, residual_g: jnp.ndarray, g2: jnp.ndarray) -> jnp.ndarray:
        """Apply Kerker preconditioner in reciprocal space.

        Args:
            residual_g: Residual in reciprocal space.
            g2: |G|^2 values.

        Returns:
            Preconditioned residual.
        """
        kernel = g2 / (g2 + self.q0**2)
        # G=0 component: no preconditioning (or set to some value)
        kernel = jnp.where(g2 == 0.0, 1.0, kernel)
        return residual_g * kernel
