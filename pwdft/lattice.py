"""Lattice and reciprocal space utilities."""

import jax
import jax.numpy as jnp
from pwdft.constants import TWO_PI


def reciprocal_lattice(a: jnp.ndarray) -> jnp.ndarray:
    """Compute reciprocal lattice vectors b from real-space lattice vectors a.

    Given lattice vectors as rows of a (3x3 matrix), computes
    b such that a_i . b_j = 2*pi*delta_ij.

    The reciprocal vectors are returned as rows of b.

    Args:
        a: (3, 3) array of real-space lattice vectors (rows).

    Returns:
        (3, 3) array of reciprocal lattice vectors (rows).
    """
    # b = 2*pi * (a^{-T})
    return TWO_PI * jnp.linalg.inv(a).T


def cell_volume(a: jnp.ndarray) -> jnp.ndarray:
    """Compute unit cell volume from lattice vectors.

    Args:
        a: (3, 3) array of lattice vectors (rows).

    Returns:
        Scalar cell volume.
    """
    return jnp.abs(jnp.linalg.det(a))


def metric_tensor(a: jnp.ndarray) -> jnp.ndarray:
    """Compute the metric tensor g_ij = a_i . a_j.

    Args:
        a: (3, 3) array of lattice vectors (rows).

    Returns:
        (3, 3) metric tensor.
    """
    return a @ a.T


def fractional_to_cartesian(frac_coords: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """Convert fractional coordinates to Cartesian.

    Args:
        frac_coords: (..., 3) fractional coordinates.
        a: (3, 3) lattice vectors (rows).

    Returns:
        (..., 3) Cartesian coordinates.
    """
    return frac_coords @ a


def cartesian_to_fractional(cart_coords: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    """Convert Cartesian coordinates to fractional.

    Args:
        cart_coords: (..., 3) Cartesian coordinates.
        a: (3, 3) lattice vectors (rows).

    Returns:
        (..., 3) fractional coordinates.
    """
    return cart_coords @ jnp.linalg.inv(a)
