"""Crystal structure definition."""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np

from pwdft.constants import ANGSTROM_TO_BOHR
from pwdft.lattice import reciprocal_lattice, cell_volume, fractional_to_cartesian


@dataclass
class Crystal:
    """Represents a periodic crystal structure.

    All internal quantities are stored in atomic units (Bohr).

    Attributes:
        a: (3, 3) real-space lattice vectors as rows, in Bohr.
        species: List of element symbols, one per atom.
        positions: (natom, 3) Cartesian atomic positions in Bohr.
        Z_vals: (natom,) valence charges for each atom.
    """
    a: jnp.ndarray
    species: list[str]
    positions: jnp.ndarray
    Z_vals: jnp.ndarray

    @classmethod
    def from_angstrom(
        cls,
        lattice_vectors: np.ndarray,
        species: list[str],
        positions: np.ndarray,
        Z_vals: np.ndarray,
        coords_are_fractional: bool = False,
    ) -> "Crystal":
        """Create a Crystal from quantities given in Angstroms.

        Args:
            lattice_vectors: (3, 3) lattice vectors in Angstroms (rows).
            species: List of element symbols.
            positions: (natom, 3) positions in Angstroms (or fractional if flag set).
            Z_vals: (natom,) valence charges.
            coords_are_fractional: If True, positions are fractional coordinates.

        Returns:
            Crystal instance in atomic units.
        """
        a = jnp.array(lattice_vectors, dtype=jnp.float64) * ANGSTROM_TO_BOHR
        Z_vals = jnp.array(Z_vals, dtype=jnp.float64)

        if coords_are_fractional:
            frac = jnp.array(positions, dtype=jnp.float64)
            cart = fractional_to_cartesian(frac, a)
        else:
            cart = jnp.array(positions, dtype=jnp.float64) * ANGSTROM_TO_BOHR

        return cls(a=a, species=species, positions=cart, Z_vals=Z_vals)

    @property
    def natom(self) -> int:
        return len(self.species)

    @property
    def nelec(self) -> float:
        return float(jnp.sum(self.Z_vals))

    @property
    def b(self) -> jnp.ndarray:
        """Reciprocal lattice vectors (rows), in 1/Bohr."""
        return reciprocal_lattice(self.a)

    @property
    def volume(self) -> jnp.ndarray:
        """Unit cell volume in Bohr^3."""
        return cell_volume(self.a)

    @property
    def fractional_positions(self) -> jnp.ndarray:
        """Atomic positions in fractional coordinates."""
        return self.positions @ jnp.linalg.inv(self.a)
