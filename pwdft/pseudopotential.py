"""Hartwigsen-Goedecker-Hutter (HGH) norm-conserving pseudopotentials.

Reference:
    C. Hartwigsen, S. Goedecker, J. Hutter, Phys. Rev. B 58, 3641 (1998).

The HGH pseudopotential has the form in reciprocal space:
    V_ps(G, G') = V_loc(G) * delta_{G,G'} + V_nl(G, G')

Local part (in reciprocal space, analytical Fourier transform):
    V_loc(G) = -Z_ion/Omega * 4*pi/|G|^2 * exp(-|G|^2 * r_loc^2 / 2)
             + sqrt(8*pi^3) * r_loc^3 / Omega * exp(-|G|^2 * r_loc^2 / 2)
             * (C1 + C2*(3 - |G|^2*r_loc^2)
             + C3*(15 - 10*|G|^2*r_loc^2 + |G|^4*r_loc^4)
             + C4*(105 - 105*|G|^2*r_loc^2 + 21*|G|^4*r_loc^4 - |G|^6*r_loc^6))

Nonlocal part uses separable Kleinman-Bylander projectors.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from pwdft.constants import PI, FOUR_PI


@dataclass
class HGHParams:
    """Parameters for a single HGH pseudopotential.

    Attributes:
        symbol: Element symbol.
        Z_ion: Number of valence electrons.
        r_loc: Local radius parameter.
        c_loc: Local potential coefficients [C1, C2, C3, C4].
        r_nl: Nonlocal radii for each angular momentum channel.
        h_nl: Nonlocal projector matrix h^l_{ij} for each l.
    """
    symbol: str
    Z_ion: int
    r_loc: float
    c_loc: list[float] = field(default_factory=list)
    r_nl: list[float] = field(default_factory=list)
    h_nl: list[np.ndarray] = field(default_factory=list)


def _pad_c_loc(c: list[float]) -> jnp.ndarray:
    """Pad c_loc to length 4 with zeros."""
    padded = [0.0] * 4
    for i, ci in enumerate(c):
        padded[i] = ci
    return jnp.array(padded, dtype=jnp.float64)


def vloc_reciprocal(g2: jnp.ndarray, params: HGHParams, omega: float) -> jnp.ndarray:
    """Compute the local pseudopotential in reciprocal space.

    V_loc(G) for G != 0. The G=0 component needs special treatment.

    Args:
        g2: (npw,) |G|^2 values.
        params: HGH pseudopotential parameters.
        omega: Unit cell volume.

    Returns:
        (npw,) V_loc(G) values.
    """
    r_loc = params.r_loc
    Z_ion = params.Z_ion
    c = _pad_c_loc(params.c_loc)
    r2 = r_loc * r_loc
    gr2 = g2 * r2

    # Avoid division by zero at G=0
    g2_safe = jnp.where(g2 == 0.0, 1.0, g2)

    # Coulomb part: -Z_ion * 4*pi / (Omega * |G|^2) * exp(-|G|^2 * r_loc^2 / 2)
    exp_term = jnp.exp(-gr2 / 2.0)
    v_coulomb = -Z_ion * FOUR_PI / (omega * g2_safe) * exp_term

    # Gaussian part
    prefactor = jnp.sqrt(8.0 * PI**3) * r_loc**3 / omega * exp_term
    poly = (c[0]
            + c[1] * (3.0 - gr2)
            + c[2] * (15.0 - 10.0 * gr2 + gr2**2)
            + c[3] * (105.0 - 105.0 * gr2 + 21.0 * gr2**2 - gr2**3))
    v_gauss = prefactor * poly

    v_loc = v_coulomb + v_gauss

    # G=0 limit: V_loc(G=0) = 2*pi*Z_ion*r_loc^2/Omega + sqrt(8*pi^3)*r_loc^3/Omega * (C1 + 3*C2 + 15*C3 + 105*C4)
    v_g0 = (2.0 * PI * Z_ion * r2 / omega
            + jnp.sqrt(8.0 * PI**3) * r_loc**3 / omega
            * (c[0] + 3.0 * c[1] + 15.0 * c[2] + 105.0 * c[3]))

    v_loc = jnp.where(g2 == 0.0, v_g0, v_loc)
    return v_loc


def projector_fourier(g_vec: jnp.ndarray, l: int, i: int, r_l: float) -> jnp.ndarray:
    """Compute the Fourier transform of the HGH projector p^l_i(G).

    The projectors are:
        p^l_i(r) = sqrt(2) * r^{l+2(i-1)} * exp(-r^2/(2*r_l^2))
                   / (r_l^{l+(4i-1)/2} * sqrt(Gamma(l+(4i-1)/2)))

    Their Fourier transforms are known analytically.

    For l=0:
        p^0_1(G) = (4*pi)^(1/4) * sqrt(2*r_l) * exp(-G^2*r_l^2/2)  [times Y_00 factor removed]
        p^0_2(G) = (4*pi)^(1/4) * sqrt(2*r_l) * (2*sqrt(2/15)) * r_l^2
                   * (3 - G^2*r_l^2) * exp(-G^2*r_l^2/2)
        p^0_3(G) = (4*pi)^(1/4) * sqrt(2*r_l) * (4/(3*sqrt(105))) * r_l^4
                   * (15 - 10*G^2*r_l^2 + G^4*r_l^4) * exp(-G^2*r_l^2/2)

    For l=1:
        p^1_1(G) = (4*pi)^(1/4) * sqrt(2*r_l) * (1/sqrt(3)) * r_l * G * exp(-G^2*r_l^2/2)
        p^1_2(G) = (4*pi)^(1/4) * sqrt(2*r_l) * (2/(sqrt(105))) * r_l^3
                   * G * (5 - G^2*r_l^2) * exp(-G^2*r_l^2/2)
        p^1_3(G) = (4*pi)^(1/4) * sqrt(2*r_l) * (4/(3*sqrt(1155))) * r_l^5
                   * G * (35 - 14*G^2*r_l^2 + G^4*r_l^4) * exp(-G^2*r_l^2/2)

    For l=2:
        p^2_1(G) = (4*pi)^(1/4) * sqrt(2*r_l) * (1/sqrt(15)) * r_l^2 * G^2 * exp(-G^2*r_l^2/2)
        p^2_2(G) = (4*pi)^(1/4) * sqrt(2*r_l) * (2/(3*sqrt(105))) * r_l^4
                   * G^2 * (7 - G^2*r_l^2) * exp(-G^2*r_l^2/2)

    These are the RADIAL parts. The angular part (spherical harmonics) is handled
    separately in the nonlocal potential construction.

    Args:
        g_vec: (npw, 3) k+G vectors in Cartesian coords.
        l: Angular momentum quantum number.
        i: Projector index (0-based: i=0 is first projector).
        r_l: Nonlocal radius for channel l.

    Returns:
        (npw,) radial projector values (without spherical harmonic).
    """
    g_norm = jnp.linalg.norm(g_vec, axis=-1)
    g_norm_safe = jnp.where(g_norm == 0.0, 1.0, g_norm)
    g2r2 = g_norm**2 * r_l**2
    exp_term = jnp.exp(-g2r2 / 2.0)
    prefactor = (FOUR_PI)**0.25 * jnp.sqrt(2.0 * r_l)

    if l == 0 and i == 0:
        return prefactor * exp_term
    elif l == 0 and i == 1:
        return prefactor * 2.0 * jnp.sqrt(2.0 / 15.0) * r_l**2 * (3.0 - g2r2) * exp_term
    elif l == 0 and i == 2:
        return prefactor * (4.0 / (3.0 * jnp.sqrt(105.0))) * r_l**4 * (15.0 - 10.0 * g2r2 + g2r2**2) * exp_term
    elif l == 1 and i == 0:
        return prefactor * (1.0 / jnp.sqrt(3.0)) * r_l * g_norm * exp_term
    elif l == 1 and i == 1:
        return prefactor * (2.0 / jnp.sqrt(105.0)) * r_l**3 * g_norm * (5.0 - g2r2) * exp_term
    elif l == 1 and i == 2:
        return prefactor * (4.0 / (3.0 * jnp.sqrt(1155.0))) * r_l**5 * g_norm * (35.0 - 14.0 * g2r2 + g2r2**2) * exp_term
    elif l == 2 and i == 0:
        return prefactor * (1.0 / jnp.sqrt(15.0)) * r_l**2 * g_norm**2 * exp_term
    elif l == 2 and i == 1:
        return prefactor * (2.0 / (3.0 * jnp.sqrt(105.0))) * r_l**4 * g_norm**2 * (7.0 - g2r2) * exp_term
    else:
        raise ValueError(f"Projector l={l}, i={i} not implemented")


def real_spherical_harmonics(g_vec: jnp.ndarray, l: int) -> jnp.ndarray:
    """Compute real spherical harmonics Y_lm(G) for all m at given l.

    Uses the convention where Y_lm are real and normalized.

    Args:
        g_vec: (npw, 3) vectors.
        l: Angular momentum.

    Returns:
        (2*l+1, npw) real spherical harmonics.
    """
    g_norm = jnp.linalg.norm(g_vec, axis=-1)
    g_safe = jnp.where(g_norm == 0.0, 1.0, g_norm)
    x = g_vec[:, 0] / g_safe
    y = g_vec[:, 1] / g_safe
    z = g_vec[:, 2] / g_safe

    if l == 0:
        # Y_00 = 1/sqrt(4*pi) -- but we use convention where this is 1
        # Actually in the KB formalism we need the 4*pi normalization
        y00 = jnp.ones_like(g_norm) * 0.5 / jnp.sqrt(PI)
        return y00[None, :]

    elif l == 1:
        # Y_{1,-1} = sqrt(3/(4*pi)) * y
        # Y_{1,0}  = sqrt(3/(4*pi)) * z
        # Y_{1,1}  = sqrt(3/(4*pi)) * x
        c = jnp.sqrt(3.0 / (FOUR_PI))
        ylm = jnp.stack([c * y, c * z, c * x], axis=0)  # (3, npw)
        # Zero out for G=0
        ylm = jnp.where(g_norm[None, :] == 0.0, 0.0, ylm)
        return ylm

    elif l == 2:
        # Real spherical harmonics for l=2
        c0 = 0.5 * jnp.sqrt(5.0 / PI)
        c1 = jnp.sqrt(15.0 / (FOUR_PI))
        c2 = 0.25 * jnp.sqrt(15.0 / PI)
        ylm = jnp.stack([
            c1 * x * y,                      # Y_{2,-2}
            c1 * y * z,                      # Y_{2,-1}
            c0 * (3.0 * z * z - 1.0) / 2.0, # Y_{2,0}
            c1 * x * z,                      # Y_{2,1}
            c2 * (x * x - y * y),            # Y_{2,2}
        ], axis=0)  # (5, npw)
        ylm = jnp.where(g_norm[None, :] == 0.0, 0.0, ylm)
        return ylm

    else:
        raise ValueError(f"l={l} not implemented (max l=2)")


# ============================================================================
# HGH Parameter Database
# Parameters from Hartwigsen, Goedecker, Hutter, PRB 58, 3641 (1998)
# and Goedecker, Teter, Hutter, PRB 54, 1703 (1996)
# ============================================================================

def get_hgh_params(symbol: str) -> HGHParams:
    """Get HGH pseudopotential parameters for a given element.

    Args:
        symbol: Element symbol (e.g., 'Si', 'H', 'C').

    Returns:
        HGHParams for the element.
    """
    db = _HGH_DATABASE
    if symbol not in db:
        raise ValueError(f"HGH parameters not available for '{symbol}'. "
                         f"Available: {list(db.keys())}")
    return db[symbol]


def _make_hgh(symbol, Z_ion, r_loc, c_loc, r_nl=None, h_nl=None):
    """Helper to build HGHParams."""
    if r_nl is None:
        r_nl = []
    if h_nl is None:
        h_nl = []
    return HGHParams(
        symbol=symbol,
        Z_ion=Z_ion,
        r_loc=r_loc,
        c_loc=c_loc,
        r_nl=r_nl,
        h_nl=[np.array(h, dtype=np.float64) for h in h_nl],
    )


# GTH-Pade parameters (widely used, well-tested)
# From: S. Goedecker, M. Teter, J. Hutter, PRB 54, 1703 (1996)
# and C. Hartwigsen, S. Goedecker, J. Hutter, PRB 58, 3641 (1998)
_HGH_DATABASE: dict[str, HGHParams] = {
    # Hydrogen: 1 valence electron
    "H": _make_hgh(
        symbol="H", Z_ion=1,
        r_loc=0.20000000,
        c_loc=[-4.18023680, 0.72507482, 0.0, 0.0],
        r_nl=[],
        h_nl=[],
    ),
    # Helium: 2 valence electrons
    "He": _make_hgh(
        symbol="He", Z_ion=2,
        r_loc=0.20000000,
        c_loc=[-9.11202340, 1.69836797, 0.0, 0.0],
        r_nl=[],
        h_nl=[],
    ),
    # Lithium: 3 valence electrons
    "Li": _make_hgh(
        symbol="Li", Z_ion=3,
        r_loc=0.40000000,
        c_loc=[-14.03493470, 9.55346109, -1.75328669, 0.08644523],
        r_nl=[],
        h_nl=[],
    ),
    # Carbon: 4 valence electrons
    "C": _make_hgh(
        symbol="C", Z_ion=4,
        r_loc=0.34883045,
        c_loc=[-8.51377110, 1.22843203, 0.0, 0.0],
        r_nl=[0.30455321],
        h_nl=[[[9.52284179, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]],
    ),
    # Nitrogen: 5 valence electrons
    "N": _make_hgh(
        symbol="N", Z_ion=5,
        r_loc=0.28917923,
        c_loc=[-12.23481988, 1.76640728, 0.0, 0.0],
        r_nl=[0.25660487],
        h_nl=[[[13.55224272, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]],
    ),
    # Oxygen: 6 valence electrons
    "O": _make_hgh(
        symbol="O", Z_ion=6,
        r_loc=0.24762086,
        c_loc=[-16.58031797, 2.39570092, 0.0, 0.0],
        r_nl=[0.22178614],
        h_nl=[[[18.26691718, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]],
    ),
    # Silicon: 4 valence electrons
    "Si": _make_hgh(
        symbol="Si", Z_ion=4,
        r_loc=0.44000000,
        c_loc=[-7.33610297, 0.0, 0.0, 0.0],
        r_nl=[0.42273813, 0.48427842],
        h_nl=[
            [[5.90692831, -1.26189397, 0.0],
             [-1.26189397, 3.25819622, 0.0],
             [0.0, 0.0, 0.0]],
            [[2.72701346, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ],
    ),
    # Aluminum: 3 valence electrons
    "Al": _make_hgh(
        symbol="Al", Z_ion=3,
        r_loc=0.45000000,
        c_loc=[-8.49135116, 0.0, 0.0, 0.0],
        r_nl=[0.46010427, 0.53674439],
        h_nl=[
            [[6.86726458, -2.15038876, 0.0],
             [-2.15038876, 3.08498668, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.93798172, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ],
    ),
    # Gallium arsenide components
    "Ga": _make_hgh(
        symbol="Ga", Z_ion=3,
        r_loc=0.49000000,
        c_loc=[-6.23392822, 0.0, 0.0, 0.0],
        r_nl=[0.61178539, 0.70575027],
        h_nl=[
            [[2.36854689, -0.57979762, 0.0],
             [-0.57979762, 0.49498862, 0.0],
             [0.0, 0.0, 0.0]],
            [[0.63760007, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ],
    ),
    "As": _make_hgh(
        symbol="As", Z_ion=5,
        r_loc=0.52000000,
        c_loc=[-5.47790000, 0.0, 0.0, 0.0],
        r_nl=[0.45590000, 0.55510000],
        h_nl=[
            [[6.32960000, -1.82040000, 0.0],
             [-1.82040000, 2.25390000, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.85240000, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ],
    ),
    # Sodium: 9 valence electrons (semicore)
    "Na": _make_hgh(
        symbol="Na", Z_ion=9,
        r_loc=0.24631780,
        c_loc=[-7.54559389, 0.94978099, 0.0, 0.0],
        r_nl=[0.14190780, 0.16260350],
        h_nl=[
            [[36.55522782, -11.57770700, 0.0],
             [-11.57770700, 14.76498206, 0.0],
             [0.0, 0.0, 0.0]],
            [[-0.72919513, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ],
    ),
    # Chlorine: 7 valence electrons
    "Cl": _make_hgh(
        symbol="Cl", Z_ion=7,
        r_loc=0.41000000,
        c_loc=[-6.39208181, 0.0, 0.0, 0.0],
        r_nl=[0.33861282, 0.39636383],
        h_nl=[
            [[14.60920767, -5.48969499, 0.0],
             [-5.48969499, 5.09007974, 0.0],
             [0.0, 0.0, 0.0]],
            [[3.82663452, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ],
    ),
    # Germanium: 4 valence electrons
    "Ge": _make_hgh(
        symbol="Ge", Z_ion=4,
        r_loc=0.54000000,
        c_loc=[-4.30885032, 0.0, 0.0, 0.0],
        r_nl=[0.42836784, 0.56646247],
        h_nl=[
            [[6.00512414, -2.15023853, 0.0],
             [-2.15023853, 1.70038991, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.35498601, 0.0, 0.0],
             [0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ],
    ),
}
