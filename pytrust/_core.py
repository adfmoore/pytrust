'''
Shared numeric primitives.

Nothing here is public. These are the pieces of linear algebra that both the
demand systems and the merger tools need, kept in one place so that neither
imports the other. The dependency direction is

    _core  <-  demand  <-  mergers

and structure.py depends on none of them. A new demand system or a new merger
screen should reach for these rather than reimplementing them.

Dependencies:
1. numpy
'''

from __future__ import annotations

import numpy as np


def _elasticities(B: np.ndarray, shares: np.ndarray,
                  industry_elasticity: float) -> np.ndarray:
    '''
    Converts price coefficients to elasticities.

    Takes three inputs:
    1. B, the J x J matrix of b_ij coefficients
    2. Shares, an array of market shares, one per brand
    3. Industry elasticity

    Returns the J x J elasticity matrix, row = responding brand, column = brand
    whose price changes. The result is not symmetric: e_ij and e_ji share a
    coefficient but divide by different shares, so they coincide only when
    s_i == s_j.
    '''
    s = np.asarray(shares, dtype=float)
    E = B / s[:, None] + s[None, :] * (industry_elasticity + 1)
    E[np.diag_indices_from(E)] -= 1.0
    return E


def _ownership(firms: np.ndarray) -> np.ndarray:
    '''
    Builds the ownership matrix from an array of firm labels.

    Takes one input:
    1. Firms, an array of firm labels, one per brand

    Returns a J x J boolean matrix with entry (i, j) True when brands i and j
    are owned by the same firm. Under Bertrand competition a firm internalises
    the effect of brand i's price on brand j's profit exactly when this entry
    is True, so a merger is represented by flipping entries from False to True.
    '''
    f = np.asarray(firms)
    return f[:, None] == f[None, :]


def _margins_from_foc(E: np.ndarray, s: np.ndarray,
                      omega: np.ndarray) -> np.ndarray:
    r'''
    Recovers price-cost margins implied by Bertrand-Nash pricing.

    Takes three inputs:
    1. E, the J x J elasticity matrix
    2. s, revenue shares
    3. omega, the J x J ownership matrix

    The first-order condition for product i owned by firm f is

        s_i + sum_{j in f} m_j * s_j * e_ji = 0,

    which is linear in the products m_j * s_j and can be solved directly.
    For a single-product firm it collapses to the Lerner index, m = -1 / e_ii.

    Returns an array of margins, (p - c) / p.
    '''
    A = omega * E.T
    ms = np.linalg.solve(A, -s)
    return ms / s
