'''
Merger tools.

Everything here consumes a demand system plus an ownership structure and says
something about the effect of combining two or more firms. The demand system
can come from either route in demand.py, or from anywhere else, since these
tools only ever see the coefficient matrix.

New tools belong here. They should take demand the same way the existing ones
do, by calling _resolve_demand(), which handles the coefficients-versus-PCAIDS
branch and all the validation that goes with it.

Dependencies:
1. numpy
2. pandas
3. scipy
'''

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import root

from ._core import _elasticities, _margins_from_foc, _ownership
from .demand import _resolve_demand

__all__ = ['simulate_merger']


def _foc_residual(x: np.ndarray, B: np.ndarray, s0: np.ndarray,
                  industry_elasticity: float, cost: np.ndarray,
                  omega: np.ndarray) -> np.ndarray:
    '''First-order conditions as a function of log price changes x.'''
    s = s0 + B @ x
    E = _elasticities(B, s, industry_elasticity)
    m = 1.0 - cost / np.exp(x)
    return s + (omega * E.T) @ (m * s)


def simulate_merger(data: pd.DataFrame, industry_elasticity: float, merging,
                    brand=None, own_elasticity: float | None = None,
                    coefficients: pd.DataFrame | None = None,
                    firm_col: str = 'firm', brand_col: str = 'brand',
                    share_col: str = 'share', margins=None,
                    normalize: bool = False) -> pd.DataFrame:
    r'''
    Simulates a merger under AIDS or PCAIDS demand and Bertrand-Nash pricing.

    Takes ten inputs:
    1. Data, a DataFrame with one row per brand, describing the market as it
       stands before the merger
    2. Industry elasticity
    3. Merging, a sequence of two or more firm labels that combine
    4. Brand, the label of the brand whose own-price elasticity is known.
       PCAIDS route only
    5. Own-price elasticity of that brand. PCAIDS route only
    6. Coefficients, a J x J DataFrame of b_ij terms from estimate(), or from
       any other source. AIDS route only
    7. firm_col, the column holding pre-merger firm labels
    8. brand_col, the column holding brand labels
    9. share_col, the column holding revenue shares
    10. Margins, optional observed margins to use in place of those implied by
       the pre-merger first-order conditions

    Supply either coefficients, in which case demand is taken as given and no
    calibration happens, or the pair (brand, own_elasticity), in which case the
    system is calibrated by PCAIDS first. The simulation itself is identical
    either way: it only ever sees B, shares, and the ownership structure.

    When coefficients are supplied they are reindexed onto the brand ordering
    in data, so an estimate() fit and a cross-section of the same market do not
    have to be sorted the same way.

    Marginal costs are held fixed, so the simulation reports the unilateral
    price effect of the merger and assumes away efficiencies, entry,
    repositioning, and any change in the mode of competition.

    Returns a DataFrame indexed by brand with pre- and post-merger shares,
    margins, and the percentage price change.
    '''
    labels, s0, B, E0, firms, merging = _resolve_demand(
        data, industry_elasticity, merging, brand, own_elasticity,
        coefficients, firm_col, brand_col, share_col, normalize)

    omega_pre = _ownership(firms)
    post_firms = np.where(np.isin(firms, merging), 'MERGED', firms.astype(object))
    omega_post = _ownership(post_firms)
    if (omega_post == omega_pre).all():
        raise ValueError('The specified firms are already commonly owned; '
                         'the merger would not change the ownership matrix.')

    if margins is None:
        m0 = _margins_from_foc(E0, s0, omega_pre)
    else:
        m0 = np.asarray(margins, dtype=float)
        if m0.shape != s0.shape:
            raise ValueError('margins must have one entry per brand.')
    if np.any(m0 <= 0) or np.any(m0 >= 1):
        raise ValueError(
            f'Implied margins outside (0, 1): {np.round(m0, 3)}. Bertrand-Nash '
            'pricing cannot rationalise these shares given the demand system. '
            'Check the elasticities, or pass observed margins via margins=.'
        )

    cost = 1.0 - m0                      # prices normalised to one
    args = (B, s0, industry_elasticity, cost, omega_post)
    sol = root(_foc_residual, np.zeros_like(s0), args=args)
    if np.max(np.abs(sol.fun)) > 1e-9:
        sol = root(_foc_residual, np.zeros_like(s0), args=args, method='lm')
    residual = np.max(np.abs(_foc_residual(sol.x, *args)))
    if residual > 1e-8:
        raise RuntimeError(
            f'Post-merger equilibrium did not converge (max |FOC| = {residual:.2e}). '
            f'Solver message: {sol.message}'
        )

    x = sol.x
    s1 = s0 + B @ x
    price_change = np.exp(x) - 1.0
    m1 = 1.0 - cost / np.exp(x)

    out = pd.DataFrame({
        'firm': firms,
        'merging': np.isin(firms, merging),
        'share_pre': s0,
        'share_post': s1,
        'margin_pre': m0,
        'margin_post': m1,
        'price_change': price_change,
    }, index=labels)
    out.index.name = 'brand'
    return out
