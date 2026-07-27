'''
Demand systems.

Two routes to the same object, the J x J matrix of price coefficients b_ij:

1. calibrate() implements PCAIDS (Epstein and Rubinfeld 2002). It builds B from
   market shares, the own-price elasticity of a single brand, and the industry
   price elasticity, using the proportionality assumption to fill in the rest.
2. estimate() implements AIDS (Deaton and Muellbauer 1980). It fits B by
   constrained least squares on a panel of revenue shares and prices, imposing
   symmetry, homogeneity, and adding-up.

Both return (elasticities, coefficients) in the same shape, and either B can be
handed to the tools in mergers.py, which are agnostic about where it came from.

A new demand system belongs here, and needs only to return that same pair to
work with everything downstream.

Dependencies:
1. numpy
2. pandas
'''

from __future__ import annotations

import numpy as np
import pandas as pd

from ._core import _elasticities

__all__ = ['calibrate', 'estimate', 'diversion_ratios']


# ---------------------------------------------------------------------------
# PCAIDS calibration
# ---------------------------------------------------------------------------

def _validate(shares: np.ndarray, own_elasticity: float, industry_elasticity: float,
              tol: float = 1e-8) -> None:
    if np.any(shares <= 0):
        raise ValueError('All market shares must be strictly positive.')
    if abs(shares.sum() - 1) > tol:
        raise ValueError(f'Market shares must sum to 1 (got {shares.sum():.6f})')
    if own_elasticity >= 0:
        raise ValueError("Own-price elasticity must be negative.")
    if industry_elasticity >= 0:
        raise ValueError('Industry price elasticity must be negative.')
    if own_elasticity > industry_elasticity:
        raise ValueError('Brand own-price elasticity must be at least as elastic as the industry, otherwise ' \
        'the implied substitution effect is negative.')


def _coefficients(shares: np.ndarray, position: int, own_elasticity: float,
                  industry_elasticity: float) -> np.ndarray:
    '''
   Numeric core. Inputs are not validated here; call calibrate() for the
    checked version.

    Takes four inputs:
    1. Shares, an array of market shares, one per brand
    2. Position, the zero-based index into that array of the brand whose
       own-price elasticity is known. A position, not a brand label
    3. Own-price elasticity of that brand
    4. Industry elasticity

    Returns the J x J matrix of b_ij coefficients, own-effects on the diagonal.
    '''
    s = np.asarray(shares, dtype=float)
    s_1 = s[position]

    b_11 = s_1 * (own_elasticity + 1 - s_1 * (industry_elasticity + 1))
    k = -b_11 / (s_1 * (1 - s_1))

    B = k * np.outer(s, s)
    np.fill_diagonal(B, -k * s * (1 - s))
    return B


def calibrate(data: pd.DataFrame, brand: str, own_elasticity: float,
              industry_elasticity: float, brand_col: str = "brand",
              share_col: str = "share", normalize: bool = False):
    '''
    Calibrates a PCAIDS system from a DataFrame with one row per brand.

    This calibration does not cooperate with panels. Maybe in a future version?

    Takes seven inputs:
    1. Data, a pandas DataFrame with one row per brand.
    2. Brand, the label of the brand whose own-price elasticity is known,
       matched by value against brand_col. May be a string or an integer,
       but is a label rather than a row position
    3. Own-price elasticity of that brand, e.g. -3.0
    4. Industry elasticity, e.g. -1.0
    5. brand_col, the column holding brand labels
    6. share_col, the column holding market shares
    7. Normalize, if True rescales shares to sum to one rather than raising

    Returns two DataFrames, (elasticities, coefficients), both indexed by brand
    label with rows the responding brand and columns the brand whose price
    changes.

    Shares are revenue shares. Unit shares are often substituted in practice;
    the two coincide only when prices are equal across brands.
    '''
    missing = [c for c in (brand_col, share_col) if c not in data]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in the DataFrame. Available: "
            f"{list(data.columns)}. Pass brand_col= and share_col= if your "
            f"columns are named differently."
        )

    labels = data[brand_col].to_numpy()
    if len(set(labels)) != len(labels):
        raise ValueError(f"Brand labels in '{brand_col}' must be unique.")

    matches = np.flatnonzero(labels == brand)
    if matches.size == 0:
        hint = ""
        if any(type(x) is not type(brand) for x in labels):
            hint = (f" Note that brand is a {type(brand).__name__} while the column "
                    f"holds {type(labels[0]).__name__}; labels are matched by value "
                    f"and type, so '2' will not match 2.")
        raise KeyError(f"Brand {brand!r} not found in column '{brand_col}'. "
                       f"Available: {list(labels)}.{hint}")

    s = data[share_col].to_numpy(dtype=float)
    if normalize:
        s = s / s.sum()

    _validate(s, own_elasticity, industry_elasticity)

    position = int(matches[0])
    B = _coefficients(s, position, own_elasticity, industry_elasticity)
    E = _elasticities(B, s, industry_elasticity)

    coefficients = pd.DataFrame(B, index=labels, columns=labels)
    elasticities = pd.DataFrame(E, index=labels, columns=labels)
    coefficients.index.name = elasticities.index.name = "brand"
    coefficients.columns.name = elasticities.columns.name = "price of"

    return elasticities, coefficients


# ---------------------------------------------------------------------------
# AIDS estimation
# ---------------------------------------------------------------------------

def _panel_to_matrices(data: pd.DataFrame, brand_col: str, market_col: str,
                       price_col: str, share_col: str):
    '''
    Reshapes a long panel into aligned T x J matrices of shares and log prices.

    Takes five inputs, the DataFrame and the four column names. Returns
    (shares, log_prices, labels), where labels is the brand ordering used by
    both matrices. Requires a balanced panel: every brand must appear exactly
    once in every market.
    '''
    missing = [c for c in (brand_col, market_col, price_col, share_col)
               if c not in data]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in the DataFrame. Available: "
            f"{list(data.columns)}."
        )

    shares = data.pivot_table(index=market_col, columns=brand_col,
                              values=share_col, aggfunc='first')
    prices = data.pivot_table(index=market_col, columns=brand_col,
                              values=price_col, aggfunc='first')

    if shares.isna().to_numpy().any() or prices.isna().to_numpy().any():
        counts = data.groupby(brand_col)[market_col].nunique()
        raise ValueError(
            'The panel is unbalanced: every brand must appear exactly once in '
            'every market. Brands appear in this many markets:\n'
            f'{counts.to_string()}'
        )

    S = shares.to_numpy(dtype=float)
    P = prices.to_numpy(dtype=float)

    if np.any(P <= 0):
        raise ValueError('All prices must be strictly positive.')
    row_sums = S.sum(axis=1)
    if np.max(np.abs(row_sums - 1.0)) > 1e-6:
        raise ValueError(
            'Revenue shares must sum to one within every market (worst market '
            f'sums to {row_sums[np.argmax(np.abs(row_sums - 1.0))]:.6f}). '
            'Compute them as p*q / sum(p*q) within each market.'
        )

    return S, np.log(P), shares.columns.to_numpy()


def estimate(data: pd.DataFrame, industry_elasticity: float,
             brand_col: str = 'brand', market_col: str = 'market',
             price_col: str = 'price', share_col: str = 'share',
             shares: np.ndarray | pd.Series | None = None):
    r'''
    Estimates an AIDS system from a panel of revenue shares and prices.

    Takes six inputs:
    1. Data, a long DataFrame with one row per brand per market
    2. Industry elasticity, e.g. -1.0, used only to convert coefficients into
       elasticities. It is not estimated here
    3. brand_col, the column holding brand labels
    4. market_col, the column identifying the market or period
    5. price_col, the column holding prices
    6. share_col, the column holding revenue shares, which must sum to one
       within each market
    7. Shares, the share vector at which to report elasticities. Defaults to
       the mean revenue share of each brand across markets

    The share equation is

        s_i = a_i + sum_j b_ij log p_j,

    fitted subject to the three standard restrictions: symmetry (b_ij = b_ji),
    homogeneity (sum_j b_ij = 0), and adding-up (sum_i b_ij = 0). Imposing
    symmetry and homogeneity lets the system be rewritten as

        s_i = a_i + sum_{j != i} b_ij (log p_j - log p_i),

    which is linear in the J(J-1)/2 free off-diagonal coefficients and can be
    stacked into a single least-squares problem. Diagonal terms follow from
    b_ii = -sum_{j != i} b_ij, and adding-up then holds automatically.

    Identification needs enough independent price variation: at least
    1 + (J-1)/2 markets, and in practice a good deal more. The estimator is
    ordinary least squares, so it inherits the usual simultaneity problem, and
    it is the source of the wrong-signed or implausibly large cross-price terms
    that motivated PCAIDS in the first place. Treat the output as a
    reduced-form summary rather than a causal one.

    Returns two DataFrames, (elasticities, coefficients), indexed by brand
    label exactly as calibrate() does, so the two are interchangeable.
    '''
    if industry_elasticity >= 0:
        raise ValueError('Industry price elasticity must be negative.')

    S, LP, labels = _panel_to_matrices(data, brand_col, market_col,
                                       price_col, share_col)
    T, J = S.shape
    if J < 2:
        raise ValueError('At least two brands are required.')

    pairs = [(k, l) for k in range(J) for l in range(k + 1, J)]
    n_free = len(pairs)
    if T * J < J + n_free:
        raise ValueError(
            f'Not enough data to estimate {J + n_free} parameters from '
            f'{T * J} observations ({J} brands over {T} markets). AIDS needs '
            f'at least {int(np.ceil(1 + (J - 1) / 2))} markets for {J} brands; '
            'PCAIDS via calibrate() is the low-data alternative.'
        )

    # Stacked design, equation-major: rows i*T:(i+1)*T hold brand i.
    design = np.zeros((T * J, J + n_free))
    for i in range(J):
        design[i * T:(i + 1) * T, i] = 1.0
    for q, (k, l) in enumerate(pairs):
        diff = LP[:, l] - LP[:, k]
        design[k * T:(k + 1) * T, J + q] = diff
        design[l * T:(l + 1) * T, J + q] = -diff

    y = S.T.ravel()
    theta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        raise ValueError(
            f'The design matrix is rank deficient ({rank} of '
            f'{design.shape[1]} columns). Prices do not vary independently '
            'enough across markets to separate every cross-price coefficient. '
            'Use fewer brands, more markets, or calibrate() instead.'
        )

    B = np.zeros((J, J))
    for q, (k, l) in enumerate(pairs):
        B[k, l] = B[l, k] = theta[J + q]
    np.fill_diagonal(B, -B.sum(axis=1))

    if shares is None:
        s = S.mean(axis=0)
        s = s / s.sum()
    else:
        s = np.asarray(shares, dtype=float)
        if s.shape != (J,):
            raise ValueError(f'shares must have one entry per brand (got '
                             f'{s.shape}, expected {(J,)}).')
    if np.any(s <= 0):
        raise ValueError('All market shares must be strictly positive.')

    E = _elasticities(B, s, industry_elasticity)

    coefficients = pd.DataFrame(B, index=labels, columns=labels)
    elasticities = pd.DataFrame(E, index=labels, columns=labels)
    coefficients.index.name = elasticities.index.name = 'brand'
    coefficients.columns.name = elasticities.columns.name = 'price of'

    return elasticities, coefficients


# ---------------------------------------------------------------------------
# Post-estimation
# ---------------------------------------------------------------------------

def diversion_ratios(elasticities: pd.DataFrame,
                     shares: pd.Series | np.ndarray) -> pd.DataFrame:
    r'''
    Computes diversion ratios from an elasticity matrix.

    Takes two inputs:
    1. Elasticities, the J x J matrix returned by calibrate() or estimate()
    2. Shares, revenue shares in the same brand order

    The diversion ratio from i to j is the share of brand i's lost unit sales
    captured by brand j,

        D_ij = -(e_ji * s_j) / (e_ii * s_i),

    evaluated at equal prices. Under proportionality D_ij is proportional to
    s_j and does not otherwise depend on j, which is the assumption a merger
    simulation is most sensitive to.

    Returns a J x J DataFrame with a zero diagonal; each row sums to the share
    of lost sales retained inside the market rather than lost to the outside
    good.
    '''
    E = np.asarray(elasticities, dtype=float)
    s = np.asarray(shares, dtype=float)
    D = -(E.T * s[None, :]) / (np.diag(E)[:, None] * s[:, None])
    np.fill_diagonal(D, 0.0)
    out = pd.DataFrame(D, index=elasticities.index, columns=elasticities.columns)
    out.index.name, out.columns.name = 'from', 'to'
    return out


# ---------------------------------------------------------------------------
# Resolving a demand system from user inputs
# ---------------------------------------------------------------------------

def _validate_coefficients(coefficients: pd.DataFrame, labels: np.ndarray) -> None:
    '''
    Checks that an externally supplied B matrix lines up with the market.

    Takes two inputs, the J x J DataFrame of b_ij terms and the brand labels
    from the cross-section being simulated. Raises if the matrix is not square,
    is not symmetric, or does not cover every brand.
    '''
    if not isinstance(coefficients, pd.DataFrame):
        raise TypeError('coefficients must be a DataFrame indexed by brand '
                        'label, as returned by estimate() or calibrate().')
    if coefficients.shape[0] != coefficients.shape[1]:
        raise ValueError(f'coefficients must be square (got '
                         f'{coefficients.shape}).')

    missing = [b for b in labels if b not in coefficients.index]
    if missing:
        raise KeyError(
            f'Brand(s) {missing} appear in the data but not in coefficients. '
            f'The matrix covers: {list(coefficients.index)}.'
        )
    extra = [b for b in coefficients.index if b not in set(labels.tolist())]
    if extra:
        raise KeyError(
            f'Brand(s) {extra} appear in coefficients but not in the data. '
            'The two must describe the same set of brands, otherwise the '
            'shares no longer sum to one over the products being simulated.'
        )

    M = coefficients.reindex(index=labels, columns=labels).to_numpy(dtype=float)
    asym = np.abs(M - M.T).max()
    if asym > 1e-6:
        raise ValueError(f'coefficients must be symmetric, b_ij = b_ji (worst '
                         f'asymmetry {asym:.2e}).')


def _resolve_demand(data: pd.DataFrame, industry_elasticity: float, merging,
                    brand, own_elasticity, coefficients,
                    firm_col: str, brand_col: str, share_col: str,
                    normalize: bool):
    '''
    Turns the arguments a merger tool was called with into the pieces it needs.

    Every tool in mergers.py accepts demand the same two ways, either an
    externally supplied coefficient matrix or the PCAIDS calibration inputs.
    That resolution lives here so the tools cannot drift apart in how they
    interpret their arguments.

    Returns (labels, shares, B, elasticities, firms, merging).
    '''
    if (coefficients is None) == (brand is None and own_elasticity is None):
        raise ValueError(
            'Supply exactly one of coefficients= (AIDS, or any externally '
            'supplied B) or the pair brand=/own_elasticity= (PCAIDS).'
        )
    for col in (firm_col, brand_col, share_col):
        if col not in data:
            raise KeyError(f"Column '{col}' not found. Available: "
                           f"{list(data.columns)}.")

    labels = data[brand_col].to_numpy()
    if len(set(labels)) != len(labels):
        raise ValueError(f"Brand labels in '{brand_col}' must be unique.")

    s0 = data[share_col].to_numpy(dtype=float)
    if normalize:
        s0 = s0 / s0.sum()

    if coefficients is not None:
        _validate_coefficients(coefficients, labels)
        B = coefficients.reindex(index=labels, columns=labels).to_numpy(dtype=float)
        if np.any(s0 <= 0):
            raise ValueError('All market shares must be strictly positive.')
        if abs(s0.sum() - 1) > 1e-8:
            raise ValueError(f'Market shares must sum to 1 (got {s0.sum():.6f})')
        if industry_elasticity >= 0:
            raise ValueError('Industry price elasticity must be negative.')
    else:
        if brand is None or own_elasticity is None:
            raise ValueError('The PCAIDS route needs both brand= and '
                             'own_elasticity=.')
        _, B_df = calibrate(data, brand, own_elasticity, industry_elasticity,
                            brand_col=brand_col, share_col=share_col,
                            normalize=normalize)
        B = B_df.to_numpy()

    row_sums = np.abs(B.sum(axis=1)).max()
    if row_sums > 1e-6:
        raise ValueError(
            f'The coefficient matrix violates adding-up: rows must sum to zero '
            f'(worst row sums to {row_sums:.2e}). Without it a price change '
            'would not conserve total revenue share and the simulation is '
            'not interpretable.'
        )

    E0 = _elasticities(B, s0, industry_elasticity)
    firms = data[firm_col].to_numpy()
    merging = list(merging)
    if len(merging) < 2:
        raise ValueError('At least two firms must be specified as merging.')
    unknown = [f for f in merging if f not in set(firms)]
    if unknown:
        raise KeyError(f"Merging firm(s) {unknown} not found in '{firm_col}'. "
                       f"Available: {sorted(set(firms.tolist()))}.")

    return labels, s0, B, E0, firms, merging
