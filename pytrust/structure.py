'''
Market structure.

Measures of concentration that need only shares and ownership, no demand
system and no assumption about how firms compete. Independent of everything
else in the package.

Dependencies:
1. pandas
'''

from __future__ import annotations

import pandas as pd

__all__ = ['hhi']


def hhi(data: pd.DataFrame, market: str, shares: str, firm: str) -> pd.Series:
    '''
    Takes four inputs:
    1. DataFrame
    2. Market identifier
    3. Shares column
    4. Firm column

    and returns a series of HHI values indexed by market.

    Market shares should be computed with revenues, although the command will function if
    given output shares.
    '''
    by_firm = data.groupby([market, firm])[shares].sum()
    s = by_firm / by_firm.groupby(level=0).sum()
    out = 10_000 * (s**2).groupby(level=0).sum()
    out.name = 'hhi'
    return out
