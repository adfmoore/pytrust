'''
Python implementations of common tools used in merger analysis.

The public API is defined here and is deliberately flat, so the internal file
layout can change without breaking anyone's code. Import from the package root
rather than from the modules:

    import pytrust as pt
    pt.calibrate(...)          not   pt.demand.calibrate(...)

Two demand systems, both of which produce the same J x J matrix of price
coefficients and both of which can drive a merger simulation:

    calibrate()        PCAIDS, from shares and two elasticities
    estimate()         AIDS, from a panel of shares and prices
    diversion_ratios() post-estimation diversion
    simulate_merger()  Bertrand-Nash price effects, given either

And one measure of market structure:

    hhi()              Herfindahl-Hirschman Index by market

Modules, in dependency order:

    _core       shared numeric primitives, private
    demand      calibrate, estimate, diversion_ratios
    mergers     simulate_merger, and future screens
    structure   hhi, and future concentration measures

To add a tool, put the function in whichever module matches what it does, then
export it here.
'''

from . import demand, mergers, structure
from .demand import calibrate, diversion_ratios, estimate
from .mergers import simulate_merger
from .structure import hhi

__version__ = '0.1.0'

__all__ = [
    # demand
    'calibrate',
    'estimate',
    'diversion_ratios',
    # mergers
    'simulate_merger',
    # structure
    'hhi',
    # modules
    'demand',
    'mergers',
    'structure',
]
