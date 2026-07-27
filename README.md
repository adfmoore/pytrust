# PyTrust 
## Python implementations of common tools used in antitrust work

### Overview
This project implements several methods used in antitrust work, with a focus (at least for now) on evaluating mergers. This is a small side project I work on as I have time available, and is in no way all-inclusive of every single tool. I plan to update this repo to include more methods over time. An item that I will not be implementing in this package are BLP-related tools -- `pyblp` already does this fantastically. 

This is similar in spirit to the R package `antitrust`, but -- at present (7/26/2026) -- is much less comprehensive. So, what is the advantage to using this package over the R equivalent? If I am honest, really nothing! The one tiny advantage is getting to use Python over R, for those with strong preferences over languages. 


### What's included

| Function | Does |
|---|---|
| `calibrate()` | PCAIDS demand, from shares and two elasticities |
| `estimate()` | AIDS demand, from a panel of shares and prices |
| `diversion_ratios()` | Diversion ratios from an elasticity matrix |
| `simulate_merger()` | Bertrand-Nash price effects, using either demand system |
| `hhi()` | Herfindahl-Hirschman Index |

Each function's docstring holds the full reference: every argument, the formulas behind it, and the assumptions the result rests on. Call `help()` on whichever one you need:

```python
import pytrust
help(pytrust.calibrate)     # or any other function in the table
```

### Quick Start

To install, clone the repo and install it in editable mode:

```bash
git clone https://github.com/adfmoore/pytrust.git
cd pytrust
pip install -e .
```

Example usage: 

```python
import pytrust as pt 
import pandas as pd

# Generate example data
data = pd.DataFrame({
    "brand": ["A", "B", "C", "D"],
    'firm': ['F1', 'F2', 'F3', 'F4'],
    'share': [0.40, 0.30, 0.20, 0.10]
})

# Use PCAIDS to sim a merger between two firms
merger = pt.simulate_merger(data=data, industry_elasticity = -1, merging = ['F1', 'F2'],
                            brand = 'A', own_elasticity = -3.0)

merger.round(4)
```

which returns

```
      firm  merging  share_pre  share_post  margin_pre  margin_post  price_change
brand                                                                            
A       F1     True        0.4      0.3571      0.3333       0.4655        0.2472
B       F2     True        0.3      0.2407      0.3000       0.4538        0.2815
C       F3    False        0.2      0.2673      0.2727       0.3339        0.0918
D       F4    False        0.1      0.1350      0.2500       0.3103        0.0874
```

You can also use `pytrust` calculate -- for lack of a better term -- 'exploratory' competition indicators. These are typically the first items to be computed when beginning to evaluate a potential merger. For now, HHI is the only included index. I plan to include UPP, GUPPI and CMCR (for example) in the future. The package is therefore still under development, and will be updated slowly over time.  

```python
data['market'] = 'M1'
pt.hhi(data, market = 'market', shares = 'share', firm = 'firm')
```

which returns a Series indexed by market

```
market
M1    3000.0
Name: hhi, dtype: float64
```

### AI Disclaimer
I used Claude Code to help write and troubleshoot portions of this package, and tasked it with writing arguments for `help()` calls. I verified its code and output, and so claim responsibility for any errors in this repo. It was also substantially helpful with setting up the structure of the files. 
