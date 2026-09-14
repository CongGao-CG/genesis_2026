# A common warming-dependent decline emerges from divergent projections of tropical cyclone frequency

[![DOI](https://zenodo.org/badge/1370452656.svg)](https://doi.org/10.5281/zenodo.22758581)

**Cong Gao and Ning Lin**  
Department of Civil and Environmental Engineering, Princeton University, Princeton, NJ, USA

**Correspondence:** [cong.gao@princeton.edu](mailto:cong.gao@princeton.edu) · [nlin@princeton.edu](mailto:nlin@princeton.edu)

Figure scripts and input data accompanying the paper.

## Contents

- `GFig1.py`–`GFig4.py`: Figures 1–4.
- `GFigS1.py`–`GFigS2.py`: Figures S1–S2.
- TSV files: tropical cyclone rates.
- `gmt/`: global-temperature inputs.

Each figure script is self-contained and uses the included input data.

## Run

Requires Python 3.10+ with `matplotlib`, `numpy`, `pandas`, `scipy`, `statsmodels`, `xarray`, `netCDF4`, and `cftime`.

From this folder, run:

```bash
for f in GFig1 GFig2 GFig3 GFig4 GFigS1 GFigS2; do
    python "$f.py"
done
```

Each script saves PDF and SVG files and opens the PDF.

## Citation and license

See [CITATION.cff](CITATION.cff) for citation metadata and [LICENSE](LICENSE) for the MIT license.

The arXiv identifier will be added when available.
