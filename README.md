# InSAR SBAS Studio

This is a public, static browser application for small InSAR SBAS experiments. Visitors do not need a ChatGPT or GitHub account to use the live page.

## Publish with GitHub Pages

1. Upload the files in this repository to the `main` branch.
2. Open **Settings → Pages** and select **Deploy from a branch**, branch `main`, folder `/(root)`.
3. Wait for GitHub Pages to publish. The public URL is `https://zhengtaiwei57-web.github.io/insar-sbas-studio/`.

Python and NumPy run in each visitor's browser using Pyodide 0.27.7 from jsDelivr. User-selected data stays in that browser. The first run downloads about 20 MB of runtime files.

## Features and limits

Visitors can run the synthetic demonstration, import an unwrapped `.npz` interferogram stack, invert the connected SBAS network, inspect four result layers, view a selected pixel's time series, and download NPZ, CSV, JSON, or PNG outputs. The browser version limits inputs to 25 MiB, 36 acquisition dates, 160 interferograms, and a maximum 512 by 512 pixel grid per stack dimension.

Original SLC focusing, interferogram generation, phase unwrapping, atmospheric/orbit/DEM corrections, and production-scale processing need a separately configured ISCE2/MintPy processing environment. They do not run on this static page.

Pyodide, CPython, and NumPy license notices are included in `runtime/`.

