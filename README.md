# InSAR SBAS Studio

This is a public, static browser application for small InSAR SBAS experiments. Visitors do not need a ChatGPT or GitHub account to use the live page.

## Publish with GitHub Pages

1. Upload the files in this repository to the `main` branch.
2. Open **Settings → Pages** and select **Deploy from a branch**, branch `main`, folder `/(root)`.
3. Wait for GitHub Pages to publish. The public URL is `https://zhengtaiwei57-web.github.io/insar-sbas-studio/`.

Python and NumPy run in each visitor's browser using Pyodide 0.27.7 from jsDelivr. User-selected data stays in that browser. The first run downloads about 20 MB of runtime files.

## Features and limits

Visitors can run the synthetic demonstration, import an unwrapped `.npz` interferogram stack, invert the connected SBAS network, inspect four result layers, view a selected pixel's time series, and download NPZ, CSV, JSON, or PNG outputs. The browser version limits inputs to 25 MiB, 36 acquisition dates, 160 interferograms, and a maximum 512 by 512 pixel grid per stack dimension.

For local Sentinel-1 IW SLC processing, download the Python desktop program. Its GUI collects the SLC/DEM/AOI inputs and configures an ISCE2 topsStack → MintPy workflow with optional unwrapping-error, tropospheric-delay and residual-DEM corrections. Windows users need WSL2/Ubuntu and must install ISCE2, SNAPHU and MintPy there first. Raw scenes and results remain on the user's machine; this static page does not upload them or run a cloud job.

The browser page itself handles pre-unwrapped small stacks only; raw SLC focusing, interferogram generation, phase unwrapping, corrections and production-scale processing run in the local desktop program through the open-source engines.

Pyodide, CPython, and NumPy license notices are included in `runtime/`.