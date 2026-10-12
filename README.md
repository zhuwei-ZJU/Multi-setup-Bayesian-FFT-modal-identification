# Multi-setup Bayesian FFT modal identification

MATLAB code for identifying multiple, possibly closely spaced modes from
multi-setup ambient vibration data. A short EM stage initializes a Newton
optimization of the Bayesian FFT likelihood. The code returns the most
probable modal parameters and a Laplace-approximate posterior covariance.

Tested with MATLAB R2024b. Add this folder and its subfolders to the path:

addpath(genpath('path/to/bayoma_EMNewton_multi'));

## Input

out = bayoma_EMNewton_multi(in) with these fields:

| Field | Meaning |
| --- | --- |
| `in.data{r}.tdata` | Time samples × channels for setup `r` |
| `in.data{r}.fs` | Sampling rate in Hz |
| `in.mode{r}.f1f2` | One `[lower upper]` frequency-band row per band, in Hz |
| `in.mode{r}.f0` | Cell array of initial frequencies, one vector per band |
| `in.selmat{r}` | Channels × global DOFs selection matrix |

`tdata` should contain acceleration responses;
Each row of `selmat{r}` must contain one `1`, mapping a column of `tdata`
to its global DOF. All setups must share the same global DOF ordering and be
connected by overlapping measured DOFs. `in.mode` may contain one entry
shared by all setups or one entry per setup. Select bands and initial
frequencies from the PSD and singular-value spectra before running the code.

## Synthetic-data example

The released synthetic dataset contains `data01.mat` (`tdata`, `fs`) and
`band.mat` (`f1f2`, `f0`). The following reconstructs its 29 setups and
their selection matrices, then runs the first band. Remove `in.iband = 1`
to analyze all bands.
dataDir = 'path/to/synthetic-data';
in = struct();
D = load(fullfile(dataDir, 'data01.mat'));
B = load(fullfile(dataDir, 'band.mat'));

floorDofs = reshape(1:size(D.tdata, 2), 8, []);
ns = size(floorDofs, 2) - 1;
in.mode = {struct('f1f2', B.f1f2, 'f0', {B.f0})};

for r = 1:ns
    dofs = [floorDofs(:, end); floorDofs(:, end-r)];
    in.data{r}.tdata = D.tdata(:, dofs);
    in.data{r}.fs = D.fs;
    in.selmat{r} = sparse((1:numel(dofs)).', dofs, 1, ...
                          numel(dofs), size(D.tdata, 2));
end

in.iband = 1; % optional
out = bayoma_EMNewton_multi(in);

## SEG Plaza example

The SEG Plaza dataset has 12 setups, each with its own band file and a
selection-matrix file containing `C{r}`:

dataDir = 'path/to/SEG-Plaza-data';
in = struct();
M = load(fullfile(dataDir, 'Selection matrix.mat'));
in.selmat = M.C;

for r = 1:numel(in.selmat)
    D = load(fullfile(dataDir, sprintf('setup%02d.mat', r)));
    B = load(fullfile(dataDir, 'band', sprintf('band_setup%02d.mat', r)));
    in.data{r}.tdata = D.tdata;
    in.data{r}.fs = D.fs;
    in.mode{r}.f1f2 = B.f1f2;
    in.mode{r}.f0 = B.f0;
end

out = bayoma_EMNewton_multi(in);

## Output

`out.Setup01`, `out.Setup02`, etc. contain setup-wise frequencies (`f`, Hz),
damping ratios (`z`, fractions), local mode shapes (`phi`), modal-force PSD
(`S`), prediction-error PSD (`Se`), and posterior coefficients of variation
(`coefv`). `out.PHI` contains the global mode shapes; `out.PCM{ib}` is the
posterior covariance matrix for band `ib`. Iteration histories and timings
are also returned. Optional controls include `iband`, `tol_cvg`,
`maxEMIter`, `maxNewtonIter`, and `dispsetup`.

## Data

- Synthetic dataset: https://doi.org/10.7910/DVN/IVJCRU
- SEG Plaza dataset: https://doi.org/10.7910/DVN/PVXGGW

## Reference
Zhu W, Li B, Yang D H, et al. Multi-setup Bayesian FFT modal identification with multiple modes leveraging on second order information[J]. Mechanical Systems and Signal Processing, 2026, 261: 115020. https://doi.org/10.1016/j.ymssp.2026.115020
