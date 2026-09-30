# Code for the figures of *Matching Spatial and Temporal Boundary Conditions for Low-k Modes*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23065744.svg)](https://doi.org/10.5281/zenodo.23065744)

Code used to produce the figures in the paper *Matching Spatial and Temporal
Boundary Conditions for Low-k Modes* by Wei-Ning Deng and Will Handley.

## Repository layout

```
Cosmo_pert_eq/
├── find_best-fit_params/   best-fit cosmological parameter search (Planck data)
├── linear_combine_sol/     perturbation solutions and their linear combinations
│   └── data/               ← .npy data files from Zenodo go here
└── PPS_CMB/                primordial power spectrum and CMB spectra (CLASS)
Solve_KG_eq/                Klein–Gordon equation solver and figures
```

## Data

The `.npy` data files (about 2.7 GB) are too large for GitHub and are archived
on Zenodo: <https://doi.org/10.5281/zenodo.23065744>.

To restore them, download `code_for_figures_matchTwo_BC_data.zip` from the
Zenodo record and unzip it in the directory **containing** this repository:

```bash
git clone https://github.com/WeininJoy/code_for_figures_matchTwo_BC.git
unzip code_for_figures_matchTwo_BC_data.zip   # run next to the cloned folder
```

The files will be placed under
`code_for_figures_matchTwo_BC/Cosmo_pert_eq/linear_combine_sol/data/`.

## Citation

If you use this code or data, please cite the paper and the Zenodo record:

> Deng, W.-N. & Handley, W. (2026). *Code and data for the paper: Matching
> Spatial and Temporal Boundary Conditions for Low-k Modes*. Zenodo.
> https://doi.org/10.5281/zenodo.23065744

## License

The code and data are released under the
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.
