## Notification 

A newer, more noise-robust, more accurate version to Bayesian-SINDy - `:star:` [ODR-BINDy](https://github.com/llfung/ODR-BINDy) `:star:`- is now available on [GitHub](https://github.com/llfung/ODR-BINDy). Go check it out!

# Bayesian Sparse Identification of Nonlinear Dynamics (Bayesian-SINDy) 

Model discovery from data for nonlinear dynamical systems using type-II maximum likelihood in the Bayesian statistics, i.e. maximising evidence (or marginal likelihood). 

Derived from the original [SINDy](https://doi.org/10.1073/pnas.1517384113) paper _"Discovering governing equations from data: Sparse identification of nonlinear dynamical systems"_ in the Proceedings of the National Academy of Sciences, **113**(15):3932-3937, 2016, by S. L. Brunton, J. L. Proctor, and J. N. Kutz.

This code is part of [this article](https://doi.org/10.1098/rspa.2024.0200) by L. Fung, U. Fasel and M. P. Juniper.

## Using the code

### Getting started (without `SparseBayes` )
1. Run one of the `.m` script in MATLAB at the top level folder and start exploring!

### (Optional) To run `SparseBayes` for comparison
1. Download the `SparseBayes` package [here](https://www.miketipping.com/downloads/SB2_Release_200.zip)
2. Unzip the folder and place it in this current folder.
3. Run one of the `.m` script at the top level folder.

## Note on dependency and license

This code is derived from the original code on [SINDy](https://doi.org/10.1073/pnas.1517384113) by Brunton, Proctor & Kutz (2016, PNAS). Some of the files (namely, `SparseGalerkin.m`, `poolData.m`, `poolDataList.m` and `sparsifyDynamics.m`) are directly copied from the original SINDy work, under the permission of the original authors. Please refer to the specific files or `LICENSE` to see the full attribution.

Comparison with `SparseBayes` is also included in the code. To respect the licensing of their code, these files are not included in this repo. To run those part of the code, please download them from [their web page](https://www.miketipping.com/downloads/SB2_Release_200.zip) according to the instructions above.

