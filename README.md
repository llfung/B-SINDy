# Bayesian Sparse Identification of Nonlinear Dynamics (Bayesian-SINDy) 
Model discovery from data for nonlinear dynamical systems using type-II maximum likelihood in the Bayesian statistics, i.e. maximising evidence (or marginal likelihood). 

Based on the original [SINDy](https://doi.org/10.1073/pnas.1517384113) by Brunton, Proctor & Kutz (2016, PNAS).

This code is part of [this article](https://arxiv.org/abs/2402.15357) by L. Fung, U. Fasel and M. P. Juniper.

## Using the code

1. Download `SINDy` [here](https://faculty.washington.edu/sbrunton/sparsedynamics.zip) (necessary for running the solver) and `SparseBayes` [here](https://www.miketipping.com/downloads/SB2_Release_200.zip) (optional, only for comparison with the `SparseBayes` package).
2. Unzip the folder and place it in this folder.
3. Run one of the `.m` script at the top level folder.

## Note on dependency and license

This code is derived from the original code on [SINDy](https://doi.org/10.1073/pnas.1517384113) by Brunton, Proctor & Kutz (2016, PNAS), and therefore require some files (namely, `SparseGalerkin.m`, `poolData.m` and `poolDataList.m`) from the original SINDy code to run. To compare Bayesian-SINDy with the original SINDy, the file `sparsifyDynamics.m` is also required. To respect the licensing of their code, these files are not included in this repo. Please download them from [their web page](https://faculty.washington.edu/sbrunton/sparsedynamics.zip) according to the instructions above.

Comparison with `SparseBayes` is also included in the code. To respect the licensing of their code, these files are not included in this repo. To run those part of the code, please download them from [their web page](https://www.miketipping.com/downloads/SB2_Release_200.zip) according to the instructions above.

