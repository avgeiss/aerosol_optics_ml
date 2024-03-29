## About

This code accompanies the manuscript: Geiss, A., Ma, P-L, Singh, B., and Hardin, J. C., (2022) "Emulating Aerosol Optics with Randomly Generated Neural Networks"

[![DOI](https://zenodo.org/badge/508153694.svg)](https://zenodo.org/badge/latestdoi/508153694)

**Abstract:** *Atmospheric aerosols have a substantial impact on climate and remain one of the largest sources of uncertainty in
climate forecasts. Accurate representation of their direct radiative effects is a crucial component of modern climate models.
Direct computation of the radiative properties of aerosols is far too computationally expensive to perform in a climate model
however, so optical properties are typically approximated using a parameterization. This work develops artificial neural net-
works (ANNs) capable of replacing the current aerosol optics parameterization used in the Energy Exascale Earth System5
Model (E3SM). A large training dataset is generated by using Mie code to directly compute the optical properties of a range
of atmospheric aerosol populations given a large variety of particle sizes, wavelengths, and refractive indices. Optimal neural
architectures for shortwave and longwave bands are identified by evaluating ANNs with randomly generated wirings. Ran-
domly generated deep ANNs are able to outperform conventional multi-layer perceptron style architectures with comparable
parameter counts. Finally, the ANN-based parameterization is found to dramatically outperform the current parameterization.10
The success of this approach makes possible the future inclusion of much more sophisticated representations of aerosol optics
in climate models that cannot be captured through simple expansion of the existing parameterization scheme.*

## Required Packages

-PyMieScatt (1.8.1.1)

-numpy (1.19.5)

-netCDF4 (1.5.4)

-tensorflow (2.7.0)

## Python Scripts

**cam_aero_optics.py** A Python implementation of the CAM5 aerosol optics parameterization described in Ghan and Zaveri, JGR, 2007.

**create_optics_tables.py** Subroutines that generate tables of aerosol optical properties integrated over a range of size distributions with the table bounds decided by the wavelenght bands and aerosol modes used by RRTMG and MAM in E3SM.

**eval_on_test_set.py** Evaluates the various optics schemes (chebyshev interpolation, lookup tables, and ANNs) on the randomly generated test data.
                        
**eval_on_valid_set.py** Evaluates the various optics ANNs on the validation data.

**generate_test_set.py** Generates a test set for evaluating optics schemes by randomly selecting query points to evaluate optics from continuous distributions (instead of the regular grids used for the training and validation data).
                        
**generate_train_set.py** Generates the training and validation sets used in the study. the numpy random seed is set to 123 to ensure the same validation split can be reproduced in the future. this script does not do any optics calculations, it simply loads the data in a pre-computed high-resolution optics table generated by 'create_optics_tables.py' and performs pre-processing and does a train/validation split.

**neural_networks.py** Code for building neural networks in keras. The RandomAnn class contains functions that can create randomly wired neural networks using keras and has some additional convenient functions: saving and loading the random networks without building a tensorflow graph, a training subroutine, a function to count the trainable parameters in the random network. The benchmark_mlp function builds a simple feed-forward fully connected network to use for comparison to the random networks. it allows for requesting a layer count and approximate total parameter count.

**optics_utils.py** This contains subroutines and important constants (such as EAMv1 spectral bands, particle radii ranges, modal size distribution log-standard deviations, and water density) that get used by both 'cam_aero_optics.py' and 'ann_optics_table.py.' The subroutines are for reading rrtmg optics files, integrating optical properties over particle size distributions, and a wrapper for MieQ.
			
**train_random.py** Training script for the random anns. Generates and trains a new random architecture each time it is called.

**train_benchmark.py** Training script for the benchmark FF-MLPs.

**train_final.py** Performs a final round of training on select ANN architectures. Uses both the training and validation datasets and trains for twice as long.
			
			
## Directories
			
**./data/** directory where scripts expect data to be stored. an image of this directory at the time of initial submission of this project for publication is available through Zenodo:  https://doi.org/10.5281/zenodo.6762700

**./figures/** Contains copies of the figures from the manuscript

**./plotting_scripts/** Contains python scripts used to generate Figures 2, 4, and 5 in the manuscript

**./mie_codes/** Archives copies of the miev0.f and mie.py scripts used to perform Mie scattering calculations. Currently, the full PyMieScatt package is available [here](https://github.com/bsumlin/PyMieScatt) (Accessed Feb 8th 2022), and miev0 can be found [here](https://www.cesm.ucar.edu/models/cesm1.0/cesm/cesmBbrowser/html_code/cam/miesubs.F.html##MIEV0) (Accessed April 25th 2022).
