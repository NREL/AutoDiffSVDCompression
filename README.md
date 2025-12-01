# AutoDiffSVDCompression

The ImplicitAD.jl package is used and redistributed under the MIT license which is reproduced in the ThirdPartyLicense directory (as well as the file ImplicitAD/License).  Thanks to Andrew Ning of BYU for use of this software.

NREL Software Record: SWR-25-162

# Paper Results Reproduction

The code in this repository was used to produce the results found in the paper: 

J. Maack and J. Allen, Reducing Memory Usage of Reverse Mode AD with SVD, SIAM Journal on Scientific Computing. Under Review.

The primary purpose of this repository is to document those results, and enable their reproduction. Below is an outline of the code as well as instructions for (hopefully) reproducing the results.

## Code Description

* `BurgersEquation` -- Code for numerical simulation of Burgers' equation. The numerical methods implemented are described in the paper. As cited in the paper, in depth discussion can be found in [Hesthaven, Jan S. Numerical methods for conservation laws: From analysis to algorithms. Society for Industrial and Applied Mathematics, 2017.](https://epubs.siam.org/doi/book/10.1137/1.9781611975109).
* `ImplicitAD` -- Modified version of ImplicitAD used for introducing SVD compression into AD process. For more details, see the [ImplicitAD documentation](https://flow.byu.edu/ImplicitAD.jl/dev/) and the paper [Ning, Andrew, and Taylor McDonnell. "Automating steady and unsteady adjoints: efficiently utilizing implicit and algorithmic differentiation." arXiv preprint arXiv:2306.15243 (2023).](https://arxiv.org/abs/2306.15243)
* `burgers_tests` -- scripts and Jupyter notebooks for running benchmarks
    * `results` -- Raw logs (.log), gradient values (.csv), error values (.csv), and memory usage (.txt). Files are divided between whether the results were generated on NREL's Kestrel supercomputer or locally. Numerical experiments for measuring run time and memory usage were conducted on Kestrel while evaluation of error values were run locally. See the section [Notes on Results](#notes-on-results) for more details.
    * `sandbox.ipynb` -- Jupyter notebook for debugging, 
    * `svd_results.ipynb` -- Jupyter notebook for generating plots used in the paper
    * `burgers_common.jl` -- Common functions used in notebooks and scripts. Some functions override functions in BurgersEquation package to so AD can be used
    * `burgers_gradient_error.jl` -- Script which computes error in the SVD compressed gradient by comparing to the standard reverse mode gradient computed with ReverseDiff.jl. See below for usage.
    * `burgers_gradient_error_copy.jl` -- Copy of the above script to enable running a second instance.
    * `burgers_large_scale.jl` -- Script which either runs optimization or gradient evaluation for a particular combination of AD method, gridsize, and target function.
    * `burgers_sweep.jl` -- Script for running a sweep through AD methods, gridsizes, and target functions. When run on a linux machine, the script will generate a slurm script for running each particular combination of AD method, gridsize, and target function, then submit these jobs to run. (This was used to run benchmarks on NREL's Kestrel supercomputer. Checking for a linux OS was a simple way of differentiating between Kestrel and one of the author's personal work laptops.) When running on a non-linux machine, the script launches a Julia subprocess which runs a single combination of AD method, gridsize, and target function. Once that subprocess completes, the script proceeds to the next (that is, it sweeps through the given parameter combinations in serial).
* `paper_figures` -- These are the figures (png files) used in the paper
* `burgers.ipynb` -- Jupyter notebook containing prototype a few tests for BurgersEquation code, and comparing optimization using SVD compression with other methods of computing derivatives. Briefly explores wavelets and Fourier series as other compression methods. Not directly used for results of the paper.

## Code Usage

Gradient error computations can be computed using the script `burgers_gradient_error.jl`. Grid sizes and target functions to sweep through are specified in the `main()` function by the variables `grid_sizes` and `targets`, respectively. The distributed code may be ignored as it largely slowed down the computations (at least on smaller systems). The `--nstart` input value specifies the number of random points at which to compute a gradient. The `--ndescent` input specifies the number of gradient descent steps to take after each randomly generated start point. So for each grid size and target function, the error in the gradient will be computed at `nstart * (ndescent + 1)` points. The SVD tolerance or number of singular values to use in can compression can also be set within the script. The script will save a CSV file containing the 2-norm and infinity-norm errors for each combination grid size and target function.

Time, memory usage, and optimization problems can be run using the script `burgers_sweep.jl`. The AD methods, grid sizes, and target functions to sweep through are specified by the variables `ad_modes`, `grid_sizes`, and `targets`, respectively. Specifying multiple seeds runs a gradient computation at or optimization starting from a random point generated using the each seed. This is useful for reducing noise in the memory and execution time measurements. Settting `optimization` to true will run an optimization for each AD mode, grid size, target, and seed. Setting `optimization` to false will run a single gradient evaluation for each AD mode, grid size, target, and seed. This script works one of two ways. For clusters using slurm, the `kestrel` variable can be set to true and the script will create and submit slurm scripts (note that information in the script headers is specified for using NREL's Kestrel supercomputer and will need to be changed) each of which run a single AD mode, grid size, target function, and seed configuration using the script `burgers_large_scale.jl`.  When the `kestrel` variable is false, the script will serially process each combination by launching a subprocess which runs the script `burgers_large_scale.jl` with the appropriate command line arguments. The SVD tolerance or number of singular values to use in can compression can also be set in the lower level script `burgers_large_scale.jl`.

## Notes on Result Files

Raw logs (.log), gradient values (.csv), error values (.csv), and memory usage (.txt). Files are divided between whether the results were generated on NREL's Kestrel supercomputer or locally. Numerical experiments for measuring run time and memory usage were conducted on Kestrel while evaluation of error values were run locally. The files are further subdivided based on the target functions described in the paper where "sin" corresponds to "smooth", "cliff" corresponds to "jump", and "weierstrass" corresponds to "weierstrass".

Results for the memory usage and run time can be found in the directory `results/kestrel/gradient` while optimization run results are stored in `results/kestrel/optimization`. The log files for these cases contain the parameters used for the result as well as the result file basename. The file names contain the AD mode, the target function, whether the run was an optimization or just gradient computation, and the grid size. The grid size is given in the form of 2^k where the value k is in the file name. In the file names, the sub-string `svdtol1e3` corresponds to a tolerance of 1e-3, `svdtol1e4` corresponds to a tolerance of 1e-4, `nsv1` corresponds to one singular value, and the absence of these strings corresponds to a tolerance of 1e-5.

Results for gradient error computation are stored in `results/local/error`. The log files are found in this top level directory while the results files are broken into the target function subdirectories. THe log files are separated by the SVD compression parameter settings where the labels `1em3` corresponds to a tolerance 1e-3, `1em4` corresponds to a tolernace of 1e-4, `nsv1` corresponds to one singular value, and the absence of a label corresponds to a tolerance of 1e-5.  The logs for this last case are split into 3 parts and labeled accordingly. The logs for the results in the paper are all found in part 1. (Parts 2 and 3 contain the output for larger grid size computations that were done separately due to run time. Again due to run time, these larger grid sizes were not run for different SVD compression parameters.)

NOTE: The directory `results/kestrel/optimization_sd1e-2` contains results not used in the paper. The initial random points were chosen with a smaller standard deviation than those used in the paper.
