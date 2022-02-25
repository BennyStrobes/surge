# SURGE: Single-cell Unsupervised Regulation of Gene Expression

This repository contains code for fitting and plotting results of the SURGE model. SURGE is a a probabilistic model that attempts to uncover context specific expression quantitative trait loci (eQTLs) withouth pre-specifying contexts of interest. SURGE achieves this goal by jointly learning unobserved contexts as well as the as the eQTL effects izes corresponding to those contexts in an unsupervised fashion. 



## Install

```
git clone https://github.com/BennyStrobes/surge
cd surge
conda env create --file environment.yml  # create envrionment with dependencies
conda activate surge  # activate environment
pip install .  # install package in surge environment
```

## How to use

SURGE is implemented as a python package. Read below for a minimum example of how to use SURGE from within python. Look at `notebooks/SURGE_demo.ipynb` for a more in-depth example of how to interact with SURGE in Python.


### Python:
```
from cafeh.cafeh import fit_cafeh_genotype
from cafeh.fitting import weight_ard_active_fit_procedure

# prepare inputs for cafeh
...

# initialize and fit model
cafehg = fit_cafeh_genotype(X, y, K=10)

# downstream analysis of model fitting
...

```







