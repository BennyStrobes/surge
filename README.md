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
To run :
```
import surge.surge_inference

# prepare inputs for SURGE
...

# initialize and fit model

surge_obj = surge.surge_inference.SURGE_VI(K=20, max_iter=3000, re_boolean=True, delta_elbo_threshold=1e-2)
surge_obj.fit(G=G, Y=Y, z=Z, cov=cov)

# downstream analysis of model fitting
...

```







