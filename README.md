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
Below is an example of running SURGE while controlling for known covariates and sample repeat structure:
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

Below is an example of running SURGE while controlling for known covariates and without controlling for sample repeat structure:
```
import surge.surge_inference

# prepare inputs for SURGE
...

# initialize and fit model

surge_obj = surge.surge_inference.SURGE_VI(K=20, max_iter=3000, re_boolean=False, delta_elbo_threshold=1e-2)
surge_obj.fit(G=G, Y=Y, cov=cov)

# downstream analysis of model fitting
...

```

### Input data:

SURGE optimization (ie. learning the SURGE latent contexts) requires an input expression matrix and genotype matrix. Both matrices should be of dimension $N$X$T$, where N is the number of RNA samples and T is the number of genome-wide independent variant gene pairs. We desire each variant-gene pair to be independent of one another because we want the SURGE to capture eQTL patterns that are persistent across the genome, not specific to a single gene or variant.

Therefore, to encourage the expression and genotype data consists of independent variant-gene pairs we limit there to be a single variant-gene pair selected for each gene and limit there to be a single variant-gene pair selected for each variant. 

Furthermore, it has been shown that context-specific eQTLs are more likely to be standard eQTLs than not. Therefore, we limit variant-gene pairs used for SURGE optimization to those that are standard eQTLs within the data set.



1. Genotype matrix



