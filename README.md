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

1. Selecting independent variant-gene pairs for model training

    SURGE optimization (ie. learning the SURGE latent contexts) requires an input expression matrix and genotype matrix. Both matrices should be of dimension NXT, where N is the number of RNA samples and T is the number of genome-wide independent variant gene pairs (ie. the number of eqtl tests). It is up to the user to select which variant-gene pairs to be used for model training. In general, we desire each variant-gene pair used in model training to be independent of one another because we want the SURGE to capture eQTL patterns that are persistent across the genome, not specific to a single gene or variant. Furthermore, it has been shown that standard eQTLs are more likely to be context-specific eqtls than random variant-gene pairs. Based on these two pieces of information, we recommned the following procedure for selecting variant-gene pairs for model training:

- Run standard eQTL analysis on your full data set. Assess genome-wide significance of this analysis according to a gene-level Bonferonni correction, followed by a genome-wide Benjamini-Hochberg correction (or whatever multiple testing correction method you like).
- Limit to eGenes (FDR < .05) and their top associated variant
- Take the top 2000 egenes and their top-associated variants as the variant-gene pairs for model training. Remove any variant-gene pairs where the variant is already mapped to another eGene.

Note: we recommend limiting to approximately 2000 variant-gene pairs for computational efficiency. 


1. Genotype matrix



