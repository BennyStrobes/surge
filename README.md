# SURGE: Single-cell Unsupervised Regulation of Gene Expression

This repository contains code for fitting and plotting results of the SURGE model. SURGE is a a probabilistic model that attempts to uncover context specific expression quantitative trait loci (eQTLs) withouth pre-specifying contexts of interest. SURGE achieves this goal by jointly learning unobserved contexts as well as the as the eQTL effects izes corresponding to those contexts in an unsupervised fashion. 

Contact Ben Strober (bstrober3@gmail.com) with any questions.


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

surge_obj = surge.surge_inference.SURGE_VI(K=20, max_iter=3000, delta_elbo_threshold=1e-2, re_boolean=True)
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

surge_obj = surge.surge_inference.SURGE_VI(K=20, max_iter=3000, delta_elbo_threshold=1e-2, re_boolean=False)
surge_obj.fit(G=G, Y=Y, cov=cov)

# downstream analysis of model fitting
...

```

### Input data:

**1. Selecting independent variant-gene pairs for model training**

   SURGE optimization (ie. learning the SURGE latent contexts) requires an input expression matrix and genotype matrix. Both matrices should be of dimension NXT, where N is the number of RNA samples and T is the number of genome-wide independent variant gene pairs (ie. the number of eqtl tests). It is up to the user to select which variant-gene pairs to be used for model training. In general, we desire each variant-gene pair used in model training to be independent of one another because we want the SURGE to capture eQTL patterns that are persistent across the genome, not specific to a single gene or variant. Furthermore, it has been shown that standard eQTLs are more likely to be context-specific eqtls than random variant-gene pairs. Based on these two pieces of information, we recommned the following procedure for selecting variant-gene pairs for model training:

   - Run standard eQTL analysis on your full data set. Assess genome-wide significance of this analysis according to a gene-level Bonferonni correction, followed by a genome-wide Benjamini-Hochberg correction (or whatever multiple testing correction method you like).
   - Limit to eGenes (FDR < .05) and their top associated variant
   - Take the top 2000 egenes and their top-associated variants as the variant-gene pairs for model training. Remove any variant-gene pairs where the variant is already mapped to another eGene. Note: we recommend limiting to approximately 2000 variant-gene pairs for computational efficiency. 


**2. Standardized Expression Matrix (Y)**

   After completing the step 1, you should have generated a set of variant-gene pairs used for model training. Now, you need create a 2-dimensional numpy array (Y) that contains expression information for those variant-gene pairs. Y should be of dimension NXT where N is the number of RNA samples and T is the number of variant-gene pairs. A column of Y reflects the standardized expression of the gene corresponding to a particular variant-gene pair. We expect Y to be standardized, meaning each column has mean 0 and variance 1.

**3. Standardized Genotype Matrix (G)**

   After completing the step 1, you should have generated a set of variant-gene pairs used for model training. Now, you need create a 2-dimensional numpy array (G) that contains genotype information for those variant-gene pairs. G should be of dimension NXT where N is the number of RNA samples and T is the number of variant-gene pairs. A column of G reflects standardized genotype of a variant corresponding to a particular variant gene pair. To standardize the genotype of the variant corresponding to test t, we center the genotype vector to have mean 0 across samples and then we scale the genotype vector for test t (G_(*t)) by the standard deviation of Y_(*t)/G_(*t). This scaling encourages the low-dimensional factorization (UV) to explain variance equally across tests instead of preferentially explaining variance in tests with small variance in Y_(*t)/G_(*t).

**4. Covariate matrix (cov)**

   You need create a 2-dimensional numpy array (cov) that contains covariates you wish to control for in your eQTL analysis. Standard covariates to include are expression pcs (or peer factors), genotype PCs, age, sex, batch, etc. cov should be of dimension NXL where N is the number of RNA samples and L is the number of covariates. One of the columns must be a column of ones corresponding to the intercept.

**5. Sample repeat array (z)**

   If you wish to control for sample repeat structure in your data, you need create a 1-dimensional numpy array (z) that contains the sample repeat structure present in your data. Specifically, z should be an array of length N where N is the numnber of samples. Each element of z should be an integer corresponding to which individual that RNA sample came from. For example `z = np.asarray[0, 0, 1, 1]` means you have expression data from 4 RNA samples and the first two RNA samples came from the same individual and the last two RNA samples came from the same individual. (Obviously 4 RNA samples is too few to run eQTL analysis. This is just there for demonstration purposes.)

### Model parameters:


**1. K**
    
   K is an integer that specifies the initial number of latent contexts. SURGE performs model selection by removing irrelevent contexts during optimization. This only works if K is set to be larger than the number of underlying latent contexts. In practice, setting `K=20` is a reasonable size. If SURGE converges, and all 20 latent contexts have PVE >= 1e-4, try setting K to be larger.

**2. max_iter**
    
   Maximum number of iterations to perform in variational optimization. Setting `max_iter=3000` is our recommendation.

**3. delta_elbo_threshold**
    
   This is the threshold used to assess model convergence. If the change in elbo from the previous iteration to the current iteration is less than delta_elbo_threshold, then SURGE converges. In practice, we found `delta_elbo_threshold=1e-2` works well in most situations. However, there is a natural give and take here. A smaller delta_elbo_threshold will give more accurate approximations, but it will take longer to converge. Whereas, a larger delta_elbo_threshold will give less accurate approximations, but it will converge faster.

**4. re_boolean**
   
   This is the boolean indicator dictating whether to model the effects of sample repeat structure with a random effects intercept. Set `re_boolean=True` if you wish to model sample repeat structure, or set `re_boolean=False` if you wish to ignore sample repeat structure. If you set `re_boolean=True` you must include the sample repeat array (z; see above).