# Leveraging Deep Learning and generative AI for predicting rheological and material composition for additive manufacturing of polyacrylamide hydrogels

This is a repository that contains the code and data of the following paper: https://doi.org/10.3390/gels10100660

## Abstract
Artificial intelligence (AI) has the ability to predict rheological properties and constituent composition of 3D-printed materials with appropriately trained models. However, these models are not currently available for use. In this work, we trained deep learning (DL) models to (1) predict the rheological properties, such as the storage (G’) and loss (G”) moduli, of 3D-printed polyacrylamide (PAA) substrates, and (2) predict the composition of materials and associated 3D printing parameters for a desired pair of G’ and G”. We employed a multilayer perceptron (MLP) and successfully predicted G’ and G” from seven gel constituent parameters in a multivariate regression process. We used a grid-search algorithm along with 10-fold cross validation to tune the hyperparameters of the MLP, and found the R2 value to be 0.89. Next, we adopted two generative DL models named variational autoencoder (VAE) and conditional variational autoencoder (CVAE) to learn data patterns and generate constituent compositions. With these generative models, we produced synthetic data with the same statistical distribution as the real data of actual hydrogel fabrication, which was then validated using Student’s t-test and an autoencoder (AE) anomaly detector. We found that none of the seven generated gel constituents were significantly different from the real data. Our trained DL models were successful in mapping the input–output relationship for the 3D-printed hydrogel substrates, which can predict multiple variables from a handful of input variables and vice versa.  

![Visual abstract](/asbtract_graphical.png)

# Repository structure

## Data
- Rheology data.xlxs

## Notebooks
- Hydro_gen_test_paper.ipynb
- Hydrogel_MLP.ipynb
- Stats_hydrogel_paper.ipynb
