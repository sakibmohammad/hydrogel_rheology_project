# Leveraging Deep Learning and generative AI for predicting rheological and material composition for additive manufacturing of polyacrylamide hydrogels

This is a repository that contains the code and data of the following paper:

## Abstract
Artificial intelligence (AI) has the ability to predict rheological properties and constituent composition of the 3D printed materials with appropriately trained models. However, these models are not currently available for use. In this work, we trained deep learning (DL) models to 1) predict rheological properties of 3D printed polyacrylamide (PAA) substrates such as the storage (G’) and loss (G”) moduli and 2) predict the composition of materials and associated printing parameters for additive manufacturing for a desired G’ and G’’. We employed a Multilayer Perceptron (MLP) and successfully predicted G’ and G” from seven gel constituent parameters in a multivariate regression process. We used a grid-search algorithm to tune the hyperparameters for the MLP and utilized a 10-fold cross validation and found the $R^2$ value to be 0.89. Next, we adopted two generative DL models named Variational Autoencoder (VAE) and Conditional Variational Autoencoder (CVAE) to learn data patterns and generate constituent composition. With these generative models, we produced synthetic data that has the same statistical distribution of the real data produced from actual hydrogel fabrication which was then validated our approach using t-test and an Autoencoder (AE) anomaly detector and found that none of the seven gel constituents were significantly different from the real data. Our trained DL models were successful in mapping the input-output relationship for the additive manufacturing of 3D printed hydrogel substrates which can predict multiple variables from a handful of input variables and vice-versa.  

![Visual abstract](/Data/visual_overview.png)

(a) Shows our method of data generation for AI model training. We 3D printed hydrogel substrates using different material composition, printing parameters, and tested them to find their rheological properties namely storage modulus (G') and loss modulus (G") at different frequencies. (b) An MLP regressor was used to predict the G' and G" from the hydrogel material constituents. (c) We predict the hydrogels constituents from G' and G". (d) Finally, we used VAE and CVAE which are generative DL models to generate hydrogel materials constituents that matched the original data.

# Repository structure

## Data
- Rheology data.xlxs
- visual_overview.png

## Notebooks
- Hydro_gen_paper_test.ipynb
- Hydrogel_MLP.ipynb
- Stats_hydrogel_paper.ipynb
