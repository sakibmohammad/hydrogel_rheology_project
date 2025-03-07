# Leveraging Deep Learning and Generative AI for Predicting Rheological and Material Composition for Additive Manufacturing of Polyacrylamide Hydrogels

This is a repository that contains the code and data of the following paper:  
📄 **[DOI: https://doi.org/10.3390/gels10100660](https://doi.org/10.3390/gels10100660)**  

---

## 📜 **Abstract**
Artificial intelligence (AI) has the ability to predict rheological properties and constituent composition of **3D-printed materials** with appropriately trained models. However, these models are not currently available for use.  

In this work, we trained deep learning (DL) models to:  
1. **Predict rheological properties** such as the **storage (G')** and **loss (G'') modulus** of 3D-printed polyacrylamide (PAA) substrates.  
2. **Predict material composition and 3D printing parameters** for a **desired pair of G' and G''**.  

We employed a **multilayer perceptron (MLP)** and successfully predicted G' and G'' from seven gel constituent parameters in a **multivariate regression process**. We then adopted two generative DL models (**VAE & CVAE**) to learn data patterns and generate synthetic compositions.  

📌 Our trained DL models were successful in mapping the **input–output relationship** for **3D-printed hydrogel substrates**, enabling prediction of **multiple variables from a handful of input variables and vice versa**.

---

## 📁 **Repository Structure**
```bash
/hydrogel_rheology_project
│── app.py                   # Gradio app for running models
│── model.py                 # Model architecture & loading functions
│── requirements.txt         # Required packages for setup
│── README.md                # Project documentation
│── weights/                 # Stores trained model weights
│   ├── model_regression.pth
│   ├── model_cvae.pth
│── scalers/                 # Stores scaler files for preprocessing
│   ├── scaler_regression.pkl
│   ├── scaler_cvae_x.pkl
│   ├── scaler_cvae_y.pkl
│── Data/                    # Data used for training/testing
│── Notebooks/               # Jupyter notebooks for experiments
│   ├── Hydro_gen_test_paper.ipynb
│   ├── Hydrogel_MLP.ipynb
│   ├── Stats_hydrogel_paper.ipynb

python3 -m venv env
source env/bin/activate  # (Linux/Mac)
env\Scripts\activate     # (Windows)

pip install --upgrade pip
pip install -r requirements.txt

python3 app.py

