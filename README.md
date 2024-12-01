# Heart-Disease-Prediction
This projects demonstrates the usage of generative AI and machine learning models to predict the presence of heart diseaase in an individual. We have used CTGAN, VAE and table diffusion as three techniques to generate our synthetic datasets and pre-tained models like Linear Regression, Decision Tree, KNN, SVM, XGBoost, CNN and an ensemble model to dtermine chances of an individual having heart disease based on tabular data. 

CONTENT:
1.Requirements
2.Introduction
3.Dataset
4.Synthetic Dataset Generation
5.Models
6.Training
7.Results and Conclusion

1. Requirements
   To execute this project, the following libraries and tools are required:
   Programming Language: Python 3.x
   
   Libraries:
pandas - Data manipulation and preprocessing
numpy - Numerical computations
scikit-learn - Machine learning models and evaluation
xgboost - Gradient boosting framework
tensorflow - Deep learning framework
imbalanced-learn - Dataset balancing (SMOTE)
seaborn and matplotlib - Visualization tools

   Dataset Files:
synthetic_data_ctgan.csv
synthetic_data_vae.csv
synthetic_data_table_diffusion.csv
heart.csv (original dataset)

Install the requirements using:
!pip install pandas numpy scikit-learn xgboost tensorflow imbalanced-learn seaborn matplotlib

2. Introduction
This project explores the use of synthetic data in building a Heart Disease Prediction System. By generating synthetic datasets with CTGAN, VAE, and Table Diffusion, we overcome data scarcity and imbalances in the original dataset. The goal is to train multiple machine learning models on the synthetic data and evaluate their performance on the original dataset to assess reliability and robustness.

3. Dataset
   
Original Dataset: A dataset containing clinical and demographic information for heart disease diagnosis has been taken kaggle containing
14 parameters like age, sex, blood pressure etc. with target 1 for heart disease, 0 for no disease.

Synthetic Datasets:
CTGAN: Data generated using a Conditional Tabular GAN.
VAE: Data generated using a Variational Autoencoder.
Table Diffusion: Data generated using a diffusion-based generative model

4. Synthetic Dataset Generation

Synthetic data generation using generative AI involves creating artificial datasets that mimic real-world data for use in training, testing, and validating machine learning models. By leveraging advanced generative models such as Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), or large language models, AI can produce highly realistic yet customizable data. This approach is particularly valuable in scenarios where real data is scarce, expensive to collect, or fraught with privacy concerns. Synthetic data enables organizations to enhance model performance, conduct robust testing, and mitigate biases while adhering to data protection regulations. Additionally, it supports innovation in fields like autonomous systems, healthcare, and finance, by providing diverse and scalable datasets.
This concept of generative AI has been used in this project to create synthetic datasets fro, the original dataset using the tools CTGAN, VAE and table diffusion.

5. Models

The following models have been employed in training the datasets generated and were tested against the original dataset.
a. Linear Regression (Ridge): Regularized linear model to prevent overfitting.
b. Decision Tree: Tree-based model with hyperparameter tuning using GridSearchCV.
c. K-Nearest Neighbors (KNN): Instance-based learning with optimal k selection.
d. Support Vector Machine (SVM): Kernel-based classification with grid search.
e. XGBoost: Gradient boosting for high accuracy and robustness.
f. Ensemble Model: Combines predictions from all models for improved performance.

6. Training

a. Preprocessing:
Missing values were imputed with mean values.
Outliers were removed using the Z-score method with threshold = 3.
Features were normalized using StandardScaler.
Class imbalance was addressed using SMOTE.

b. Synthetic Data Training:
Each model was trained using synthetic datasets (CTGAN, VAE, Table Diffusion).
Hyperparameter tuning was performed for optimal model configurations.

c. Testing:
Models were validated using the original dataset heart.csv.

7. Results and Conclusion

Synthetic datasets generated using advanced generative AI techniques significantly improve model training by addressing data imbalance and scarcity.
Models trained on Table Diffusion data showed the most consistent performance.
Ensemble models delivered higher accuracy and stability by combining multiple prediction mechanisms.
This work demonstrates the potential of synthetic data in healthcare machine learning applications.
