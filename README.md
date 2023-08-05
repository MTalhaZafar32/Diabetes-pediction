# Diabetes Prediction using XGBoost Model
This repository contains a project focused on predicting diabetes in patients based on various factors using the XGBoost machine learning model. The dataset consists of information from female patients of Pima Indian heritage and includes features such as pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age.

## Project Overview
The primary objective of this project is to build a predictive model that can effectively identify individuals at risk of diabetes based on their health attributes. The project comprises the following key components:

## Data Collection:
The dataset used for this project was obtained from a study involving female patients of Pima Indian heritage. It contains relevant health features that play a role in diabetes prediction.

## Data Cleaning: 
After obtaining the dataset, we performed data cleaning to ensure consistency, handle missing values, and remove any irrelevant information. The cleaned data forms the foundation for building the predictive model.

## Exploratory Data Analysis (EDA):
In the exploratory data analysis phase, we conducted descriptive statistics and visualizations to gain insights into the distribution and relationships between various features in the dataset. This step helped us understand the data and identify any patterns that might be relevant for the prediction model.

## Model Training:
For predicting diabetes, we opted to use the XGBoost model, a popular gradient boosting algorithm known for its effectiveness in classification tasks. We trained the XGBoost model on the cleaned dataset and fine-tuned its hyperparameters to achieve optimal performance.

## Test Results:
After training the XGBoost model, we evaluated its performance on a separate test dataset. The test results were saved in the test_results.csv file for further analysis.

## Repository Structure
The repository is organized into the following structure:

```
Diabetes-Prediction-Project/
  ├── data/
  │   ├── cleaned_data.csv
  │   ├── test_results.csv
  │
  ├── notebooks/
  │   ├── data_cleaning.ipynb
  │   ├── exploratory_analysis.ipynb
  │   └── model_training.ipynb
  │
  ├── README.md
  └── .gitignore
```
## data/:
This directory contains the cleaned data (cleaned_data.csv) used for training the XGBoost model and the test results (test_results.csv) obtained after evaluating the model on a separate test dataset.

## notebooks/:
This directory contains Jupyter notebooks used for data cleaning (data_cleaning.ipynb), exploratory data analysis (exploratory_analysis.ipynb), and model training (model_training.ipynb).

## Getting Started
To reproduce the analysis or build upon it, follow these steps:

1. Clone the repository to your local machine.
2. git clone (https://github.com/MTalhaZafar32/Diabetes-pediction.git)
3. Explore the data/ directory to find the cleaned dataset `(cleaned_data.csv)` used for training the XGBoost model, and the test results `(test_results.csv)` obtained after evaluation.
4. Open the Jupyter notebooks in the notebooks/ directory to examine the data cleaning, exploratory data analysis, and model training code.
5. Use the trained XGBoost model and test results to predict diabetes in new patients or make improvements to the existing model.

Contributing
We welcome contributions to enhance the project further. If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

License
This project is licensed under the `MIT License` - see the LICENSE file for details
