"""
project: learning-pipelines
author: Iker De Loma-Osorio
date: 06/12/2020

This is the main script to use pipelines in Python.
Data source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=test.csv

"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load train and test datasets
path = r"C:\Users\iker.delomaosorio\Documents\Datasets\HousePricesAdvancedRegressionTechniques_Kaggle"

file_train = "train.csv"
file_test = "test.csv"

train = pd.read_csv(path + "\\" + file_train)
test = pd.read_csv(path + "\\" + file_test)

# Check if there are missing values
print("Missing values in train:", train.isnull().sum().sum())
print("Missing values in test:", test.isnull().sum().sum())

# Check if there are missing values in the target (SalePrice) variable
# in the training dataset
print("Missing values in SalePrice:", train.SalePrice.isnull().sum())

# Separate the target variables from the predictors
y_train = train.SalePrice
X_train_full = train.drop(['SalePrice'], axis=1)

# Check the amount of numerical and categorical variables
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
print(type(numerical_cols))

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_test = test[my_cols].copy()

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')


