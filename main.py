"""
project: learning-pipelines
author: Iker De Loma-Osorio
date: 06/12/2020

This is the main script to use pipelines in Python.
Data source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=test.csv

"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load train and test datasets
# path = r"C:\Users\iker.delomaosorio\Documents\Datasets\HousePricesAdvancedRegressionTechniques_Kaggle"
path = input("Select the dataset path:")
file = input("Introduce the CSV file name (Example: data.csv):")
df = pd.read_csv(path + "\\" + file)

# Check if there are missing values
print("Missing values in train:", df.isnull().sum().sum())

# Check if there are missing values in the target (SalePrice) variable
# in the training dataset
print("Missing values in SalePrice:", df.SalePrice.isnull().sum())

# Separate the target variables from the predictors
y = df.SalePrice
X = df.drop(['SalePrice'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                              random_state=0)


# Check the amount of numerical and categorical variables
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_test_full[my_cols].copy()

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Create and evaluate the pipeline
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                              ])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
predictions = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_test, predictions)
print('MAE:', score)
