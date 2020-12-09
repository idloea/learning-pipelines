"""
project: learning-pipelines
author: Iker De Loma-Osorio
date: 06/12/2020

This is the main script to use pipelines in Python.
Data source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=test.csv
Resource: https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf

"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

# Load train and test datasets

path = r"C:\Users\iker.delomaosorio\Documents\Datasets\HousePricesAdvancedRegressionTechniques_Kaggle"

file_train = "train.csv"
file_test = "test.csv"

train = pd.read_csv(path + "\\" + file_train)
test = pd.read_csv(path + "\\" + file_test)

# Remove the Id column
train = train.drop('Id', axis=1)

# Remove the target column (SalePrice) from the other columns
X = train.drop('SalePrice', axis=1)
y = train.SalePrice
print("Missing values in the features:", X.isnull().sum().sum())
print("Missing values in target:", y.isnull().sum())

# Split the training dataset into train and test to later validate the performance of the models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Divide the features by type
print("Unique data types in the train dataset:", train.dtypes.unique())
# There are numerical (int64 and float64) and categorical (object) types
# Select the numeric features (without the target)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
# Select the non-numeric (categorical) columns
categorical_features = X.select_dtypes(include=['object']).columns

# Transformers
numeric_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Let's preprocess the features
preprocesor = ColumnTransformer(transformers=[
    ('numerical', numeric_transformer, numeric_features),
    ('categorical', categorical_transformer, categorical_features)])

# Machine learning algorithm model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Create the pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocesor),
    ('model', model)])

# Fit the pipeline
my_pipeline.fit(X_train, y_train)

# Predictions
predictions = my_pipeline.predict(X_test)

# Evaluate the model
score = mean_absolute_error(y_test, predictions)
print("MAE:", score)




# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
# categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and
#                     X_train[cname].dtype == "object"]

