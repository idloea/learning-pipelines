"""
project: learning-pipelines
author: Iker De Loma-Osorio
date: 06/12/2020

This is the main script to use pipelines in Python.

Data sources: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

References:
https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf
https://www.kaggle.com/alexisbcook/pipelines

"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

# Load the dataset
path = r"C:\Users\iker.delomaosorio\Documents\Datasets\HousePricesAdvancedRegressionTechniques_Kaggle"
file = "train.csv"
data = pd.read_csv(path + "\\" + file)

# Remove the Id column, since it does not add any value for the later predictions
data = data.drop('Id', axis=1)

# Separate the dataset into features (predictors) and target (value to predict)
X = data.drop('SalePrice', axis=1)  # features
y = data.SalePrice  # target

# Check the missing values
print("Missing values in the features:", X.isnull().sum().sum())  # Some missing values
print("Missing values in target:", y.isnull().sum())  # No missing values

# Split the dataset into train and test to later validate the performance of the models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Check the type of features of the dataset
print("Unique data types in the train dataset:", data.dtypes.unique())
# There are numerical (int64 and float64) and categorical (object) features in this dataset

# Select the numeric features. Do not consider the target (y),as it does not have any missing value
# Thus, use X dataframe instead of data
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Select the non-numeric (categorical) features
categorical_features = X.select_dtypes(include=['object']).columns

# Each type of feature needs an individual transformation
# For the numeric features imputing the missing values (imputation) is enough
numeric_transformer = SimpleImputer(strategy='median')

# For the categorical features besides of imputation, one-hot encoding needs to be performed since the models
# only accept numerical inputs
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Group the numerical and categorical transformations as they belong to the preprocessing step of the whole process
preprocesor = ColumnTransformer(transformers=[
    ('numerical', numeric_transformer, numeric_features),
    ('categorical', categorical_transformer, categorical_features)])

# Create a list of Machine Learning models that will be fed with the preprocessed data
models = [KNeighborsClassifier(3),  # K-Nearest Neighbor model
          SVC(kernel="rbf", C=0.025, probability=True),  # Support Vector Classifier model
          RandomForestClassifier(n_estimators=100, random_state=0)]  # Random Forest Classifier model

# Go through the chosen models
score_list = []  # For storing the score results for each iteration inside the for loop
for model in models:
    # Preprocess and create the model with the preprocessed data
    pipe = Pipeline(steps=[('preprocessor', preprocesor),
                           ('model', model)])
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    score = mean_absolute_error(y_test, predictions)
    score_list.append(score)

# Create a list of the used models
model_names = ['KNN', 'SVC', 'RandomForest']

# Create a dataframe with the scores
score_results = pd.DataFrame()
score_results['Model'] = model_names
score_results['MAE'] = score_list

# Sort the MAE values from smallest to bigger to rank the models from best to worse
print(score_results.sort_values(by='MAE'))
