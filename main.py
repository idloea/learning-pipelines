"""
project: learning-pipelines
author: Iker De Loma-Osorio
date: 06/12/2020

This is the main script to use pipelines in Python.
Data source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=test.csv
Resource: https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf

"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# Remove the Id column, since it does not add any value
data = data.drop('Id', axis=1)

# Separate the dataset into features (predictors) and target (value to predict)
X = data.drop('SalePrice', axis=1)
y = data.SalePrice

# Check the missing values
print("Missing values in the features:", X.isnull().sum().sum())  # Some missing values
print("Missing values in target:", y.isnull().sum())  # No missing values

# Split the dataset into train and test to later validate the performance of the models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Check the type of features of the dataset
print("Unique data types in the train dataset:", data.dtypes.unique())
# There are numerical (int64 and float64) and categorical (object) types

# Select the numeric features. Do not consider the target,as it does not have any missing value
# Thus, use X instead of data dataframe
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Select the non-numeric (categorical) features
categorical_features = X.select_dtypes(include=['object']).columns

# The transformation will be made according to the data type of each feature
# For the numeric features imputing the missing values is enough
numeric_transformer = SimpleImputer(strategy='median')

# For the categorical features besides of imputation, one-hot encoding needs to be performed since the models
# only accept numerical data
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

# With more models
models = [KNeighborsClassifier(3),
          SVC(kernel="rbf", C=0.025, probability=True),
          RandomForestClassifier(n_estimators=100, random_state=0)]

# Go through the chosen models
score_list = []
for model in models:
    pipe = Pipeline(steps=[('preprocessor', preprocesor),
                           ('model', model)])
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    score = mean_absolute_error(y_test, predictions)
    score_list.append(score)

# Create a list of the models used
model_names = ['KNN', 'SVC', 'RandomForest']

# Create a dataframe with the scores
score_results = pd.DataFrame()
score_results['Model'] = model_names
score_results['MAE'] = score_list

# Sort the MAE values from smallest to bigger to rank the models from best to worse
print(score_results.sort_values(by='MAE'))
