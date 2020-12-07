"""
project: learning-pipelines
author: Iker De Loma-Osorio
date: 06/12/2020

This is the main script to use pipelines in Python.
Data source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=test.csv

"""

import pandas as pd

# Load train and test datasets
path = r"C:\Users\iker.delomaosorio\Documents\Datasets\HousePricesAdvancedRegressionTechniques_Kaggle"

file_train = "train.csv"
file_test = "test.csv"

train = pd.read_csv(path + "\\" + file_train)
test = pd.read_csv(path + "\\" + file_test)


# Check if there are missing values
print("Missing values in train:", train.isnull().sum().sum())
print("Missing values in test:", test.isnull().sum().sum())

# Exploratory Data Analysis
print(train.head())
print(test.head())

