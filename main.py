"""
name: main.py
author: Iker De Loma-Osorio
date: 06/12/2020

This is the main script to use pipelines in Python

"""

import numpy as np
import pandas as pd

# Load data
path = r"C:\Users\Usuario\Documents\00_Iker\Datasets\CreditCardFraudDetection_Kaggle"
file = "creditcard.csv"
df = pd.read_csv(path + "\\" + file)

