# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:59:29 2019

@author: Saksham
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset= pd.read_csv("Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#Taking care of mising data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

print("sak")