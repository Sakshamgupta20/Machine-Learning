# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:55:40 2019

@author: Saksham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Data 
dataset=pd.read_csv("Salary_data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#Splitting the dataset into the Training set and Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Fitting Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set
y_pred=regressor.predict(X_test)

# Visualising the Training set
plt.scatter(X_train,y_train,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('Salary vs Experience v (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience v (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()