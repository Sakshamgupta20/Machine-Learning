# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:02:13 2019

@author: Saksham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor=LinearRegression()
linear_regressor.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualising the Linear Regression results
plt.scatter(X,y,color='red')
plt.plot(X,linear_regressor.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result in Linear Regression
linear_regressor.predict(6.5)

# Predicting a new result in Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))