# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:53:31 2019

@author: Saksham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values 

# Fitting the Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300 ,random_state=0)
regressor.fit(X,y)

# Predicting a new result 
y_pred=regressor.predict(6.5)

# Visualising the Regression results
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()