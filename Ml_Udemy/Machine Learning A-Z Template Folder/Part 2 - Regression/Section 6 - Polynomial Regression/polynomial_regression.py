#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:33:16 2020

@author: sahilgogna
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data
dataset = pd.read_csv('Position_Salaries.csv')

# to have a look into the data I plotted it, it was a polynomial
# x should always be a matrix and y a vector, by this x is a vector size(10,)
# x = dataset.iloc[:, 1].values
# y = dataset.iloc[:, 2].values

# plt.scatter(x, y, color = 'red')
# plt.xlabel('Level')
# plt.ylabel('Salary')
# plt.show()

# this x is a matrix
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# we have very less data, so no test and train set

# fitting linear regression to the data set, this is only for the comparision
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


# fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# visualizing the linear regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Linear')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# visualising polynomial regression model
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Polynomial')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# predicting the new result with the Linear regression, assume level is 6.5
print(lin_reg.predict([[6.5]]))

# predicting the new result with the Linear regression, assume level is 6.5
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))




# (for higher resolution and smoother curve) replcae x by
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))


