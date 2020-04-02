#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:47:22 2020

@author: sahilgogna
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data
# keep this for template
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# taking care of missing data
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(x[:, 1:3])
x[:, 1:3]=missingvalues.transform(x[:, 1:3])

# dummmy variables - we have Strings in column 1, but we need only numbers.
# so we label them as 0,1,2,... but python will start comaparing thinking that 1 is greater than 0
# so we break the columns to dummy columns where each category represents a column

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_x = LabelEncoder()
x[:,0] = labelEncoder_x.fit_transform(x[:,0]) # this categroize the data

#breaking into columns as per category
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype = np.str)


labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

#splitting the data into training set and test set
#keep this for template
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


#feature scaling
#keep this for template
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

