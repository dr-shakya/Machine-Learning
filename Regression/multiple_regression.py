#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 10:17:41 2018

@author: shakyadigbijaya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#importing dataset
dataset = pd.read_csv('winequality-white.csv', delimiter=';')
#dataset.head()
#dataset.info()
#dataset.describe()
sns.pairplot(dataset)

#checking for missing values
sns.heatmap(dataset.isnull())


'''
###Uncomment to perform backward elimination###

#before backward elimination
X= dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values
#OR
X_drop = dataset.drop('quality', axis=1)
y_drop = dataset['quality']
#for backward elimination
#select Significance Level(SL) = 0.5
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((4898, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 2, 4, 6, 7, 8, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 2, 4, 6, 8, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
'''

#after backward elimination
X = dataset.iloc[:, [0, 1, 2, 4, 6, 7, 8, 9, 10, 11]].values
y = dataset.iloc[:, 11].values
#y = y.astype(np.float64)

#no encoding categorical data

#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred = np.round_(y_pred) #to round off the values

#prediction analysis
#y_testf = y_test.astype(float)
sns.distplot((y_test-y_pred),kde=False)

#print(y_test)

# multiple linear regression
from sklearn import metrics
#print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.scatter(y_test, y_pred)
plt.xlabel('True Quality')
plt.ylabel('Predicted Quality')
plt.title('Predicted Quality Against True Quality ')
plt.show()

accuracy = regressor.score(X_test, y_test)
print("Accuracy: {}%".format(int(accuracy * 100)))