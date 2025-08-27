# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('cvd_2d_crystal_data.csv')
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 6].values
# print('matrix of features')
# print(X)

# print('Targeted Value')

# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


# normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)
# print(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(y_test)
print(y_pred)












