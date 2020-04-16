# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Split Training and Testing datasets
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

Y_Pred = regressor.predict(X_Test)

plt.scatter(X_Train, Y_Train, color = 'Red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'Blue')
plt.title('Experience vs Salary (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_Test, Y_Test, color = 'Red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'Blue')
plt.title('Experience vs Salary (Testing Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()