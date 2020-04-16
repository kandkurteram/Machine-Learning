# Polynomial Regression
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fit Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Fit Polynomial Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_Poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_Poly, y)

# Visualize Linear Regression results
plt.scatter(X, y, color = 'Red')
plt.plot(X, lin_reg.predict(X), color = 'Blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel ('Level')
plt.ylabel('Salary')
plt.show()

# Visualize Polynomial Regression results
X_Grid = np.arange(min(X), max(X), 0.1)
X_Grid = X_Grid.reshape(len(X_Grid), 1)
plt.scatter(X, y, color = 'Red')
plt.plot(X_Grid, lin_reg2.predict(poly_reg.fit_transform(X_Grid)), color = 'Blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel ('Level')
plt.ylabel('Salary')
plt.show()

# Predict using Linear Regression Model
print(lin_reg.predict(6.5))

# Predict using Polynomial regression model
print(lin_reg2.predict(poly_reg.fit_transform(6.5)))