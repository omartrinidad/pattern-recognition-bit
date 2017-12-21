import numpy as np
import matplotlib.pyplot as plt

def vandermonde(x, d):
  # Expanding x in Vandermonde Matrix: f(x) = 1 + x + x^2 ... + x^d
  X = np.matrix(np.polynomial.polynomial.polyvander(x, d))
  return X

def polynomial_regression(x, y, d):
  # Expanding x in Vandermonde Matrix
  X = vandermonde(x, d)
  # Calculating model weights: W = Inv(X.T * X) * X.T * Y
  #w = np.linalg.inv(np.transpose(X) * X) * np.transpose(X) * np.transpose(np.matrix(y))
  w = np.linalg.lstsq(X, y)[0]
  return np.transpose(np.matrix(w))

def bayesian_regression(x, y, d, var_o):
  # Expanding x in Vandermonde Matrix
  X = vandermonde(x, d)
  # Calculating y variance
  var_y = np.var(y)
  # Calculating model weights: W = Inv(X.T * X) * X.T * Y
  w = np.linalg.inv(np.transpose(X) * X + var_y / var_o * np.identity(X.shape[1])) * np.transpose(X) * np.transpose(np.matrix(y))
  return w

# Importing dataset
## Dataset path
dataset_path = 'whData.dat'
## Column names
dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
## Loading the dataset
dataset = np.loadtxt(dataset_path, dtype=dt, comments='#', delimiter=None)
## Loading dataset without outliers
dataset_without_outliers = dataset[np.where(dataset['w'] > 0)]
## Loading dataset outliers (w < 0)
dataset_outliers = dataset[np.where(dataset['w'] < 0)]

# Heights from the training data
train_data = dataset_without_outliers['h']
# Weights from the training data
train_labels = dataset_without_outliers['w']
# Heights to predict their corresponding weights
test_data = dataset_outliers['h']

# Regression line X points to draw the model fitted line
regressionX = np.linspace(np.amin(train_data)-5, np.amax(train_data)+5, 1000)


# Regression for d = 5
d = 5
# Expand regression line X points in Vandermonde Matrix
X_regression = vandermonde(regressionX, d)
# Expand outlier height points in Vandermonde Matrix
X_test = vandermonde(test_data, d)

# Polynomial Regression
## Calculating the coefficients (weights) of the polynomial regression fitted line
Wp = polynomial_regression(train_data, train_labels, d)
# Calculating the fitted line Y points
yp = X_regression * Wp
# Calculating the corresponding weights (prediction values)
predictions = X_test * Wp

## Calculating the coefficients (weights) of the bayesian regression fitted line
var_o = 3.0
Wb = bayesian_regression(train_data, train_labels, d, var_o)
# Calculating the fitted line Y points
yb = X_regression * Wb
# Calculating the corresponding weights (prediction values)
predictionsb = X_test * Wb

# Plot the results
## Plot the data
plt.plot(train_data, train_labels, 'k.', label='Data')
## Plot the polynomial fitted line
plt.plot(regressionX, yp, 'b-', label='Polynomial Fit')
## plot the bayesian fitted line
plt.plot(regressionX, yb, 'g-', label='Bayesian Regression')
## Setting axes limits
plt.xlim(np.amin(train_data)-5, np.amax(train_data)+5)
plt.ylim(np.amin(train_labels)-5, np.amax(train_labels)+5)
plt.legend(loc='lower right')

plt.show()
