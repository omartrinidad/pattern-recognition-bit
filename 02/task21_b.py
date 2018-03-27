import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

def vandermonde(x, d):
  # Expanding x in Vandermonde Matrix: f(x) = 1 + x + x^2 ... + x^d
  X = np.matrix(np.polynomial.polynomial.polyvander(x, d))
  return X

def polynomial_regression(x, y, d):
  # Expanding x in Vandermonde Matrix
  X = vandermonde(x, d)
  # Calculating model weights: W = Inv(X.T * X) * X.T * Y
  w = np.linalg.inv(np.transpose(X) * X) * np.transpose(X) * np.transpose(np.matrix(y))
  return w

# Importing dataset
## Dataset path
dataset_path = 'data/whData.dat'
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

# Polynomial fitting for d = {1, 5, 10}
d = [1, 5, 10]
# Data mean(mu) and standard deviation (sigma)
mu = train_data.mean()
sigma = train_data.std()

for i in range(len(d)):
  # Standardize the data
  fig, ax = plt.subplots(1, 1, sharey=True)
  X_train = (train_data - mu) / sigma
  # Calculating the coefficients (weights) of the polynomial fitted line
  W = polynomial_regression(X_train, train_labels, d[i])
  # Standardize the regression line X points and expand them in Vandermonde Matrix
  X_regression = vandermonde((regressionX - mu) / sigma, d[i])
  # Calculating the fitted line Y points
  y = X_regression * W
  # Standardize the outlier height points and expand them in Vandermonde Matrix
  X_test = vandermonde((test_data - mu) / sigma, d[i])
  # Calculating the corresponding weights (prediction values)
  predictions = X_test * W
  # Plot the results
  ## Plot the data

  l1, = ax.plot(train_data, train_labels, 'ko', alpha=0.555, c="#2222ee")
  ## Plot the model line
  l2, = ax.plot(regressionX, y, 'b-')
  ## plot the predictions
  l3, = ax.plot(test_data, predictions, 'ro', alpha=0.555, c="#ee2222")

  ## Setting axes limits and title
  ax.set_xlim(np.amin(train_data)-5, np.amax(train_data)+5)
  ax.set_ylim(np.amin(train_labels)-5, np.amax(train_labels)+5)

  #fig.legend((l1, l2, l3), ('Data', 'Polynomial fitted line', 'Predictions'), 'lower right')
  ax.set_facecolor("#ffeeee")
  ax.set_xlim(150, 195)
  plt.xlabel("Height")
  plt.ylabel("Weight")

  plt.savefig("out/01/regression_d_{}.png".format(i), bbox_inches="tight", pad_inches=0)
  tikz_save("latex/regression_d_{}.tex".format(i))
  plt.show()
