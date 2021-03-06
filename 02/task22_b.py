import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def gauss(x,y):
  exp_term = -1.0/(2*(1-corr_hw**2))*((x-mean_h)**2/var_h+(y-mean_w)**2/var_w-2*corr_hw*(x-mean_h)*(y-mean_w)/(std_h*std_w))
  return 1.0 / (2*np.pi*std_h*std_w*np.sqrt(1-corr_hw**2)) * np.exp(exp_term)

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

# Calculate the Covariance matrix
covariance_matrix = np.cov(np.array([train_data, train_labels]))
## Heights mean
mean_h = np.mean(train_data)
## Heights variance
var_h = covariance_matrix[0, 0]
## Heights standard deviation
std_h = np.sqrt(var_h)
## Weight mean
mean_w = np.mean(train_labels)
## Weight variance
var_w = covariance_matrix[1, 1]
## Weights standard deviation
std_w = np.sqrt(var_w)
## Correlation coefficient between heights and weights
corr_hw = covariance_matrix[0, 1] / (std_h * std_w)

# Calculate (predict) the corresponding weight for given height
f = lambda x : mean_w + corr_hw * std_w / std_h * (x - mean_h)

# Plot data
plt.plot(train_data, train_labels, 'k.', label='Data')
# Plot predicted value
plt.plot(test_data, f(test_data), 'r.', label='Predictions')
# Plot the model
## Grid XY points to build contour
x = np.linspace(np.amin(train_data)-5, np.amax(train_data)+5, 1000)
y = np.linspace(np.amin(train_labels)-5, np.amax(train_labels)+5, 1000)
X, Y = np.meshgrid(x, y)
zi = mlab.bivariate_normal(X, Y, std_h, std_w, mean_h, mean_w, covariance_matrix[0, 1])
## Contour the gridded data
plt.contour(x,y,zi)

plt.xlim(np.amin(train_data)-5, np.amax(train_data)+5)
plt.ylim(np.amin(train_labels)-5, np.amax(train_labels)+5)
plt.legend(loc='upper left')
plt.savefig("out/task22/bivariate_gaussian.png", bbox_inches="tight", pad_inches=0)
plt.show()
