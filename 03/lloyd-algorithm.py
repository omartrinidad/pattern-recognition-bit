import numpy as np

k = 3
X = np.genfromtxt("data-clustering-1.csv", dtype=float, delimiter=',')
X = X.T
max_iter = 100

n_examples = X.shape[0]
n_features = X.shape[1]
n_classes = k

# Initialization
t = 0
u = np.random.rand(k, n_features)
converged = False

indices = np.arange(n_examples)

# Loop
while(not(converged)):
  u_old = u
  distance_to_centroids = np.vstack([np.linalg.norm(X-u[i], axis=1) for i in range(k)])
  closest_cluster = np.argmin(distance_to_centroids, axis=0)
  C = np.array([indices[np.where(closest_cluster == i)] for i in range(k)])
  u = np.vstack([np.mean(X[C[i], :], axis=0)] for i in range(k))
  t = t + 1
  if(t==max_iter): converged = True
  if(np.array_equal(u_old,u)): converged = True
