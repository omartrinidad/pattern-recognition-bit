import numpy as np

k = 3
X = np.genfromtxt("data/data-clustering-1.csv", dtype=float, delimiter=',')
X = X.T

n_examples = X.shape[0]
n_features = X.shape[1]
n_classes = k

indices = np.arange(n_examples)

# Initialization
closest_cluster = np.random.randint(k, size=n_examples)
C = [indices[np.where(closest_cluster == i)] for i in range(k)]
u = np.vstack([[np.mean(X[C[i], :], axis=0)] for i in range(k)])

# Loop
while(True):
  converged = True
  for i in range(n_examples):
    C_i = closest_cluster[i]
    C[C_i] = np.delete(C[C_i], np.argwhere(C[C_i]==i))
    u[C_i, :] = np.mean(X[C[C_i], :], axis=0)
    distance_to_centroids = np.linalg.norm(X[i]-u, axis=1)
    C_w = np.argmin(distance_to_centroids, axis=0)
    if(C_w != C_i): converged = False
    closest_cluster[i] = C_w
    C[C_w] = np.append(C[C_w], i)
    u[C_w, :] = np.mean(X[C[C_w], :], axis=0)
  if(converged): break
