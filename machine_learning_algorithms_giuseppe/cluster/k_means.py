from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from machine_learning_algorithms_giuseppe.cluster import plot_points

nb_samples = 1000
X, y = make_blobs(n_samples=nb_samples, n_features=2, centers=3, cluster_std=1.5)

plot_points(X, y)

km = KMeans(n_clusters=3)
km.fit(X)
print(km.cluster_centers_)

