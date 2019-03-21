from sklearn.datasets import make_circles
from sklearn.cluster import KMeans
from machine_learning_algorithms_giuseppe.cluster import plot_points

nb_samples = 1000
X, y = make_circles(n_samples=nb_samples, noise=0.05)

plot_points(X, y)

km = KMeans(n_clusters=2)
km.fit(X)
print(km.cluster_centers_)


y2 = km.predict(X)
plot_points(X, y2)
