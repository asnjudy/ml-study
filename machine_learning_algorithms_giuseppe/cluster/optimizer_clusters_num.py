from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from machine_learning_algorithms_giuseppe.cluster import *

nb_samples = 1000
X, y = make_blobs(n_samples=nb_samples, n_features=2, centers=3, cluster_std=1.5)

plot_points(X, y)


def optimizer_inertia():
    """
    优化惯性
    :return:
    """
    nb_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    inertia = []

    for n in nb_clusters:
        km = KMeans(n_clusters=n)
        km.fit(X)
        inertia.append(km.inertia_)

    plot_bar(nb_clusters, inertia)


from sklearn.metrics import silhouette_score

nb_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
avg_silhouettes = []

for n in nb_clusters:
    km = KMeans(n_clusters=n)
    y2 = km.fit_predict(X)
    avg_silhouettes.append(silhouette_score(X, labels=y2))
plot_bar(nb_clusters, avg_silhouettes)
