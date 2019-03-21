from machine_learning_algorithms_giuseppe.cluster import plot_points

from sklearn.datasets import make_moons

nb_samples = 1000
X, Y = make_moons(n_samples=nb_samples, noise=0.05)
plot_points(X, Y)


def k_means_cluster():
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=2)
    Y2 = km.fit_predict(X)
    plot_points(X, Y2)


def dbscan_cluster():
    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN(eps=0.1)
    Y2 = dbscan.fit_predict(X)
    plot_points(X, Y2)


dbscan_cluster()
