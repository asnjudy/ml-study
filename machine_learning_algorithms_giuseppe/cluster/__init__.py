__all__ = ["plot_points", "plot_bar"]


def plot_points(X, y):
    from itertools import cycle
    import matplotlib.pyplot as plt

    colors = cycle('rgb')
    markers = cycle('.dx')
    labels = set(y)
    targets = range(len(labels))

    plt.figure()
    for target, color, label, marker in zip(targets, colors, labels, markers):
        plt.scatter(X[target == y, 0], X[target == y, 1], c=color, label=label, marker=marker)
    plt.show()


def plot_bar(X, Y):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(X, Y)
    plt.show()
