from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier():
    from sklearn.datasets import load_digits

    digits = load_digits()

    nb_classifications = 100
    accuracy = []

    for i in range(1, nb_classifications):
        a = cross_val_score(RandomForestClassifier(n_estimators=i),
                            digits.data, digits.target, scoring='accuracy', cv=10).mean()
        accuracy.append(a)

    plt.figure()
    plt.plot(range(len(accuracy)), accuracy)
    plt.show()


def extra_trees_classifier():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.datasets import load_digits

    digits = load_digits()

    nb_classifications = 100
    accuracy = []

    for i in range(1, nb_classifications):
        a = cross_val_score(ExtraTreesClassifier(n_estimators=i),
                            digits.data, digits.target, scoring='accuracy', cv=10).mean()
        accuracy.append(a)

    plt.figure()
    plt.plot(range(len(accuracy)), accuracy)
    plt.show()


nb_samples = 1000
X, Y = make_classification(n_samples=nb_samples,
                           n_features=50,
                           n_informative=30,
                           n_redundant=20,
                           n_classes=2,
                           n_clusters_per_class=5)
rf = RandomForestClassifier(n_estimators=20)
rf.fit(X, Y)