# coding=utf-8
import numpy as np
from sklearn.datasets import make_classification

nb_samples = 5000
X, Y = make_classification(n_samples=nb_samples,
                           n_features=30,
                           n_informative=30,
                           n_redundant=0,
                           n_classes=3,
                           n_clusters_per_class=1)


def classify_gini_impurity():
    """
    默认使用的是基尼不纯的指标
    :return:
    """
    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier()
    dt.fit(X, Y)

    print(dt.feature_importances_)
    print(np.argsort(dt.feature_importances_))


def export_tree_graph():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import export_graphviz

    dt = DecisionTreeClassifier()
    dt.fit(X, Y)
    with open('dt.dot', 'w') as df:
        df = export_graphviz(dt, out_file=df, feature_names=['A', 'B', 'C'], class_names=['C1', 'C2', 'C3'])


def classify_decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score

    print(cross_val_score(DecisionTreeClassifier(),
                          X, Y, scoring='accuracy', cv=10).mean())
    print(cross_val_score(DecisionTreeClassifier(max_features='auto'),
                          X, Y, scoring='accuracy', cv=10).mean())
    print(cross_val_score(DecisionTreeClassifier(min_samples_split=100),
                          X, Y, scoring='accuracy', cv=10).mean())

    # logistic regression
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    print(cross_val_score(lr, X, Y, scoring='accuracy', cv=10).mean())


if __name__ == '__main__':
    classify_decision_tree()
