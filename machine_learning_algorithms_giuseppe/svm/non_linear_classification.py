from sklearn.datasets import make_circles

nb_samples = 500
X, Y = make_circles(n_samples=nb_samples, noise=0.1)


def logistic_regression():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    lr = LogisticRegression()
    print(cross_val_score(lr, X, Y, scoring='accuracy', cv=10).mean())


def svm_grid_search():
    import multiprocessing
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    param_grid = [
        {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'C': [0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 3.0]
        }
    ]

    gs = GridSearchCV(estimator=SVC(),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=multiprocessing.cpu_count())
    gs.fit(X, Y)
    print(gs.best_estimator_)
    print(gs.best_score_)


if __name__ == '__main__':
    svm_grid_search()
