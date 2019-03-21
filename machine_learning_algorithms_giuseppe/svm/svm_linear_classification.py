from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

nb_samples = 500
X, Y = make_classification(n_samples=nb_samples,
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_clusters_per_class=1)

svc = SVC(kernel='linear')
cross_val_score(svc, X, Y, scoring='accuracy', cv=10).mean()




