from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

nb_samples = 500
X, Y = make_classification(n_samples=nb_samples,
                           n_features=2,
                           n_redundant=0,
                           n_classes=2)

lr = LogisticRegression()
svc = SVC(kernel='poly', probability=True)
dt = DecisionTreeClassifier()

classifiers = [('lr', lr), ('dt', dt), ('svc', svc)]
vc = VotingClassifier(estimators=classifiers, voting='hard')

accuracy_list = list()
accuracy_list.append(cross_val_score(lr, X, Y, scoring='accuracy', cv=10).mean())
accuracy_list.append(cross_val_score(dt, X, Y, scoring='accuracy', cv=10).mean())
accuracy_list.append(cross_val_score(svc, X, Y, scoring='accuracy', cv=10).mean())
accuracy_list.append(cross_val_score(vc, X, Y, scoring='accuracy', cv=10).mean())

print(accuracy_list)
