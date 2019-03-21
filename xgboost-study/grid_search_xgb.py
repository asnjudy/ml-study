# 运行 xgboost安装包中的示例程序
from xgboost import XGBClassifier
# 加载LibSVM格式数据模块
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

# 读取数据
X_train, y_train = load_svmlight_file('demo/data/agaricus.txt.train')
X_test, y_test = load_svmlight_file('demo/data/agaricus.txt.test')

bst = XGBClassifier(max_depth=2,
                    learning_rate=0.1,
                    silent=True,
                    objective='binary:logistic')

# bst.fit(X_train, y_train,
#         eval_metric=['error'],
#         eval_set=[(X_train, y_train), (X_test, y_test)],
#         early_stopping_rounds=10,
#         verbose=True)

param_grid = {'n_estimators': range(1, 51, 1)}

clf = GridSearchCV(estimator=bst, param_grid=param_grid, scoring='accuracy', cv=5)
clf.fit(X_train, y_train)
print(clf.best_params_, clf.best_score_)
