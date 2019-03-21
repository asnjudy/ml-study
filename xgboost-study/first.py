import xgboost as xgb
from sklearn.metrics import accuracy_score


def xgboost_native_api():
    dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
    dtest = xgb.DMatrix('demo/data/agaricus.txt.test')

    """
    specify parameters via map
    max_depth: 弱分类器树的深度
    eta(learning_rate)
    num_round: 弱分类器树的数量
    
    max_depth = 2, num_round = 2
        train accuracy: 0.977737
        test accuracy: 0.978274
    
    max_depth = 2, num_round = 3
        train accuracy: 0.992937
        test accuracy: 0.993793
    
    max_depth = 3, num_round = 2
        train accuracy: 0.998772
        test accuracy: 1.000000
    
    max_depth = 3, num_round = 3
        train accuracy: 1.000000
        test accuracy: 1.000000
    """
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 3

    # train
    bst = xgb.train(param, dtrain, num_round)
    # predict
    preds_train = bst.predict(dtrain)
    preds_test = bst.predict(dtest)

    preds_train = [round(value) for value in preds_train]
    preds_test = [round(value) for value in preds_test]

    y_train = dtrain.get_label()
    y_test = dtest.get_label()
    print('train accuracy: %.6f' % accuracy_score(y_train, preds_train))
    print('test accuracy: %.6f' % accuracy_score(y_test, preds_test))

    xgb.plot_tree(bst, num_trees=2, rankdir='LR')


def xgboost_sklearn_api():
    from xgboost.sklearn import XGBClassifier
    from sklearn.datasets import load_svmlight_file

    # 读取数据
    X_train, y_train = load_svmlight_file('demo/data/agaricus.txt.train')
    X_test, y_test = load_svmlight_file('demo/data/agaricus.txt.test')

    bst = XGBClassifier(max_depth=2,
                        learning_rate=1,
                        n_estimators=2,
                        silent=True,
                        objective='binary:logistic')

    bst.fit(X_train, y_train)
    # predict
    preds_train = bst.predict(X_train)
    preds_test = bst.predict(X_test)
    # compute the accuracy
    print('train accuracy: %.6f' % accuracy_score(y_train, preds_train))
    print('test accuracy: %.6f' % accuracy_score(y_test, preds_test))

    pass


def xgboost_sklearn_cross_validation():
    """
    使用 K折交叉验证，评估模型的准确率
    """
    import numpy as np
    from xgboost.sklearn import XGBClassifier
    from sklearn.datasets import load_svmlight_file
    from scipy.sparse import vstack
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # 读取数据
    X_train, y_train = load_svmlight_file('demo/data/agaricus.txt.train')
    X_test, y_test = load_svmlight_file('demo/data/agaricus.txt.test')

    # 拼接
    # scipy.sparse.csr.csr_matrix, numpy.ndarray
    X = vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    bst = XGBClassifier(max_depth=2,
                        learning_rate=1,
                        n_estimators=2,
                        silent=True,
                        objective='binary:logistic')

    # 交叉验证
    kfold = StratifiedKFold(n_splits=10, random_state=7)
    scores = cross_val_score(bst, X, y, cv=kfold)
    print('cross validation accuracy:')
    print(scores)
    """
    scores.mean()
     0.9778435568593962
    """


###
def xgboost_early_stop():
    from xgboost.sklearn import XGBClassifier
    from sklearn.datasets import load_svmlight_file
    from matplotlib import pyplot

    # 读取数据
    X_train, y_train = load_svmlight_file('demo/data/agaricus.txt.train')
    X_test, y_test = load_svmlight_file('demo/data/agaricus.txt.test')

    bst = XGBClassifier(max_depth=2,
                        learning_rate=0.1,
                        n_estimators=100,
                        silent=True,
                        objective='binary:logistic')

    bst.fit(X_train, y_train,
            eval_metric=['error'],
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=10,
            verbose=True)

    # retrieve performance metrics
    results = bst.evals_result()
    # print(results)

    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Test')
    ax.legend()
    pyplot.ylabel('Error')
    pyplot.xlabel('Round')
    pyplot.title('XGBoost Early Stop')
    pyplot.show()

    # predict
    # make prediction
    preds_test = bst.predict(X_test)
    test_accuracy = accuracy_score(y_test, preds_test)
    print("Test Accuracy: %.4f%%" % (test_accuracy * 100.0))


if __name__ == '__main__':
    # xgboost_native_api()
    # xgboost_sklearn_api()

    pass
