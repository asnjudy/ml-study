import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt

dataset = []
f = open('./fraud_data_3.csv', 'r')
try:
    reader = csv.reader(f, delimiter=',')
    next(reader, None)
    for row in reader:
        dataset.append(row)
finally:
    f.close()

target = np.array([x[0] for x in dataset])
data = np.array([x[1:] for x in dataset])

categorical_mask = [False, True, True, True, False, True]
enc = LabelEncoder()

for i in range(0, data.shape[1]):
    if categorical_mask[i]:
        label_encoder = enc.fit(data[:, i])
        print('Categorical classes:', label_encoder.classes_)
        integer_classes = label_encoder.transform(label_encoder.classes_)
        print('Integer classes:', integer_classes)
        t = label_encoder.transform(data[:, i])
        data[:, i] = t

mask = np.ones(data.shape, dtype=bool)
for i in range(0, data.shape[1]):
    if categorical_mask[i]:
        mask[:, i] = False

data_non_categoricals = data[:, np.all(mask, axis=0)]
data_categoricals = data[:, ~np.all(mask, axis=0)]

hotenc = OneHotEncoder()
hot_encoder = hotenc.fit(data_categoricals)
encoded_hot = hot_encoder.transform(data_categoricals)

new_data = data_non_categoricals
new_data = new_data.astype(np.float)

X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size=0.4, random_state=0)

logreg = linear_model.LogisticRegression(tol=1e-10)
logreg.fit(X_train[:, 0], y_train[:, 0])
log_output = logreg.predict_log_proba(X_test[:, 0])
print('Odds: ' + str(np.exp(logreg.coef_)))
print('Odds intercept : ' + str(np.exp(logreg.intercept_)))
print('Likelihood Intercept: ' + str(np.exp(logreg.intercept_) / (1 + np.exp(logreg.intercept_))))







