from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy.optimize import linprog
import numpy as np

def practice_data():
    iris = datasets.load_iris()

    colnames = copy.deepcopy(iris['feature_names'])
    colnames.append('target')

    iris = (
        pd.concat([pd.DataFrame(iris['data']), pd.Series(iris['target'])], axis=1)
            .set_axis(colnames, axis=1, inplace=False)
            .query('target.isin([0, 1])')
            .replace(0, -1)
    )

    return iris

def fit_svc(x, y_true):
    w = np.zeros(x.shape[1])
    alpha = .0001
    epochs = 1

    while epochs < 10000:
        pred = sum([w[i] * x.iloc[:, i] for i in range(x.shape[1])]) * y_true

        for i in range(pred.shape[0]):
            for j in range(w.shape[0]):
                if pred[i] >= 1:
                    w[j] = w[j] - alpha * (2 * 1/epochs * w[j])
                else:
                    w[j] = w[j] + alpha * (x.iloc[i,j] * y_true[i] - 2 * 1 / epochs * w[j])

        if epochs % 1000 == 0:
            print(w)

        epochs += 1

    return w

def predict(test_x, weights):

    preds = np.where((sum([test_x.iloc[:,j] * weights[j] for j in range(weights.shape[0])])) >= 1, 1, -1)

    return preds

def acc(pred_y, true_y):
    right = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            right += 1

    return right / len(pred_y)


df = practice_data()

train = df.sample(frac = .9)
test = df.sample(frac = .9)

weights = fit_svc(df.iloc[:, 0:4], df.iloc[:, 4])

preds = predict(test.iloc[:, 0:4], weights)

print(preds)
print(acc(preds, test.target.reset_index(drop = True)))