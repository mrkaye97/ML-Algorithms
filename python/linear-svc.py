from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import copy
import numpy as np
from python.diagnostics import Diagnostics


class LinearSVC:

    @staticmethod
    def fit_svc(x, y_true, epochs, alpha):
        w = np.zeros(x.shape[1])
        epoch = 1

        while epoch < epochs:
            pred = sum([w[i] * x.iloc[:, i] for i in range(x.shape[1])]) * y_true

            for i in range(pred.shape[0]):
                for j in range(w.shape[0]):
                    if pred[i] >= 1:
                        w[j] = w[j] - alpha * (2 * 1 / epoch * w[j])
                    else:
                        w[j] = w[j] + alpha * (x.iloc[i, j] * y_true[i] - 2 * 1 / epoch * w[j])

            if epoch % 500 == 0:
                print(w)

            epoch += 1

        return w

    @staticmethod
    def predict(test_x, weights):

        preds = np.where((sum([test_x.iloc[:, j] * weights[j] for j in range(weights.shape[0])])) >= 1, 1, -1)

        return preds


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


df = practice_data()

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:4], df.iloc[:, 4])

weights = LinearSVC.fit_svc(x_train, y_train, 1000, .01)

preds = LinearSVC.predict(y_test, weights)

print(preds)
print(Diagnostics.accuracy(preds, y_test.reset_index(drop = True)))
print(Diagnostics.recall(preds, y_test.reset_index(drop = True)))
print(Diagnostics.precision(preds, y_test.reset_index(drop = True)))

