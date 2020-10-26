from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import copy
import numpy as np
from src.diagnostics import Diagnostics


class Perceptron:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def fit(x, y_true, epochs, alpha):
        x = pd.concat([pd.Series([1]*len(x), index = x.index, name = 'bias'), x], axis = 1)
        w = np.zeros(x.shape[1])
        epoch = 1

        while epoch < epochs:
            for i in x.index:
                pred = Perceptron.sigmoid(sum([w[j] * x.loc[i][j] for j in range(len(w))]))

                for j in range(len(w)):
                    w[j] = w[j] + alpha * (y_true[i] - pred)*x.loc[i][j]


            if epoch % 100 == 0:
                print(w)

            epoch += 1

        return w

    @staticmethod
    def predict(test_x: pd.DataFrame, weights, threshold):
        test_x = pd.concat([pd.Series([1]*len(test_x), name = 'bias', index = test_x.index), test_x], axis = 1)
        preds = pd.Series([0]*len(test_x), index=test_x.index)

        for index, row in test_x.iterrows():
            pred_prob = Perceptron.sigmoid(sum([weights[col] * row[col] for col in range(len(row))]))
            preds[index] = 1 if pred_prob > threshold else 0

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

y_train.replace(-1, 0, inplace = True)
y_test.replace(-1, 0, inplace = True)

weights = Perceptron.fit(x_train, y_train, 100, .01)

preds = Perceptron.predict(x_test, weights, .5)

print(preds)
print(Diagnostics.accuracy(preds.reset_index(drop = True), y_test.reset_index(drop = True)))
print(Diagnostics.recall(preds.reset_index(drop = True), y_test.reset_index(drop = True)))
print(Diagnostics.precision(preds.reset_index(drop = True), y_test.reset_index(drop = True)))


