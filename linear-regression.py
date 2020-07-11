import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from diagnostics import Diagnostics


class LinearRegression:

    @staticmethod
    def linear_regression(x, y_true, epochs=500, alpha=.01):

        epoch = 1
        betas = np.zeros(shape=x.shape[1] + 1)
        n = len(x)

        while epoch < epochs:
            pred = betas[0] + sum([betas[col + 1] * x.iloc[:, col] for col in range(len(betas) - 1)])

            for j in range(betas.shape[0]):
                if j == 0:
                    update = (-2 / n) * sum([(y_true[i] - pred[i]) for i in range(n)])
                else:
                    update = (-2 / n) * sum([x.iloc[i, j - 1] * (y_true[i] - pred[i]) for i in range(n)])

                betas[j] = betas[j] - alpha * update

            epoch += 1

            if epoch % 500 == 0:
                print(betas)

        return betas

    @staticmethod
    def predict(betas, x):

        return betas[0] + sum([betas[col + 1] * x.iloc[:, col] for col in range(len(betas) - 1)])


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


if __name__ == '__main__':
    df = practice_data()
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:3], df.iloc[:, 3], random_state=646)

    weights = LinearRegression.linear_regression(x_train.reset_index(drop = True), y_train.reset_index(drop = True), 1000, .01)
    preds = LinearRegression.predict(weights, x_test.reset_index(drop=True))

    print(weights)
    print(Diagnostics.rmse(y_test, preds))