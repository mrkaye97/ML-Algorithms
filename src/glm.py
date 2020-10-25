import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from src.diagnostics import Diagnostics


class GLM(object):

    def __init__(self):
        self.betas = None

    @staticmethod
    def glm(c):
        if c == 'logit':
            return Logistic()
        if c == 'gamma':
            return Gamma()
        if c == 'poisson':
            return Poisson()

    def fit(self):
        pass

    def predict(self, x):
        pass


class Logistic(GLM):

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def predictfromprob(probs, threshold=.5):
        return [1 if prob >= threshold else 0 for prob in probs]

    def predict(self, x, threshold=.5):
        link = self.betas[0] + sum([self.betas[col + 1] * x.iloc[:, col] for col in range(len(self.betas) - 1)])
        probs = self.sigmoid(link)
        return self.predictfromprob(probs, threshold)

    def fit(self, x, y_true, epochs, alpha):
        epoch = 1
        self.betas = np.zeros(shape=x.shape[1] + 1)

        n = len(x)

        while epoch < epochs:
            lp = self.betas[0] + sum([self.betas[col + 1] * x.iloc[:, col] for col in range(len(self.betas) - 1)])
            probs = self.sigmoid(lp)
            pred = self.predictfromprob(probs, .5)

            for j in range(self.betas.shape[0]):
                if j == 0:
                    update = sum([(pred[i] - y_true[i]) for i in range(n)])
                else:
                    update = sum([x.iloc[i, j - 1] * (pred[i] - y_true[i]) for i in range(n)])

                self.betas[j] = self.betas[j] - alpha * update

            epoch += 1

            if epoch % 500 == 0:
                print(self.betas)

        return self.betas


class Poisson(GLM):

    def predict(self, x, threshold=.5):
        return None

    def fit(self, x, y_true, epochs, alpha):
        return None


class Gamma(GLM):
    def predict(self, x, threshold=.5):
        return None

    def fit(self, x, y_true, epochs, alpha):
        return None


def practice_data():
    iris = datasets.load_iris()

    colnames = copy.deepcopy(iris['feature_names'])
    colnames.append('target')

    iris = (
        pd.concat([pd.DataFrame(iris['data']), pd.Series(iris['target'])], axis=1)
            .set_axis(colnames, axis=1, inplace=False)
            .query('target.isin([0, 1])')
    )

    return iris


if __name__ == '__main__':
    df = practice_data()
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:4], df.iloc[:, 4], random_state=646)

    mod = GLM().glm('logit')
    weights = mod.fit(x_train.reset_index(drop=True),
                      y_train.reset_index(drop=True),
                      1000,
                      .01)

    preds = mod.predict(x_test.reset_index(drop=True))

    print(preds)
    print(Diagnostics.accuracy(y_test, preds))
    print(Diagnostics.recall(y_test, preds))
    print(Diagnostics.precision(y_test, preds))
