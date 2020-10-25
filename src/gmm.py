import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from src.diagnostics import Diagnostics
from matplotlib import pyplot as plt
from scipy.spatial.distance import mahalanobis
import copy


class GMM:

    def __init__(self, x: pd.DataFrame, y: pd.Series, n_clusters=2):
        self.nclust = n_clusters
        self.x = x
        self.y = y
        self.means = None
        self.sigmas = None

    def get_x(self):
        return self.x

    def get_nclust(self):
        return self.nclust

    def initialize(self, x = pd.DataFrame, k = int):
        self.means = x.sample(k).reset_index(drop=True)
        self.sigmas = [x.cov()]*k


    def expectation(self, x: pd.DataFrame, assignments: pd.Series):

        for index, row in x.iterrows():
            min_mahala = np.inf
            for cluster in range(self.nclust):
                try:
                    dist = mahalanobis(row, self.means.iloc[cluster], self.sigmas[cluster])
                except:
                    pass
                if dist < min_mahala:
                    min_mahala = dist
                    assignments[index] = cluster

        return assignments

    def maximization(self, x: pd.DataFrame, assignments: pd.Series):

        df = pd.concat([x, assignments.rename('cl')], axis = 1)

        self.means = df.groupby('cl').mean()

        grouped = df.groupby('cl')
        self.sigs = [gr.drop('cl', axis=1).cov() for name, gr in grouped]

    def fit(self):

        x = self.x
        nclust = self.nclust

        self.initialize(x, nclust)

        oldassignments = pd.Series([0]*len(x), index=x.index)
        nothing_changed = False
        while not nothing_changed:
            assignments = self.expectation(x, assignments=copy.deepcopy(oldassignments))
            self.maximization(x, assignments)

            if all(assignments == oldassignments):
                nothing_changed = True
                print('great success!')
            else:
                oldassignments = assignments
                print(sum(assignments))


    def predict(self, x: pd.DataFrame):
        return self.expectation(x, pd.Series([0]*len(x), index=x.index))

    @staticmethod
    def run():
        df = Diagnostics.practice_data()
        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:4], df.iloc[:, 4], random_state=646)

        mod = GMM(x_train, y_train, n_clusters=2)
        mod.fit()
        preds = mod.predict(x_train)
        plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1])
        plt.show()
        plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c = preds)
        plt.show()


if __name__ == '__main__':
    GMM.run()
