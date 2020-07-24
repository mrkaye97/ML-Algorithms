import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from python.diagnostics import Diagnostics
from matplotlib import pyplot as plt


class KMeans:

    def __init__(self, x: pd.DataFrame, y: pd.Series, n_clusters=2):
        self.nclust = n_clusters
        self.x = x
        self.y = y
        self.centroids = None

    def get_x(self):
        return self.x

    def get_nclust(self):
        return self.nclust

    def distance(self, point, center):
        return abs(np.linalg.norm(point - center))

    def assignment(self, x: pd.DataFrame, centers: pd.DataFrame):

        assignments = []
        for i, obs in x.iterrows():
            nearest = None
            mindist = np.Inf
            for j, center in centers.iterrows():
                d = self.distance(obs, center)
                if d < mindist:
                    nearest = j
                    mindist = d

            assignments.append(nearest)

        return pd.Series(assignments)

    def update(self, x: pd.DataFrame, assignments: pd.Series):

        return (
            pd.concat([x, assignments.rename('assignments')], axis=1)
                .groupby('assignments')
                .mean()
        )

    def fit(self):
        x = self.get_x().reset_index(drop=True)
        n = self.get_nclust()

        prev = None
        curr = pd.Series(np.random.choice(n, len(x)))

        while not (prev == curr).all():
            centroids = self.update(x, curr)
            prev = curr
            curr = self.assignment(x, centroids)

        self.centroids = centroids

    def predict(self, x, centroids):
        return self.assignment(x, centroids)

    @staticmethod
    def run():
        df = Diagnostics.practice_data()
        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:4], df.iloc[:, 4], random_state=646)

        mod = KMeans(x_train, y_train, n_clusters=2)
        mod.fit()
        preds = mod.predict(x_train, mod.centroids)

        plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c = preds)
        plt.show()


if __name__ == '__main__':
    KMeans.run()
