import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from python.diagnostics import Diagnostics
import random


class Node:

    def __init__(self, features: pd.DataFrame, target: pd.Series, parent = None, depth=0, feature_for_split = None, cl = None, th = None):
        self.features = features
        self.target = target
        self.left = None
        self.right = None
        self.depth = depth
        self.parent = parent
        self.feature_for_split = feature_for_split
        self.feature_threshold = th
        self.classification = cl

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_feautres(self):
        return self.features

    def get_target(self):
        return self.target

    def get_depth(self):
        return self.depth

    def get_parent(self):
        return self.parent

    def get_feature_for_split(self):
        return self.feature_for_split



class Tree:

    def __init__(self, max_depth: int, x, y, forest=False):
        self.root = Node(x, y)
        self.max_depth = max_depth
        self.isForest = forest

    def entropy(self, target: pd.Series, feature=None):
        if feature is not None:
            return self._entropy_x_y(target, feature)
        else:
            return sum([-1 * (c / len(target)) * np.log2(c / len(target)) for c in target.value_counts()])

    def _entropy_x_y(self, target: pd.Series, feature: pd.Series):
        df = (
            pd.DataFrame({'f': feature,
                          't': target})
                .groupby(['t', 'f'])
                .size()
                .reset_index(drop=False, name='size')
        )

        levels = df.t.unique()
        tot = df.sum()['size']
        entropy = 0

        for lvl in levels:
            sub = (df
                   .query("t==@lvl")
                   .sum()
                   )['size']

            pc = sub / tot
            ec = self.entropy(df.query("t==@lvl")['size'])
            entropy += pc * ec

        return entropy

    def infogain(self, target: pd.Series, feature: pd.Series):
        return self.entropy(target) - self.entropy(target, feature)

    def col_separate(self, f, y):
        min_entropy = np.inf
        cutoff = np.inf

        for val in f:
            pred = f < val
            ylhs = y[pred]
            yrhs = y[~pred]

            tot_ent = self.entropy(ylhs) + self.entropy(yrhs)
            if tot_ent <= min_entropy:
                min_entropy = tot_ent
                cutoff = val

        return cutoff

    def split(self, x, y_true, col):
        cutoff = self.col_separate(x.iloc[:, col], y_true)

        l = x.iloc[:, col] < cutoff
        r = x.iloc[:, col] >= cutoff
        ly = y_true[l]
        ry = y_true[r]
        lx = x[l].drop(x.columns[col], axis = 1)
        rx = x[r].drop(x.columns[col], axis = 1)

        return lx, ly, rx, ry, cutoff

    def fit(self, curr=None):
        max_info_gain = -9999999
        best_split = None

        if curr is None:
            curr = self.root

        if curr.depth == self.max_depth:
            curr.classification = curr.target.value_counts().idxmax()
        elif curr.features.empty:
            curr.classification = curr.target.value_counts().idxmax()
        elif len(curr.target.unique()) == 1 and curr.target is not None:
            curr.classification = curr.target.unique()[0]
        else:
            if self.isForest and curr.features.shape[1] != 0:
                num_features_to_use = random.choice(list(range(1, curr.features.shape[1] + 1)))
                features_to_use = random.sample(list(range(curr.features.shape[1])),
                                                num_features_to_use)
                curr.features = curr.features.iloc[:, 0:curr.features.shape[1]][curr.features.columns[features_to_use]]

            for j in range(curr.features.shape[1]):
                ig = self.infogain(curr.target, curr.features.iloc[:, j])
                if ig >= max_info_gain:
                    best_split = j
                    max_info_gain = ig

            lx, ly, rx, ry, cutoff = self.split(curr.features, curr.target, best_split)

            if len(ly) == 0:
                curr.left = None
            else:
                curr.left = Node(lx, ly, curr, curr.depth + 1)
            if len(ry) == 0:
                curr.right = None
            else:
                curr.right = Node(rx, ry, curr, curr.depth + 1)

            curr.feature_for_split = best_split
            curr.feature_threshold = cutoff

            self.fit(curr.left)
            self.fit(curr.right)

        return self

    def predict(self, x: pd.DataFrame, train_target: pd.Series):

        preds = []
        for ix, row in x.iterrows():
            curr = self.root
            while curr.classification is None:
                if row[curr.feature_for_split] <= curr.feature_threshold:
                    curr = curr.left
                else:
                    curr = curr.right


            preds.append(curr.classification)

        return preds


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
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:2], df.iloc[:, 4], random_state=646)

    mod = Tree(4, x_train, y_train)
    x = mod.fit()

    preds = mod.predict(x_test, y_train)

    print(Diagnostics.accuracy(y_test, preds))
    print(Diagnostics.recall(y_test, preds))
    print(Diagnostics.precision(y_test, preds))
    print(Diagnostics.confusion(y_test, preds))
