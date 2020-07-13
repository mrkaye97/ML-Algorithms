import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from diagnostics import Diagnostics
from decision_tree_classifier import Tree


class RandomForest:

    def __init__(self, x: pd.DataFrame, y: pd.Series, n_trees = 500, max_depth = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.x = x
        self.y = y
        self.df = pd.concat([x, y], axis=1)
        self.trees = None

    def fit(self):
        trees = []
        for tree in range(self.n_trees):
            df = self.df.sample(frac=1)
            x = df.iloc[:, 0:self.x.shape[1]].reset_index(drop=True)
            y = df.target.reset_index(drop=True)

            tree = Tree(self.max_depth, x, y, forest=True)
            mod = tree.fit()
            trees.append(mod)

        self.trees = trees

        return self

    def predict(self, x, threshold=.5):
        preds = pd.DataFrame()
        for tree in self.trees:
            p = pd.Series(tree.predict(x, self.y))
            preds = pd.concat([preds, p], axis=1)

        preds = preds.mean(axis=1)

        return np.where(preds > threshold, 1, 0)


if __name__ == '__main__':
    df = Diagnostics.practice_data()
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:2], df.iloc[:, 4], random_state=646)

    mod = RandomForest(x_train, y_train, n_trees=250)
    x = mod.fit()
    preds = x.predict(x_test)

    print(preds)
    print(Diagnostics.accuracy(y_test, preds))
    print(Diagnostics.recall(y_test, preds))
    print(Diagnostics.precision(y_test, preds))
    print(Diagnostics.confusion(y_test, preds))
