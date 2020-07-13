import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import copy


class Diagnostics:

    @staticmethod
    def accuracy(t, p):
        out = (pd
               .DataFrame({'t': t,
                           'p': p})
               .assign(c=lambda x: np.where(x.t == x.p, 1, 0))
               .sum()
               )

        return out.c / len(t)

    @staticmethod
    def precision(t, p):
        out = (pd
               .DataFrame({'t': t,
                           'p': p})
               .assign(tp=lambda x: np.where((x.t == 1) & (x.p == 1), 1, 0),
                       fp=lambda x: np.where((x.t == 0) & (x.p == 1), 1, 0))
               .sum()
               )

        return out.tp / (out.tp + out.fp)

    @staticmethod
    def recall(t, p):
        out = (pd
               .DataFrame({'t': t,
                           'p': p})
               .assign(tp=lambda x: np.where((x.t == 1) & (x.p == 1), 1, 0),
                       fn=lambda x: np.where((x.t == 1) & (x.p == 0), 1, 0))
               .sum()
               )

        return out.tp / (out.tp + out.fn)

    @staticmethod
    def mse(t, p):

        r = (
            pd
                .DataFrame({'t': t,
                            'p': p})
                .assign(r=lambda x: (x.t - x.p) ** 2)
                .sum()
        )

        return r.r / len(t)

    @staticmethod
    def rmse(t, p):
        return Diagnostics.mse(t, p) ** .5

    @staticmethod
    def confusion(t, p):

        df = pd.DataFrame({'t': t,
                           'p': p})

        return pd.crosstab(df.t, df.p, rownames=['Actual'], colnames=['Predicted'])

    @staticmethod
    def practice_data():
        iris = load_iris()

        colnames = copy.deepcopy(iris['feature_names'])
        colnames.append('target')

        iris = (
            pd.concat([pd.DataFrame(iris['data']), pd.Series(iris['target'])], axis=1)
                .set_axis(colnames, axis=1, inplace=False)
                .query('target.isin([0, 1])')
        )

        return iris


