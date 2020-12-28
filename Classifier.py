import random
from collections import Counter

import numpy as np
from missingpy import MissForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from Utils import argmax, calc_uncertainty, argmin


class Classifier:
    def __init__(self, type, categorical=None, imputer_type="mf"):
        if type == "decision_tree" or type == "dt":
            self._clf = DecisionTreeClassifier()
        elif type == "neural_network" or type == "nn":
            self._clf = MLPClassifier()
        else:
            raise ValueError("invalid classifier type")

        if imputer_type == "miss_forest" or imputer_type == "mf":
            self.imputer = MissForest()
        else:
            raise ValueError("invalid imputer type")

        self._random_forests = None
        self._categorical = categorical

    def train(self, X, Y, incomplete=True):
        _X = np.copy(X)
        if incomplete:
            _X = self._impute_missing(_X)
        return self._clf.fit(_X, Y)

    def next_features(self, X, method="leu"):
        """
        :param method: the method to be used to calculate the index of the next feature to learn.
        :param X: (num_examples, num_features) with missing features as np.nan.
        :return: a list of the next feature to learn for each example.
        """
        # while calculating for one unknown feature I use the imputed value for the other unknown features.
        next_features = []
        if method is "random":
            for x in X:
                p = [a[0] for a in np.argwhere(~np.isnan(x))]
                try:
                    next_features.append(random.choice(p))
                except:
                    print("already know all features")
                    next_features.append(0)
        elif method is "leu":
            _X = np.copy(X)
            _X = self._impute_missing(_X)
            uncert_matrix = np.ones(_X.shape)
            for j in range(_X.shape[1]):
                out = self._random_forests[j].apply(np.concatenate([_X[:, :j], _X[:, j + 1:]], axis=1))
                for i in range(X.shape[0]):
                    if np.isnan(X[i, j]):
                        # TODO adjust for real valued features
                        f_counts = Counter([argmax(est.tree_.value[k][0])
                                            for k, est in zip(out[i], self._random_forests[j].estimators_)])
                        num = 0
                        den = 0
                        for c, v in f_counts.items():
                            x = _X[i, :]
                            x[j] = c
                            cls_probs = self.predict(x.reshape((1, len(x))), incomplete=False)[0]
                            num += v * calc_uncertainty(cls_probs)
                            den += v
                        uncert_matrix[i, j] = num / den
            next_features = [argmin(x) for x in uncert_matrix]
        else:
            raise ValueError("Incorrect method name")
        return next_features

    def predict(self, X, incomplete=True):
        _X = np.copy(X)
        if incomplete:
            _X = self._impute_missing(_X)
        return self._clf.predict_proba(X)

    def test(self, X, Y, incomplete=True):
        _X = np.copy(X)
        if incomplete:
            _X = self._impute_missing(_X)
        return self._clf.score(_X, Y)

    def _impute_missing(self, _X):
        """
        imputes the missing values in the input matrix
        :param _X: (num_examples, num_features) with missing features as np.nan.
        :return: nothing
        """
        X_imputed = self.imputer.fit_transform(_X, cat_vars=self._categorical)
        if self._random_forests is None:
            self._train_rfs(X_imputed)
        for j in range(_X.shape[1]):
            out = np.zeros(_X.shape)
            out[:, j] = self._random_forests[j].predict(
                np.concatenate([X_imputed[:, :j], X_imputed[:, j + 1:]], axis=1))
            for i in range(_X.shape[0]):
                if np.isnan(_X[i, j]):
                    _X[i, j] = 0
                else:
                    out[i, j] = 0
            _X = _X + out
        return _X

    def _train_rfs(self, _X):
        """
        trains the list of rf for each feature given the other features.
        :param _X: (num_examples, num_features) without any missing features.
        :return: nothing
        """
        rfs = []
        for j in range(_X.shape[1]):
            clf = RandomForestClassifier()
            clf.fit(np.concatenate([_X[:, :j], _X[:, j + 1:]], axis=1), _X[:, j])
            rfs.append(clf)
        self._random_forests = rfs
