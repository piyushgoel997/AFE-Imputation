import random

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from MissForest import MissForest
from Utils import argmax, calc_uncertainty, argmin


class Classifier:
    def __init__(self, type, categorical, uncertainty_measure="confidence", set_alpha=False):
        self._categorical = categorical
        self._num_imputers = 10
        self._internal_loop = 10
        self._list_of_imputers = []

        self._uncertainty_measure = uncertainty_measure
        self._alpha = 0.5
        self._set_alpha = set_alpha
        if type == "decision_tree" or type == "dt":
            self._clf = DecisionTreeClassifier()
        elif type == "neural_network" or type == "nn":
            self._clf = MLPClassifier(hidden_layer_sizes=(100, 100,))
        else:
            print("The type is", type)
            raise ValueError("invalid classifier type")

    def train(self, X, Y, incomplete=True):
        _X = None
        for k in range(self._num_imputers):
            imp = MissForest()
            _X = imp.fit_transform(X)
            self._list_of_imputers.append(imp)
        return self._clf.fit(_X, Y)

    def next_features(self, X, method):
        """
        :param method: the method to be used to calculate the index of the next feature to learn.
        :param X: (num_examples, num_features) with missing features as np.nan.
        :return: a list of the next feature to learn for each example.
        """
        if method == "random":
            next_features = []
            for _x in X:
                p = [a[0] for a in np.argwhere(~np.isnan(_x))]
                try:
                    next_features.append(random.choice(p))
                except:
                    print("already know all features")
                    next_features.append(0)
        elif method == "leu":
            expected_uncert_matrix = self._num_imputers * np.ones(X.shape)
            for k in range(self._num_imputers):
                # only use the kth imputer now
                _X = np.copy(X)
                _X = self._impute_missing(_X, k=k)
                for j in range(_X.shape[1]):
                    out = self._list_of_imputers[k].list_of_forests[j].apply(np.concatenate([_X[:, :j], _X[:, j + 1:]],
                                                                                            axis=1))
                    for i in range(X.shape[0]):
                        if np.isnan(X[i, j]):
                            f = [argmax(est.tree_.value[leaf][0])
                                 for leaf, est in zip(out[i], self._list_of_imputers[k].list_of_forests[j].estimators_)]
                            total_uncert = 0
                            saved = {}
                            for c in f:
                                if c not in saved:
                                    x = np.copy(X[i, :])
                                    x[j] = c
                                    expected_p = self._expected_prob(x, k)
                                    un = calc_uncertainty(expected_p, method=self._uncertainty_measure, alpha=self._alpha)
                                    saved[c] = un
                                else:
                                    un = saved[c]
                                total_uncert += un
                            total_uncert = total_uncert / len(f)
                            expected_uncert_matrix[i, j] = total_uncert
            expected_uncert_matrix = expected_uncert_matrix / self._num_imputers
            next_features = [argmin(x) for x in expected_uncert_matrix]
        else:
            raise ValueError("Incorrect method name")
        return next_features

    def predict(self, X, incomplete=True):
        _X = np.copy(X)
        if incomplete:
            _X = self._impute_missing(_X)
        return self._clf.predict_proba(X)

    def test_accuracy(self, X, Y, incomplete=True):
        _X = np.copy(X)
        if incomplete:
            _X = self._impute_missing(_X)
        return self._clf.score(_X, Y)

    def test_auc(self, X, Y, incomplete=True):
        _X = np.copy(X)
        if incomplete:
            _X = self._impute_missing(_X)
        return roc_auc_score(Y, self.predict(_X))

    def _impute_missing(self, _X, k=0):
        """
        DO NOT USE THIS FOR TRAINING DATA, ONLY WHILE TESTING!!
        imputes the missing values in the input matrix
        :param _X: (num_examples, num_features) with missing features as np.nan.
        :return: nothing
        """
        return self._list_of_imputers[k].transform(_X, test=True)

    def _expected_prob(self, x, k):
        _X = np.tile(x, (self._internal_loop, 1))
        _X = self._impute_missing(_X, k)
        pred = self.predict(_X, incomplete=False)
        return np.average(pred, axis=0)
