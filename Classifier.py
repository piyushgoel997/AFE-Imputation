import random
from collections import Counter

import numpy as np
from missingpy import MissForest
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from Utils import argmax, calc_uncertainty, argmin


class Classifier:
    def __init__(self, type, categorical, imputer_type="mf", uncertainty_measure="confidence", set_alpha=False):
        self._random_forests = []
        self._categorical = categorical
        self._num_forests = 20
        self._internal_loop = 50

        for c in self._categorical:
            if c:
                self._random_forests.append([RandomForestClassifier() for _ in range(self._num_forests)])
            else:
                self._random_forests.append([RandomForestRegressor() for _ in range(self._num_forests)])

        self._uncertainty_measure = uncertainty_measure
        self._alpha = 0.5
        self._set_alpha = set_alpha
        if type == "decision_tree" or type == "dt":
            self._clf = DecisionTreeClassifier()
        elif type == "neural_network" or type == "nn":
            self._clf = MLPClassifier()
        else:
            print("The type is", type)
            raise ValueError("invalid classifier type")

        if imputer_type == "miss_forest" or imputer_type == "mf":
            self.imputer = MissForest()
        else:
            raise ValueError("invalid imputer type")

    def train(self, X, Y, incomplete=True):
        _X = np.copy(X)
        self.imputer.fit(_X, cat_vars=[i for i, c in enumerate(self._categorical) if c])
        if incomplete:
            _X = self.imputer.transform(_X)
        self._train_rfs(_X, X)
        if self._set_alpha:
            self._alpha = np.sum(Y) / len(Y)
        return self._clf.fit(_X, Y)

    def next_features(self, X, method):
        """
        :param method: the method to be used to calculate the index of the next feature to learn.
        :param X: (num_examples, num_features) with missing features as np.nan.
        :return: a list of the next feature to learn for each example.
        """
        # while calculating for one unknown feature I use the imputed value for the other unknown features.
        next_features = []
        if method == "random":
            for _x in X:
                p = [a[0] for a in np.argwhere(~np.isnan(_x))]
                try:
                    next_features.append(random.choice(p))
                except:
                    print("already know all features")
                    next_features.append(0)
        elif method == "leu":
            expected_uncert_matrix = self._num_forests * np.ones(X.shape)
            for k in range(self._num_forests):
                _X = np.copy(X)
                _X = self._impute_missing(_X, k=k)
                for j in range(_X.shape[1]):
                    out = self._random_forests[j][k].apply(np.concatenate([_X[:, :j], _X[:, j + 1:]], axis=1))
                    for i in range(X.shape[0]):
                        if np.isnan(X[i, j]):
                            f = [argmax(est.tree_.value[l][0]) for l, est in
                                 zip(out[i], self._random_forests[j][k].estimators_)]
                            un = 1
                            if self._categorical[j]:
                                num = 0
                                den = 0
                                for c, v in Counter(f).items():
                                    _x = _X[i, :]
                                    _x[j] = c
                                    x = X[i, :]
                                    x[j] = c
                                    expected_p = self._expected_prob(_x, x)
                                    # cls_probs = self.predict(_x.reshape((1, len(_x))), incomplete=False)[0]
                                    num += v * calc_uncertainty(expected_p, method=self._uncertainty_measure,
                                                                alpha=self._alpha)
                                    den += v
                                un = num / den
                            else:
                                # real valued feature, distr approximated as a gaussian.
                                mean = np.mean(f)
                                std_dev = np.std(f)
                                step = 8 * std_dev / 100
                                distr = norm(f)
                                for ii in range(mean - 4 * std_dev, mean - 4 * std_dev, step):
                                    _x = _X[i, :]
                                    _x[j] = ii
                                    x = X[i, :]
                                    x[j] = ii
                                    expected_p = self._expected_prob(_x, x)
                                    un += distr.pdf(ii) * calc_uncertainty(expected_p, method=self._uncertainty_measure,
                                                                           alpha=self._alpha)
                            expected_uncert_matrix[i, j] = un
            expected_uncert_matrix = expected_uncert_matrix / self._num_forests
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
        imputes the missing values in the input matrix
        :param _X: (num_examples, num_features) with missing features as np.nan.
        :return: nothing
        """
        X_imputed = self.imputer.transform(_X)
        for j in range(_X.shape[1]):
            out = np.zeros(_X.shape)
            out[:, j] = self._random_forests[j][k].predict(
                np.concatenate([X_imputed[:, :j], X_imputed[:, j + 1:]], axis=1))
            for i in range(_X.shape[0]):
                if np.isnan(_X[i, j]):
                    _X[i, j] = 0
                else:
                    out[i, j] = 0
            _X = _X + out
        return _X

    def _train_rfs(self, _X_imputed, _X):
        """
        trains the list of rf for each feature given the other features.
        :param _X_imputed: (num_examples, num_features) without any missing features.
        :return: nothing
        """
        for j in range(_X_imputed.shape[1]):
            _X_without_nan = []
            for i in range(_X_imputed.shape[0]):
                if not np.isnan(_X[i, j]):
                    _X_without_nan.append(_X_imputed[i, :])
            _X_without_nan = np.array(_X_without_nan)
            for k in range(self._num_forests):
                self._random_forests[j][k].fit(
                    np.concatenate([_X_without_nan[:, :j], _X_without_nan[:, j + 1:]], axis=1), _X_without_nan[:, j])

    def _expected_prob(self, _x, x):
        # impute randomly for the other features
        _X = np.zeros((self._internal_loop, len(x)))
        for j in range(_X.shape[1]):
            if not np.isnan(x[j]):
                continue
            _x_minus_j = np.concatenate([_x[:j], _x[j + 1:]])
            for i in range(_X.shape[0]):
                _X[i, j] = self._random_forests[j][0].estimators_[
                    random.randint(0, self._random_forests[j][0].n_estimators - 1)].predict(
                    _x_minus_j.reshape((1, len(_x_minus_j))))
        pred = self.predict(_X, incomplete=False)
        return np.average(pred, axis=0)
