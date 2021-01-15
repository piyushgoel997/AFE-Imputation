import argparse
import pickle
import random
import time
import warnings
from collections import Counter
from multiprocessing import Pool

import numpy as np

from Classifier import Classifier

warnings.filterwarnings("ignore")


class Exp:

    def __init__(self, data, clf_type, uncertainty_measure, categorical_):
        self.data = data
        self.NUM_FOLDS = 10
        self.MAX_TEST_POINTS = 500
        self.CLASSIFIER_TYPE = clf_type
        self.NUM_SAMPLES = 3
        self.REMOVE_RATIO = 0.75
        self.uncertainty_measure = uncertainty_measure.split("*")[0]
        self.set_alpha = len(uncertainty_measure.split("*")) == 2
        self.categorical = categorical_

    def remove_data(self, X, r):
        _X = np.copy(X)
        for i in range(_X.shape[0]):
            for j in range(_X.shape[1]):
                if random.random() < r:
                    _X[i, j] = np.nan
        return _X

    def run_exp(self, X_test, X_test_complete, Y_test, clf):
        A = []
        AR = []
        U = []
        S = []
        for method in ["random", "leu"]:
            print(method, "started")
            _X = np.copy(X_test)

            a_exp = []
            ar_exp = []
            u_exp = []
            s_exp = []

            for _ in range(self.NUM_SAMPLES):
                tt = time.time()
                learn_idx = clf.next_features(_X, method=method)
                s_exp.append((time.time() - tt) / len(_X))
                for i_, j in enumerate(learn_idx):
                    _X[i_, j] = X_test_complete[i_, j]
                a_exp.append(clf.test_accuracy(_X, Y_test))
                ar_exp.append(clf.test_auc(_X, Y_test))
                u_exp.append(clf.average_uncertainty(_X))

            A.append(ar_exp)
            AR.append(ar_exp)
            U.append(u_exp)
            S.append(s_exp)
            print(method, "ended")
        # no sampling
        A.append([clf.test_accuracy(X_test, Y_test)] + [-1] * (self.NUM_SAMPLES - 1))
        AR.append([clf.test_auc(X_test, Y_test)] + [-1] * (self.NUM_SAMPLES - 1))
        U.append([clf.average_uncertainty(X_test)] + [-1] * (self.NUM_SAMPLES - 1))
        S.append([0] * (self.NUM_SAMPLES))
        return A, AR, U, S

    def split_indices(self, exp_no):
        size = self.data.shape[0]
        start = int((exp_no * size) / self.NUM_FOLDS)
        stop = int(((exp_no + 1) * size) / self.NUM_FOLDS)
        test = list(range(start, stop))
        test = test[:min(len(test), self.MAX_TEST_POINTS)]
        train = list(range(0, start)) + list(range(stop, size))
        return train, test

    def one_exp(self, exp_no):
        train_ind, test_ind = self.split_indices(exp_no)

        X_train_complete = self.data[train_ind, :-1]
        Y_train = self.data[train_ind, -1]

        X_test_complete = self.data[test_ind, :-1]
        Y_test = self.data[test_ind, -1]

        c_clf = Classifier(self.CLASSIFIER_TYPE, categorical=self.categorical,
                           uncertainty_measure=self.uncertainty_measure, set_alpha=self.set_alpha)
        c_clf.train(X_train_complete, Y_train, incomplete=False)
        complete_accuracy_exp = c_clf.test_accuracy(X_test_complete, Y_test, incomplete=False)
        complete_auc_exp = c_clf.test_auc(X_test_complete, Y_test, incomplete=False)

        t1 = time.time()
        print("Starting with complete data")
        X_test = self.remove_data(X_test_complete, self.REMOVE_RATIO)
        ac, arc, uc, sc = self.run_exp(X_test, X_test_complete, Y_test, c_clf)
        print("Complete data finished in", time.time() - t1)
        t1 = time.time()

        ############################################################################################

        X_train = self.remove_data(X_train_complete, self.REMOVE_RATIO)
        clf = Classifier(self.CLASSIFIER_TYPE, categorical=self.categorical)
        clf.train(X_train, Y_train)
        ai, ari, ui, si = self.run_exp(X_test, X_test_complete, Y_test, clf)
        print("Incomplete data finished in", time.time() - t1)
        print(ac, arc, uc, sc, ai, ari, ui, si)

        acc_exp = np.array(ac + ai)
        auc_exp = np.array(arc + ari)
        uncert_exp = np.array(uc + ui)
        sampling_times_exp = np.array(sc + si)

        return acc_exp, auc_exp, uncert_exp, sampling_times_exp, complete_accuracy_exp, complete_auc_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="car_0")
    parser.add_argument('--clf', default="nn")
    parser.add_argument('--um', default="confidence")
    parser.add_argument('--parallel', dest='parallel', action='store_true')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false')
    parser.set_defaults(parallel=True)
    args = parser.parse_args()

    start_time = time.time()

    # load data
    data = np.load("data/" + args.data + ".npy", allow_pickle=True).astype(np.float)
    cat = np.load("data/" + args.data + "_cat.npy")
    print("Data loaded in", time.time() - start_time)

    categorical = [False] * (data.shape[1] - 1)
    for c in cat:
        categorical[c] = True

    print(categorical)
    exp = Exp(data, args.clf, args.um, categorical)
    acc = np.zeros((6, exp.NUM_SAMPLES))
    auc = np.zeros((6, exp.NUM_SAMPLES))
    uncertainty = np.zeros((6, exp.NUM_SAMPLES))
    sampling_times = np.zeros((6, exp.NUM_SAMPLES))

    complete_accuracy = 0
    complete_auc = 0

    print("Starting experiments")

    if args.parallel:
        pool = Pool()
        res = pool.map_async(exp.one_exp, list(range(exp.NUM_FOLDS))).get()
    else:
        res = map(exp.one_exp, list(range(exp.NUM_FOLDS)))

    # axis 1 -> folds
    #     2 -> acc, auc roc, uncertainty, sampling times, complete accuracy, complete auc roc
    # acc, auc roc, uncertainty, sampling times -> Axis 1 -> first, second, third sample
    #      Axis 2 -> random, leu, no first for complete and then for incomplete.
    pickle.dump(res, open("saved_results/" + args.data + "_" + args.um[:-6] + "_" + args.clf, 'wb'))

    for r in res:
        a, ar, u, s, ca, car = r
        acc += a
        auc += ar
        uncertainty += u
        sampling_times += s
        complete_accuracy += ca
        complete_auc += car
        t = time.time()
    print("Experiments finished")

    acc /= exp.NUM_FOLDS
    auc /= exp.NUM_FOLDS
    uncertainty /= exp.NUM_FOLDS
    sampling_times /= exp.NUM_FOLDS
    complete_accuracy /= exp.NUM_FOLDS
    complete_auc /= exp.NUM_FOLDS
    for a, ar, u, s, r in zip(acc.T, auc.T, uncertainty.T, sampling_times.T, range(exp.NUM_SAMPLES)):
        print("==========================================================")
        print("Complete training data! Sampling number =", r)
        print()
        print("Averaged Accuracies ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(a[2], a[0], a[1]))
        print()
        print("Averaged AUCs ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(ar[2], ar[0], ar[1]))
        print()
        print("Averaged Uncertainties ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(u[2], u[0], u[1]))
        print()
        print("Averaged sampling times ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(s[2], s[0], s[1]))
        print("==========================================================")
        print("Incomplete training data! Sampling number =", r)
        print()
        print("Averaged Accuracies ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(a[5], a[3], a[4]))
        print()
        print("Averaged AUCs ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(ar[5], ar[3], ar[4]))
        print()
        print("Averaged Uncertainties ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(u[5], u[3], u[4]))
        print()
        print("Averaged sampling times ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(s[5], s[3], s[4]))
        print("==========================================================")
    print("==========================================================")
    print("Accuracy with complete data = {:.5f}".format(complete_accuracy))
    print("AUC with complete data = {:.5f}".format(complete_auc))
    print("Accuracy of trivial classifier = {:.5f}".format(
        max(sum(data[:, -1]) / len(data[:, -1]), (len(data[:, -1]) - sum(data[:, -1])) / len(data[:, -1]))))
    print("total time taken = {:.5f}".format(time.time() - start_time))
    print("Class counts in the data", Counter(list(data[:, -1])))
    print("Number of total instances =", data.shape[0], "\nNumber of attributes =", (data.shape[1] - 1))
