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
        self.remove_ratios = [0.25, 0.5, 0.75]
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
            tt = time.time()
            learn_idx = clf.next_features(_X, method=method)
            S.append((time.time() - tt) / len(_X))
            for i_, j in enumerate(learn_idx):
                _X[i_, j] = X_test_complete[i_, j]
            A.append(clf.test_accuracy(_X, Y_test))
            AR.append(clf.test_auc(_X, Y_test))
            U.append(clf.average_uncertainty(_X))
            print(method, "ended")
        # no sampling
        A.append(clf.test_accuracy(X_test, Y_test))
        AR.append(clf.test_auc(X_test, Y_test))
        U.append(clf.average_uncertainty(X_test))
        S.append(0)
        return A, AR, U, S

    def split_indices(self, exp_no):
        size = self.data.shape[0]
        start = int((exp_no * size) / self.NUM_FOLDS)
        stop = int(((exp_no + 1) * size) / self.NUM_FOLDS)
        test = list(range(start, stop))
        test = test[:min(len(test), 500)]
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

        acc_exp = np.zeros((len(self.remove_ratios), 6))
        auc_exp = np.zeros((len(self.remove_ratios), 6))
        uncert_exp = np.zeros((len(self.remove_ratios), 6))
        sampling_times_exp = np.zeros((len(self.remove_ratios), 6))

        for j, rr in enumerate(self.remove_ratios):
            print("Starting exp for remove ratio", rr)
            t1 = time.time()
            X_test = self.remove_data(X_test_complete, rr)
            ac, arc, uc, sc = self.run_exp(X_test, X_test_complete, Y_test, c_clf)
            print("Complete data finished in", time.time() - t1)
            t1 = time.time()
            ############################################################################################
            X_train = self.remove_data(X_train_complete, rr)
            clf = Classifier(self.CLASSIFIER_TYPE, categorical=self.categorical)
            clf.train(X_train, Y_train)
            ai, ari, ui, si = self.run_exp(X_test, X_test_complete, Y_test, clf)
            print("Incomplete data finished in", time.time() - t1)
            print(rr, ac, arc, sc, ai, ari, si)
            acc_exp[j] = np.array(ac + ai)
            auc_exp[j] = np.array(arc + ari)
            uncert_exp[j] = np.array(uc + ui)
            sampling_times_exp[j] = np.array(sc + si)
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
    acc = np.zeros((len(exp.remove_ratios), 6))
    auc = np.zeros((len(exp.remove_ratios), 6))
    uncertainty = np.zeros((len(exp.remove_ratios), 6))
    sampling_times = np.zeros((len(exp.remove_ratios), 6))

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
    # acc, auc roc, uncertainty, sampling times -> Axis 1 -> missing ratios 0.25, 0.50, 0.75
    #      Axis 2 -> random, leu, no first for complete and then for incomplete.
    pickle.dump(res, open("saved_results/" + args.data))

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

    for i in range(len(exp.remove_ratios)):
        acc[i] /= exp.NUM_FOLDS
        auc[i] /= exp.NUM_FOLDS
        uncertainty[i] /= exp.NUM_FOLDS
        sampling_times[i] /= exp.NUM_FOLDS
    complete_accuracy /= exp.NUM_FOLDS
    complete_auc /= exp.NUM_FOLDS
    for a, ar, u, s, r in zip(acc, auc, uncertainty, sampling_times, exp.remove_ratios):
        print("==========================================================")
        print("Complete training data! Remove ratio =", r)
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
        print("Incomplete training data! Remove ratio =", r)
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
