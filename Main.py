import argparse
import random
import sys
import time
import warnings
from collections import Counter
from multiprocessing import Pool

import numpy as np

from Classifier import Classifier

warnings.filterwarnings("ignore")


class Exp:

    def __init__(self, uncertainty_measure, feature_cols):
        self.NUM_EXPERIMENTS = 10
        self.MAX_TEST_POINTS = 500
        self.CLASSIFIER_TYPE = "nn"
        self.remove_ratios = [0.25, 0.5, 0.75]
        self.uncertainty_measure = uncertainty_measure.split("*")[0]
        self.set_alpha = len(uncertainty_measure.split("*")) == 2
        self.feature_cols = feature_cols

    def remove_data(self, X, r):
        _X = np.copy(X)
        for i in range(_X.shape[0]):
            for j in range(len(self.feature_cols) - 1):
                if random.random() < r:
                    for i_ in range(self.feature_cols[j], self.feature_cols[j + 1]):
                        _X[i, i_] = np.nan
        return _X

    def run_exp(self, X_test, X_test_complete, Y_test, clf):
        a = []
        s = []
        for method in ["random", "leu"]:
            print(method, "started")
            _X = np.copy(X_test)
            t = time.time()
            learn_idx = clf.next_features(X_test, method=method)
            s.append((time.time() - t) / len(_X))
            for i, j in enumerate(learn_idx):
                _X[i, j] = X_test_complete[i, j]
            a.append(clf.test(_X, Y_test))
            print(method, "ended")
        # no sampling
        a.append(clf.test(X_test, Y_test))
        s.append(0)
        return a, s

    def one_exp(self, data):
        np.random.shuffle(data)
        X_train_complete = data[:int(0.8 * len(data)), :-1]
        Y_train = data[:int(0.8 * len(data)), -1]

        X_test_complete = data[max(int(0.8 * len(data)), len(data) - self.MAX_TEST_POINTS):, :-1]
        Y_test = data[max(int(0.8 * len(data)), len(data) - self.MAX_TEST_POINTS):, -1]

        c_clf = Classifier(self.CLASSIFIER_TYPE, categorical=list(range(X_train_complete.shape[1])),
                           uncertainty_measure=self.uncertainty_measure, set_alpha=self.set_alpha)
        c_clf.train(X_train_complete, Y_train, incomplete=False)
        complete_accuracy = c_clf.test(X_test_complete, Y_test, incomplete=False)

        acc = np.zeros((len(self.remove_ratios), 6))
        sampling_times = np.zeros((len(self.remove_ratios), 6))

        for j, r in enumerate(self.remove_ratios):
            print("Starting exp for remove ratio", r)
            t1 = time.time()
            X_test = self.remove_data(X_test_complete, r)
            ac, sc = self.run_exp(X_test, X_test_complete, Y_test, c_clf)
            print("Complete data finished in", time.time() - t1)
            t1 = time.time()
            ############################################################################################
            X_train = self.remove_data(X_train_complete, r)
            clf = Classifier(self.CLASSIFIER_TYPE, categorical=list(range(X_train_complete.shape[1])))
            clf.train(X_train, Y_train)
            ai, si = self.run_exp(X_test, X_test_complete, Y_test, clf)
            print("Incomplete data finished in", time.time() - t1)
            print(r, ac, sc, ai, si)
            acc[j] = np.array(ac + ai)
            sampling_times[j] = np.array(sc + si)
        return acc, sampling_times, complete_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--clf', default="nn")
    parser.add_argument('--um', default="confidence")
    parser.add_argument('--parallel', dest='parallel', action='store_true')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false')
    parser.set_defaults(parallel=True)
    args = parser.parse_args()

    start_time = time.time()

    # args.data = "car_0"
    CLASSIFIER_TYPE = args.clf
    sys.stdout = open("logs/" + args.data + "_" + args.clf + "_" + args.um + ".txt", "w")

    # load data
    data = np.load("data/" + args.data + ".npy", allow_pickle=True)
    feature_cols = np.load("data/" + args.data + "_fc.npy")
    print("Data loaded in", time.time() - start_time)

    exp = Exp(args.um, feature_cols)
    acc = np.zeros((len(exp.remove_ratios), 6))
    sampling_times = np.zeros((len(exp.remove_ratios), 6))

    complete_accuracy = 0

    print("Starting experiments")

    exp_list = []
    for _ in range(exp.NUM_EXPERIMENTS):
        exp_list.append(np.copy(data))

    if args.parallel:
        pool = Pool(10)
        res = pool.map_async(exp.one_exp, exp_list).get()
    else:
        res = map(exp.one_exp, exp_list)

    for r in res:
        a, s, ca = r
        acc += a
        sampling_times += s
        complete_accuracy += ca
        t = time.time()
    print("Experiments finished")

    for i in range(len(exp.remove_ratios)):
        acc[i] /= exp.NUM_EXPERIMENTS
        sampling_times[i] /= exp.NUM_EXPERIMENTS
    complete_accuracy /= exp.NUM_EXPERIMENTS
    for a, s, r in zip(acc, sampling_times, exp.remove_ratios):
        print("==========================================================")
        print("Complete training data! Remove ratio =", r)
        print()
        print("Averaged Accuracies ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(a[2], a[0], a[1]))
        print()
        print("Averaged sampling times ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(s[2], s[0], s[1]))
        print("==========================================================")
        print("Incomplete training data! Remove ratio =", r)
        print()
        print("Averaged Accuracies ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(a[5], a[3], a[4]))
        print()
        print("Averaged sampling times ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(s[5], s[3], s[4]))
        print("==========================================================")
    print("==========================================================")
    print("Accuracy with complete data = {:.5f}".format(complete_accuracy))
    print("Accuracy of trivial classifier = {:.5f}".format(
        max(sum(data[:, -1]) / len(data[:, -1]), (len(data[:, -1]) - sum(data[:, -1])) / len(data[:, -1]))))
    print("total time taken = {:.5f}".format(time.time() - start_time))
    print("Class counts in the data", Counter(list(data[:, -1])))
    print("Number of total instances =", data.shape[0], "\nNumber of attributes =", (data.shape[1] - 1))
