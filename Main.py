import argparse
import random
import sys  # TODO
import time
import warnings
from collections import Counter

import numpy as np

from Classifier import Classifier

warnings.filterwarnings("ignore")

NUM_EXPERIMENTS = 1  # TODO
CLASSIFIER_TYPE = "nn"


def remove_data(X, r):
    _X = np.copy(X)
    for i in range(_X.shape[0]):
        for j in range(_X.shape[1]):
            if random.random() < r:
                _X[i, j] = np.nan
    return _X


def run_exp(X_test, X_test_complete, Y_test, clf):
    a = []
    s = []
    for method in ["random", "leu"]:
        _X = np.copy(X_test)
        t = time.time()
        learn_idx = clf.next_features(X_test, method=method)
        s.append((time.time() - t) / len(_X))
        for i, j in enumerate(learn_idx):
            _X[i, j] = X_test_complete[i, j]
        print(method)
        a.append(clf.test(_X, Y_test))
    # no sampling
    a.append(clf.test(X_test, Y_test))
    s.append(0)
    return a, s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()

    args.data = "car_0"  # TODO

    # sys.stdout = open("logs/" + args.data + ".txt", "w")  # TODO

    # load data
    data = np.load("data/" + args.data + ".npy", allow_pickle=True)
    remove_ratios = [0.25, 0.50, 0.75]
    # Complete train random, complete train afe, without sampling
    # incomplete train random, incomplete train afe, without sampling
    acc = np.zeros((len(remove_ratios), 6))
    sampling_times = np.zeros((len(remove_ratios), 6))
    start_time = time.time()
    complete_accuracy = 0
    for _ in range(NUM_EXPERIMENTS):
        exp_start_time = time.time()
        np.random.shuffle(data)
        X_train_complete = data[:int(0.8 * len(data)), :-1]
        Y_train = data[:int(0.8 * len(data)), -1]

        X_test_complete = data[int(0.8 * len(data)):, :-1]
        Y_test = data[int(0.8 * len(data)):, -1]

        clf = Classifier(CLASSIFIER_TYPE, categorical=list(range(X_train_complete.shape[1])))
        clf.train(X_train_complete, Y_train, incomplete=False)
        complete_accuracy += clf.test(X_test_complete, Y_test, incomplete=False)

        for j, r in enumerate(remove_ratios):
            # train classifier on complete data
            t = time.time()
            X_train = X_train_complete
            clf = Classifier(CLASSIFIER_TYPE, categorical=list(range(X_train_complete.shape[1])))
            clf.train(X_train, Y_train, incomplete=False)
            print("training clf (complete data) done in ", time.time() - t)
            t = time.time()
            X_test = remove_data(X_test_complete, r)
            ac, sc = run_exp(X_test, X_test_complete, Y_test, clf)
            print(ac, sc)
            print("experiment (complete data) done in", time.time() - t)
            ############################################################################################
            t = time.time()
            X_train = remove_data(X_train_complete, r)
            clf = Classifier(CLASSIFIER_TYPE, categorical=list(range(X_train_complete.shape[1])))
            clf.train(X_train, Y_train)
            print("training clf (incomplete data) done in ", time.time() - t)
            t = time.time()
            X_test = remove_data(X_test_complete, r)
            ai, si = run_exp(X_test, X_test_complete, Y_test, clf)
            print(ai, si)
            print("experiment (complete data) done in", time.time() - t)

            acc[j] += np.array(ac + ai)
            sampling_times[j] += np.array(sc + si)
        print("Experiment done in", time.time() - exp_start_time)
    for i in range(len(remove_ratios)):
        acc[i] /= NUM_EXPERIMENTS
        sampling_times[i] /= NUM_EXPERIMENTS
    complete_accuracy /= NUM_EXPERIMENTS
    for a, s, r in zip(acc, sampling_times, remove_ratios):
        print("==========================================================")
        print("Complete data! Remove ratio =", r)
        print()
        print("Averaged Accuracies ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(a[2], a[0], a[1]))
        print()
        print("Averaged sampling times ->\nNo sampling = {:.5f}\nRandom selection = {:.5f}\n"
              "Least Expected Uncertainty = {:.5f}".format(s[2], s[0], s[1]))
        print("==========================================================")
        print("Incomplete data! Remove ratio =", r)
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
    print("Number of total instances =", data.shape[0], "\nNumber of attributes =", data.shape[1])

# TODO
# ==========================================================
# Complete data! Remove ratio = 0.25
#
# Averaged Accuracies ->
# No sampling = 0.89740
# Random selection = 0.89740
# Least Expected Uncertainty = 0.91243
#
# Averaged sampling times ->
# No sampling = 0.00000
# Random selection = 0.00003
# Least Expected Uncertainty = 0.10272
# ==========================================================
# Incomplete data! Remove ratio = 0.25
#
# Averaged Accuracies ->
# No sampling = 0.87341
# Random selection = 0.87428
# Least Expected Uncertainty = 0.88642
#
# Averaged sampling times ->
# No sampling = 0.00000
# Random selection = 0.00003
# Least Expected Uncertainty = 0.09738
# ==========================================================
# ==========================================================
# Complete data! Remove ratio = 0.5
#
# Averaged Accuracies ->
# No sampling = 0.77225
# Random selection = 0.76965
# Least Expected Uncertainty = 0.78931
#
# Averaged sampling times ->
# No sampling = 0.00000
# Random selection = 0.00002
# Least Expected Uncertainty = 0.09819
# ==========================================================
# Incomplete data! Remove ratio = 0.5
#
# Averaged Accuracies ->
# No sampling = 0.76012
# Random selection = 0.76040
# Least Expected Uncertainty = 0.77832
#
# Averaged sampling times ->
# No sampling = 0.00000
# Random selection = 0.00002
# Least Expected Uncertainty = 0.10394
# ==========================================================
# ==========================================================
# Complete data! Remove ratio = 0.75
#
# Averaged Accuracies ->
# No sampling = 0.70751
# Random selection = 0.69942
# Least Expected Uncertainty = 0.72717
#
# Averaged sampling times ->
# No sampling = 0.00000
# Random selection = 0.00002
# Least Expected Uncertainty = 0.11306
# ==========================================================
# Incomplete data! Remove ratio = 0.75
#
# Averaged Accuracies ->
# No sampling = 0.71156
# Random selection = 0.71156
# Least Expected Uncertainty = 0.72081
#
# Averaged sampling times ->
# No sampling = 0.00000
# Random selection = 0.00002
# Least Expected Uncertainty = 0.09699
# ==========================================================
# ==========================================================
# Accuracy with complete data = 0.97543
# Accuracy of trivial classifier = 0.74016
# total time taken = 9675.18560
# Class counts in the data Counter({1.0: 1279, 0.0: 449})
# Number of total instances = 1728
# Number of attributes = 22
