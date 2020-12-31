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

    def __init__(self, ):
        self.NUM_EXPERIMENTS = 10
        self.MAX_TEST_POINTS = 500
        self.CLASSIFIER_TYPE = "nn"
        self.remove_ratios = [0.25, 0.50, 0.75]

    def remove_data(self, X, r):
        _X = np.copy(X)
        for i in range(_X.shape[0]):
            for j in range(_X.shape[1]):
                if random.random() < r:
                    _X[i, j] = np.nan
        return _X

    def run_exp(self, X_test, X_test_complete, Y_test, clf):
        a = []
        s = []
        for method in ["random", "leu"]:
            _X = np.copy(X_test)
            t = time.time()
            learn_idx = clf.next_features(X_test, method=method)
            s.append((time.time() - t) / len(_X))
            for i, j in enumerate(learn_idx):
                _X[i, j] = X_test_complete[i, j]
            a.append(clf.test(_X, Y_test))
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

        c_clf = Classifier(self.CLASSIFIER_TYPE, categorical=list(range(X_train_complete.shape[1])))
        c_clf.train(X_train_complete, Y_train, incomplete=False)
        complete_accuracy = c_clf.test(X_test_complete, Y_test, incomplete=False)

        acc = np.zeros((len(self.remove_ratios), 6))
        sampling_times = np.zeros((len(self.remove_ratios), 6))

        for j, r in enumerate(self.remove_ratios):
            X_test = self.remove_data(X_test_complete, r)
            ac, sc = self.run_exp(X_test, X_test_complete, Y_test, c_clf)
            ############################################################################################
            X_train = self.remove_data(X_train_complete, r)
            clf = Classifier(self.CLASSIFIER_TYPE, categorical=list(range(X_train_complete.shape[1])))
            clf.train(X_train, Y_train)
            X_test = self.remove_data(X_test_complete, r)
            ai, si = self.run_exp(X_test, X_test_complete, Y_test, clf)
            print(r, ac, sc, ai, si)

            acc[j] = np.array(ac + ai)
            sampling_times[j] = np.array(sc + si)
        return [acc, sampling_times, complete_accuracy]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--clf', default="nn")
    args = parser.parse_args()

    # args.data = "car_0"
    CLASSIFIER_TYPE = args.clf
    sys.stdout = open("logs/" + args.data + ".txt", "w")

    # load data
    data = np.load("data/" + args.data + ".npy", allow_pickle=True)
    print("data loaded")

    exp = Exp()
    acc = np.zeros((len(exp.remove_ratios), 6))
    sampling_times = np.zeros((len(exp.remove_ratios), 6))
    start_time = time.time()
    complete_accuracy = 0

    exp_list = []
    for _ in range(exp.NUM_EXPERIMENTS):
        exp_list.append(np.copy(data))

    pool = Pool()
    res = pool.map(exp.one_exp, exp_list)
    print("Starting experiments")
    for a, s, ca in res:
        acc += a
        sampling_times += s
        complete_accuracy += ca
    pool.close()

    for i in range(len(exp.remove_ratios)):
        acc[i] /= exp.NUM_EXPERIMENTS
        sampling_times[i] /= exp.NUM_EXPERIMENTS
    complete_accuracy /= exp.NUM_EXPERIMENTS
    for a, s, r in zip(acc, sampling_times, exp.remove_ratios):
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
