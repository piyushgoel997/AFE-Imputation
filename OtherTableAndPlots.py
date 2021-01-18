import pickle
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np

percentage_index = 2  # for 75 %

directory = "other_saved_results"
results = {}
for filename in listdir(directory):
    if not isfile(join(directory, filename)):
        continue
    res = []
    f = pickle.load(open(join(directory, filename), 'rb'))
    for i in range(len(f[0])):
        try:
            r = f[0][i].T
        except:
            r = f[0][i]
        for j in range(1, len(f)):
            try:
                r += f[j][i].T
            except:
                r += f[j][i]
        res.append(r / len(f))
    results[filename] = res

# now I have the sum of all the the folds in the results directory, for every experiment
# now, combine different exp types for data sets

nn_results = {}
dt_results = {}

for d in [x.split("_")[0] for x in results.keys()]:
    # acc (complete[0-12], incomplete[12-24])
    # auc (complete[24-36], incomplete[36-48])
    # uncert (complete[48-60], incomplete[60-72])
    # complete acc[72], complete auc[73]
    nn_results[d] = [0] * 74
    dt_results[d] = [0] * 74

for f, r in results.items():
    dn = f.split("_")[0]
    clf = f.split("_")[-1]
    um = f.split("_")[-2]
    if clf == "nn":
        arr = nn_results[dn]
    else:
        arr = dt_results[dn]

    # for acc, auc, uncert
    for k in range(3):
        # for complete and incomplete
        for i, j in [(0, 0), (12, 3)]:
            a = r[k]
            nos = a[0][2 + j]
            arr[24 * k + 0 + i] += a[0][0 + j] - nos
            arr[24 * k + 1 + i] += a[1][0 + j] - nos
            arr[24 * k + 2 + i] += a[2][0 + j] - nos
            if um == "conf":
                arr[24 * k + 3 + i] = a[0][1 + j] - nos
                arr[24 * k + 4 + i] = a[1][1 + j] - nos
                arr[24 * k + 5 + i] = a[2][1 + j] - nos
            elif um == "va":
                arr[24 * k + 6 + i] = a[0][1 + j] - nos
                arr[24 * k + 7 + i] = a[1][1 + j] - nos
                arr[24 * k + 8 + i] = a[2][1 + j] - nos
            else:  # e
                arr[24 * k + 9 + i] = a[0][1 + j] - nos
                arr[24 * k + 10 + i] = a[1][1 + j] - nos
                arr[24 * k + 11 + i] = a[2][1 + j] - nos
    arr[72] += r[4]
    arr[73] += r[5]
    if clf == "nn":
        nn_results[dn] = arr
    else:
        dt_results[dn] = arr

for f in nn_results.keys():
    # normalize random sampling
    for k in range(6):
        nn_results[f][12 * k + 0] /= 3
        nn_results[f][12 * k + 1] /= 3
        nn_results[f][12 * k + 2] /= 3
        dt_results[f][12 * k + 0] /= 3
        dt_results[f][12 * k + 1] /= 3
        dt_results[f][12 * k + 2] /= 3

    # normalize complete acc, complete auc
    for k in range(1, 3):
        nn_results[f][-k] /= 3
        dt_results[f][-k] /= 3

import pandas as pd

df = pd.DataFrame.from_dict(nn_results, orient='index')

# now just print rows according to the table
# print(nn_results)
# print(dt_results)

# box plots

acc = []
auc = []
uncert = []
for r in dt_results.values():
    acc.append(r[:24])
    auc.append(r[24:48])
    uncert.append(r[48:74])


def make_plot(b, name):
    mean = np.mean(b, axis=0)
    std = np.std(b, axis=0)
    ci = std  # TODO
    # complete and incomplete
    for k in range(2):
        x = [1, 2, 3]
        for i, color, um in [(0, 'r', "Random"), (1, 'g', "LEU_1"), (2, 'b', "LEU_2"), (3, 'y', "LEU_3")]:
            plt.plot(x, mean[12 * k + 3 * i + 0:12 * k + 3 * i + 3], color=color, label=um)
            plt.fill_between(x, mean[12 * k + 3 * i + 0:12 * k + 3 * i + 3] - ci[12 * k + 3 * i + 0:12 * k + 3 * i + 3],
                             mean[12 * k + 3 * i + 0:12 * k + 3 * i + 3] + ci[12 * k + 3 * i + 0:12 * k + 3 * i + 3],
                             color=color, alpha=0.1)
        if k == 0:
            plt.title(name + " (Complete Data)")
        else:
            plt.title(name + " (Incomplete Data)")
        plt.legend()
        plt.show()


make_plot(np.array(acc), "Accuracy")
make_plot(np.array(auc), "AUC")
