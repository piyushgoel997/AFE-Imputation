import pickle
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np

percentage_index = 2  # for 75 %

priors = {"abalone": 0.547282739, "adult": 0.66666, "avila": 0.638376384, "bank": 0.883015195, "biodeg": 0.66256,
          "cardiotocography": 0.59172, "car": 0.740162037, "drug": 0.529973475, "faults": 0.65791, "frogs": 0.68965,
          "mushroom": 0.517971443, "obesity": 0.597820938, "phishing": 0.51885, "shill": 0.89321,
          "spambase": 0.60596, "statlog-is": 0.57143, "statlog-ls": 0.5455, "wine": 0.63306}

directory = "saved_results"
results = {}
for filename in listdir(directory):
    if not isfile(join(directory, filename)):
        continue
    res = []
    f = pickle.load(open(join(directory, filename), 'rb'))
    for i in range(len(f[0])):
        r = f[0][i]
        for j in range(1, len(f)):
            r += f[j][i]
        res.append(r / len(f))
    results[filename] = res

# now I have the sum of all the the folds in the results directory, for every experiment
# now, combine different exp types for data sets

nn_results = {}
dt_results = {}

for d in [x.split("_")[0] for x in results.keys()]:
    # acc (complete[0-5], incomplete[5-10])
    # auc (complete[10-15], incomplete[15-20])
    # uncert (complete[20-25], incomplete[25-30])
    # NOPE - sampling time (complete[30-35], incomplete[35-40])
    # complete acc[40], complete auc[41], prior[42]
    nn_results[d] = [0] * 33
    dt_results[d] = [0] * 33

for f, r in results.items():
    dn = f.split("_")[0]
    clf = f.split("_")[-1]
    um = f.split("_")[-2]
    if clf == "nn":
        arr = nn_results[dn]
    else:
        arr = dt_results[dn]

    # for acc, auc, uncert, sampling times
    for k in range(3):
        # for complete and incomplete
        for i, j in [(0, 0), (5, 3)]:
            a = r[k][percentage_index]
            arr[10 * k + 0 + i] += a[2 + j]
            arr[10 * k + 1 + i] += a[0 + j] - a[2 + j]
            if um == "conf":
                arr[10 * k + 2 + i] = a[1 + j] - a[2 + j]
            elif um == "va":
                arr[10 * k + 3 + i] = a[1 + j] - a[2 + j]
            else:  # e
                arr[10 * k + 4 + i] = a[1 + j] - a[2 + j]
    arr[30] += r[4]
    arr[31] += r[5]
    if clf == "nn":
        nn_results[dn] = arr
    else:
        dt_results[dn] = arr

for f in nn_results.keys():
    # normalize no sampling and random sampling
    for k in range(6):
        nn_results[f][5 * k + 0] /= 3
        nn_results[f][5 * k + 1] /= 3
        dt_results[f][5 * k + 0] /= 3
        dt_results[f][5 * k + 1] /= 3
    # normalize complete acc, complete auc, prior
    for k in range(1, 4):
        nn_results[f][-k] /= 3
        dt_results[f][-k] /= 3
    # make the uncert thing from increase in uncert to decrease in uncert
    # for k in range(20, 30):
    #     nn_results[f][k] *= -1
    #     dt_results[f][k] *= -1

# import pandas as pd
#
# df = pd.DataFrame.from_dict(nn_results, orient='index')

# TODO now just print rows according to the table, add prior information first.
print(nn_results)
print(dt_results)

# box plots

acc = []
auc = []
uncert = []
for r in nn_results.values():
    acc.append(r[1:5] + r[6:10])
    auc.append(r[11:15] + r[16:20])
    uncert.append(r[21:25] + r[26:30])


def make_boxplot(a, name):
    plt.boxplot(x=np.array(a)[:, :4])
    plt.xticks([1, 2, 3, 4], ["RS", "LEU_1", "LEU_2", "LEU_3"])
    plt.title(name + " (Complete Data)")
    plt.show()

    plt.boxplot(x=np.array(a)[:, 4:])
    plt.xticks([1, 2, 3, 4], ["RS", "LEU_1", "LEU_2", "LEU_3"])
    plt.title(name + " (Incomplete Data)")
    plt.show()


make_boxplot(acc, "Accuracy")
make_boxplot(auc, "AU-ROC")
make_boxplot(uncert, "Uncertainty")
