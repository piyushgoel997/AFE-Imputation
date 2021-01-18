import pickle
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np

percentage_index = 2  # for 75 %

# path = "data"
# priors = {}
# for f in listdir(path):
#     if isfile(join(path, f)) and "_cat." not in f:
#         data = np.load(str(join(path, f)))
#         priors[str(f).split("_")[0]] = max(np.sum(data[:, -1])/len(data), 1 - np.sum(data[:, -1])/len(data))

priors = {'abalone': 0.5017955470433325, 'adult': 0.7607182343065395, 'avila': 0.5598792351559879,
          'bank': 0.8830151954170445, 'biodeg': 0.6625592417061612, 'cardiotocography': 0.5159924741298212,
          'car': 0.7002314814814814, 'credit': 0.7787999999999999, 'drug': 0.529973474801061,
          'faults': 0.6532715095311695, 'frogs': 0.6143154968728284, 'mushroom': 0.517971442639094,
          'obesity': 0.5978209379441023, 'online': 0.8452554744525548, 'phishing': 0.5188470066518847,
          'sat': 0.5585081585081585, 'Sensorless': 0.5454545454545454, 'shill': 0.8932130991931656,
          'spambase': 0.6059552271245381, 'statlog-is': 0.5714285714285714, 'statlog-ls': 0.5409999999999999,
          'wine': 0.6330614129598277}

# path = "data"
# features = {}
# for f in listdir(path):
#     if isfile(join(path, f)) and "_cat." not in f:
#         data = np.load(str(join(path, f)))
#         features[str(f).split("_")[0]] = data.shape[1] - 1

features = {'abalone': 8, 'adult': 14, 'avila': 10, 'bank': 16, 'biodeg': 41, 'cardiotocography': 24, 'car': 6,
            'credit': 23, 'drug': 12, 'faults': 27, 'frogs': 22, 'mushroom': 21, 'obesity': 16, 'online': 17,
            'phishing': 9, 'sat': 36, 'Sensorless': 48, 'shill': 9, 'spambase': 57, 'statlog-is': 18, 'statlog-ls': 36,
            'wine': 12}

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

# \multicolumn{2}{c}{wine} & & \multirow{2}{*}{.839} & .649 & .039 & .048 & .067 & .059 & & \multirow{2}{*}{.772} & .636 & .023 & .043 & .044 & .048 \\
# (.633) & (12) & & & .629 & .018 & .021 & .018 & .007 & & & .621 & .041 & .023 & .036 & .042 \\
# \hline


def strip_str(x, sig_figs=3):
    x = str(float(x)) + "00000000"
    out = "." + x.split(".")[1][:sig_figs]
    if x[0] == '-':
        out = "-" + out
    return out


for f, d in nn_results.items():
    row = "\\multicolumn{2}{c}{" + f + "} & & \\multirow{2}{*}{" + strip_str(d[-2]) + "}"
    for x in d[10:15]:
        row += " & " + strip_str(x)
    row += " & & \\multirow{2}{*}{" + strip_str(d[-3]) + "}"
    for x in d[0:5]:
        row += " & " + strip_str(x)

    row += " \\\\\n"
    row += "(" + strip_str(priors[f]) + ") & (" + str(features[f]) + ") & &"
    for x in d[5:10]:
        row += " & " + strip_str(x)
    row += " & &"
    for x in d[15:20]:
        row += " & " + strip_str(x)
    row += " \\\\\n\\hline"
    print(row)

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
