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

names = {'abalone': "Abalone", 'adult': "Adult", 'avila': "Avila", 'bank': "Bank Marketing",
         'biodeg': "QSAR Biodegradation", 'cardiotocography': "Cardiotocography", 'car': "Car Evaluation",
         'credit': "default of credit card clients", 'drug': "Drug consumption", 'faults': "Steel Plates Faults",
         'frogs': "Anuran Calls (MFCCs)", 'mushroom': "Mushroom", 'obesity': "Obesity*",
         'online': "Online Shoppers Purchasing Intention", 'phishing': "Website Phishing", 'shill': "Shill Bidding",
         'spambase': "Spambase", 'statlog-is': "Statlog (Image Segmentation)",
         'statlog-ls': "Statlog (Landsat Satellite)", 'wine': "Wine Quality"}

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
    # acc (complete[0-6], incomplete[6-12])
    # auc (complete[12-18], incomplete[18-24])
    # complete acc[25], complete auc[26]
    nn_results[d] = [0] * 27
    dt_results[d] = [0] * 27

for f, r in results.items():
    dn = f.split("_")[0]
    clf = f.split("_")[-1]
    um = f.split("_")[-2]
    if clf == "nn":
        arr = nn_results[dn]
    else:
        arr = dt_results[dn]

    # for acc, auc
    for k in range(2):
        # for complete and incomplete
        for i, j in [(0, 0), (6, 3)]:
            a = r[k][percentage_index]
            arr[12 * k + 0 + i] += a[2 + j]
            arr[12 * k + 1 + i] += a[0 + j] - a[2 + j]
            if um == "b2":
                arr[12 * k + 2 + i] = a[1 + j] - a[2 + j]
            elif um == "conf":
                arr[12 * k + 3 + i] = a[1 + j] - a[2 + j]
            elif um == "va":
                arr[12 * k + 4 + i] = a[1 + j] - a[2 + j]
            else:  # e
                arr[12 * k + 5 + i] = a[1 + j] - a[2 + j]
    arr[25] += r[4]
    arr[26] += r[5]
    if clf == "nn":
        nn_results[dn] = arr
    else:
        dt_results[dn] = arr

for f in nn_results.keys():
    # normalize no sampling and random sampling
    for k in range(4):
        nn_results[f][6 * k + 0] /= 3
        nn_results[f][6 * k + 1] /= 3
        dt_results[f][6 * k + 0] /= 3
        dt_results[f][6 * k + 1] /= 3
    # normalize complete acc, complete auc, prior
    for k in range(1, 4):
        nn_results[f][-k] /= 3
        dt_results[f][-k] /= 3

import pandas as pd

df = pd.DataFrame.from_dict(nn_results, orient='index')


# \multicolumn{2}{c}{wine} & & \multirow{2}{*}{.839} & .649 & .039 & .048 & .067 & .059 & & \multirow{2}{*}{.772} & .636 & .023 & .043 & .044 & .048 \\
# (.633) & (12) & & & .629 & .018 & .021 & .018 & .007 & & & .621 & .041 & .023 & .036 & .042 \\
# \hline


def strip_str(x, sig_figs=3):
    x = str(float(x)) + "00000000"
    out = "." + x.split(".")[1][:sig_figs]
    if x[0] == '-':
        out = "-" + out
    return out


# for obesity add {Obesity* = Estimation of obesity levels based on eating habits and physical condition} to the caption
# for wine add {Wine Quality* = The wine quality data sets for white and red wines are combine by taking the color of the wine as an extra attribute}
def print_row(f, d, first):
    if first:
        row = "& & \\multirow{2}{*}{nn} & &"
    else:
        row = "(" + strip_str(priors[f]) + ") & (" + str(features[f]) + ") & \\multirow{2}{*}{rf} & &"
    row += "\\multirow{2}{*}{" + strip_str(d[-1]) + "}"
    for x in d[12:18]:
        row += " & " + strip_str(x)
    row += " & & \\multirow{2}{*}{" + strip_str(d[-2]) + "}"
    for x in d[0:6]:
        row += " & " + strip_str(x)

    row += " \\\\\n"
    if first:
        row += "\multicolumn{2}{c}{" + names[f] + "}  & & &"
    else:
        row += "& & & &"
    for x in d[18:24]:
        row += " & " + strip_str(x)
    row += " & &"
    for x in d[6:12]:
        row += " & " + strip_str(x)
    row += " \\\\\n"
    if not first:
        row += "\\hline"
    else:
        row += "\\cmidrule(lr){5-11} \\cmidrule(lr){13-19}"
    print(row)


for f in nn_results.keys():
    print_row(f, nn_results[f], True)
    print_row(f, dt_results[f], False)
    print()

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
