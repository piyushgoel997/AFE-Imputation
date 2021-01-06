from collections import Counter

import numpy as np


def binarize_1(classes):
    mapping = {}
    b = True
    for cls, _ in sorted(dict(Counter(classes)).items(), key=lambda i: i[1]):
        if b:
            mapping[cls] = 0
        else:
            mapping[cls] = 1
        b = not b
    return np.array([mapping[cls] for cls in classes])


def binarize_2(classes):
    mapping = {}
    b = True
    ct = 0
    for cls, _ in sorted(dict(Counter(classes)).items(), key=lambda i: i[0]):
        if ct < len(classes) / 2:
            mapping[cls] = 0
            ct += _
        else:
            mapping[cls] = 1
    mapping[sorted(set(classes))[-1]] = 1
    print(mapping)
    return np.array([mapping[cls] for cls in classes])


def car_bin(classes):
    mapping = {2: 0, 1: 1, 3: 1, 0: 1}
    return np.array([mapping[cls] for cls in classes])


# name = "abalone_0"  # {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0, 6.0: 0, 7.0: 0, 8.0: 0, 9.0: 1, 10.0: 1, 11.0: 1, 12.0: 1, 13.0: 1, 14.0: 1, 15.0: 1, 16.0: 1, 17.0: 1, 18.0: 1, 19.0: 1, 20.0: 1, 21.0: 1, 22.0: 1, 23.0: 1, 24.0: 1, 25.0: 1, 26.0: 1, 27.0: 1} 2081.0 2096.0
# name = "avila_0"  # {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 1, 6.0: 1, 7.0: 1, 8.0: 1, 9.0: 1, 10.0: 1, 11.0: 1} 9184.0 11683.0
# name = "car_0"  # unacc (2) as 0 and rest as 1.  518 1210
# name = "cardiotocography_0" # {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 1, 5.0: 1, 6.0: 1, 7.0: 1, 8.0: 1, 9.0: 1} 1029.0 1097.0
# name = "cardiotocography_1" # {0.0: 0, 1.0: 1, 2.0: 1}471.0 1655.0
# name = "faults_0" # {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0, 6.0: 1} 673.0 1268.0
# name = "frogs_0" # {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 1} 4420.0 2775.0
# name = "frogs_1" # {0.0: 0, 1.0: 1, 2.0: 1, 3.0: 1, 4.0: 1, 5.0: 1, 6.0: 1, 7.0: 1}3045.0 4150.0
# name = "frogs_2" # {0.0: 0, 1.0: 0, 2.0: 1, 3.0: 1, 4.0: 1, 5.0: 1, 6.0: 1, 7.0: 1, 8.0: 1, 9.0: 1}3045.0 4150.0
# name = "sat_0"  # {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}2841 3594
# name = "Sensorless_drive_diagnosis_0" # {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0, 6.0: 1, 7.0: 1, 8.0: 1, 9.0: 1, 10.0: 1}26595.0 31914.0
# name = "statlog-is_0" # {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 1, 5.0: 1, 6.0: 1} 990.0 1320.0
name = "statlog-ls_0"  # {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1} 918 1082

d = np.load("DataSets/Processed/" + name + ".npy", allow_pickle=True)
d[:, -1] = binarize_2(d[:, -1])
print(sum(d[:, -1]), d.shape[0] - sum(d[:, -1]))
np.save("DataSets/Completely_Processed/" + name + ".npy", d)
