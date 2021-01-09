from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

name = "wine_quality_merged"
file_name = "DataSets/Slightly_Processed/" + name + ".csv"
f = pd.read_csv(file_name, header=None, delimiter=',')
cols = f.columns
features = []
categorical = []
split_at = -1
label_encoder = LabelEncoder()
idx = 0

num_examples = len(f[cols[0]])

for c in cols[:split_at]:
    num_feature_values = len(set(f[c]))
    print(num_feature_values)
    if num_feature_values == 1:
        print("Ignoring", set(f[c]))
        continue
    if type(f.loc[0][c]) == str:
        print("Categorical", Counter(f[c]))
        features.append(label_encoder.fit_transform(f[c]).reshape((num_examples, 1)))
        categorical.append(idx)
    else:
        print(np.array(f[c]))
        features.append(np.array(f[c]).reshape((num_examples, 1)))
    idx += 1
print("Cat vars", categorical)
print("out")
for i, c in enumerate(cols[split_at:]):
    data = features.copy()
    num_feature_values = len(set(f[c]))
    print(num_feature_values, Counter(f[c]))
    data.append(label_encoder.fit_transform(f[c]).reshape((num_examples, 1)))
    data = np.concatenate(data, axis=1)
    print(data.shape)
    np.random.shuffle(data)
    n = name + "_" + str(i)
    np.save("DataSets/Processed/" + n + ".npy", data)
    np.save("DataSets/Processed/" + n + "_cat.npy", np.array(categorical))
