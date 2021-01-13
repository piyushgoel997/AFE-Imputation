import os
import numpy as np


def norm_col(da):
    return (da - np.mean(da)) / np.std(da)


directory = "C:/MyFolder/Thesis Work/AFE-Imputation/DataSets/Completely_Processed/"
d2 = "C:/MyFolder/Thesis Work/AFE-Imputation/DataSets/CP2/"
for filename in os.listdir(directory):
    if "_cat." in filename:
        continue
    print(filename)
    data = np.load(directory + filename).astype(np.float)
    for c in range(data.shape[1] - 1):
        data[:, c] = norm_col(data[:, c])
    print(data)
    np.save(d2+filename, data)
