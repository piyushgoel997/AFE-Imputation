import math
import os

import numpy as np
import pandas as pd

done = {}
directory = "C:/MyFolder/Thesis Work/AFE-Imputation/DataSets/Completely_Processed/"

for filename in os.listdir(directory):
    if "_cat." in filename:
        continue
    if filename.split("_")[0] in done:
        continue
    done[filename.split("_")[0]] = ""
    data = np.load(directory + filename)[:, :-1]
    data = pd.DataFrame(data)
    p = len(data.columns)

    corr = np.array(data.corr())

    rho = 0
    for i in range(p):
        temp = 0
        for j in range(i):
            temp += corr[i, j] ** 2
        rho += math.sqrt(temp)
    rho = (2 / ((p - 1) * p)) * rho
    print("\n" + filename.split("_")[0], "\nRho (from the paper)", rho, "\nL2 norm of the corr matrix",
          np.linalg.norm(corr), "\nInformation", math.log10(len(data) / p))
