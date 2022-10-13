import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ERA5 = pd.read_csv("gmt_ERA5.csv")
ERA5_value = ERA5.loc[:, ["ERA5 (degC)"]].values

HadCRUT5 = pd.read_csv("gmt_HadCRUT5.csv")
HadCRUT5_value = HadCRUT5.loc[:, "HadCRUT5 (degC)"].values
np.save("HadCRUT5.npy", HadCRUT5_value)

print()
plt.show()
print()
