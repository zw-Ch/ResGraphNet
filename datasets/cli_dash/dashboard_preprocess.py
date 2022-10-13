import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


address = "HadSST3.txt"
txt = np.loadtxt(address)
value = txt[:, 1:].reshape(-1)
np.save("HadSST3.npy", value)

address = "Berkeley_Earth.txt"
txt = np.loadtxt(address, dtype=str)
value = txt[1:, 2].astype(float)
np.save("Berkeley_Earth.npy", value)

address = "ERA5.csv"
df = pd.read_csv(address)
value_global = df.loc[:, "global"].values
value_Euro = df.loc[:, "European"].values
np.save("ERA5_Global.npy", value_global)
np.save("ERA5_European.npy", value_Euro)

print()
plt.show()
print()
