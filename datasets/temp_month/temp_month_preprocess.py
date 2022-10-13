import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp


"""
2015_11_v3_GLB.TsERSSTv4.csv
"""
folder = "csv"
csv = "2015_11_v3_GLB.TsERSSTv4.csv"
df_address = osp.join(folder, csv)
df = pd.read_csv(df_address)

mon_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
value = df.loc[:, mon_list].values[:-1, :].astype(float).reshape(-1)
np.save("ERSSTv4.npy", value)
# plt.figure()
# plt.plot(value)

"""
2015_11_v3_GLB.TsERSSTv3.csv
"""
folder = "csv"
csv = "2013_01_v3_GLB.TsERSSTv3b.csv"
df_address = osp.join(folder, csv)
df = pd.read_csv(df_address)

mon_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
value = df.loc[:, mon_list].values[:-1, :].astype(float)
value = value.reshape(-1)
np.save("ERSSTv3b.npy", value)
plt.figure(figsize=(12, 12))
plt.plot(value, c="red")
x_tick = [0, 266, 532, 798, 1064, 1330, 1583]
x_label = ["1880.Jan", "1902.Jan", "1924.Jan", "1946.Jan", "1968.Jan", "1990.Jan", "2011.Jan"]
plt.xticks(x_tick, x_label, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Date", fontsize=30)
plt.ylabel("Temperature", fontsize=30)
plt.savefig("ERSSTv3b.png")

"""
2012_12_v3_GLB.Tsho2.csv
"""
folder = "csv"
csv = "2012_12_v3_GLB.Tsho2.csv"
df_address = osp.join(folder, csv)
df = pd.read_csv(df_address)

mon_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
value = df.loc[:, mon_list].values.astype(float).reshape(-1)
np.save("ho2.npy", value)
# plt.figure()
# plt.plot(value)

"""
2012_12_v3_GLB.Tsho2.csv
"""
folder = "csv"
csv = "2001-01_v2_GLB.TsRey.csv"
df_address = osp.join(folder, csv)
df = pd.read_csv(df_address)

mon_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
value = df.loc[:, mon_list].values[:-1, :].astype(float).reshape(-1)
np.save("Rey.npy", value)
# plt.figure()
# plt.plot(value)

"""
1997-05_NCAR+MCDW+NOAA_GLB.TsRey.csv
"""
folder = "csv"
csv = "1997-05_NCAR+MCDW+NOAA_GLB.TsRey.csv"
df_address = osp.join(folder, csv)
df = pd.read_csv(df_address)

mon_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
value = df.loc[:, mon_list].values[:-1, :].astype(float).reshape(-1)
np.save("NOAA.npy", value)
# plt.figure()
# plt.plot(value)


print()
plt.show()
print()
