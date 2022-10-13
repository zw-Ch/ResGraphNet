import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


global_address = "HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.csv"
global_df = pd.read_csv(global_address)
global_value_full = global_df.loc[:, "Anomaly (deg C)"].values
np.save("HadCRUT5_global_full.npy", global_value_full)

low_con = global_df.loc[:, "Lower confidence limit (2.5%)"].values
up_con = global_df.loc[:, "Upper confidence limit (97.5%)"].values

global_value = global_value_full[:-3]
np.save("HadCRUT5_global.npy", global_value)

plt.figure(figsize=(12, 12))
plt.plot(global_value_full, c="red", label="Observation", alpha=0.5)
# plt.plot(low_con, c="blue", label="Lower confidence limit", alpha=0.3)
x_tick = [0, 360, 720, 1080, 1440, 1800]
x_label = ["1850", "1880", "1910", "1940", "1970", "2000"]
plt.xticks(x_tick, x_label, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Year", fontsize=30)
plt.ylabel("Anomaly ($^{\circ}$C)", fontsize=30)
# plt.legend(fontsize=30)
plt.savefig("HadCRUT5_global.png")


north_address = "HadCRUT.5.0.1.0.analysis.summary_series.northern_hemisphere.monthly.csv"
north_df = pd.read_csv(north_address)
north_value_full = north_df.loc[:, "Anomaly (deg C)"].values
np.save("HadCRUT5_northern_full.npy", north_value_full)
north_value = north_value_full[:-3]
np.save("HadCRUT5_northern.npy", north_value)
# plt.figure()
# plt.plot(value)


south_address = "HadCRUT.5.0.1.0.analysis.summary_series.southern_hemisphere.monthly.csv"
south_df = pd.read_csv(south_address)
south_value_full = south_df.loc[:, "Anomaly (deg C)"].values
np.save("HadCRUT5_southern_full.npy", south_value_full)
south_value = south_value_full[:-3]
np.save("HadCRUT5_southern.npy", south_value)
# plt.figure()
# plt.plot(value)


print()
plt.show()
print()
