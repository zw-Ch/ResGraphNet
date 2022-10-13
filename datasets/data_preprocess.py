import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from func.well import NewWellInfo


# # Electricity
# sample = 2              # 采样间隔
# ele = np.load("electricity/electricity.npy")
# ele = ele[0, :]
# ele_sam = np.zeros(shape=(int(ele.shape[0] / sample)))
# for i in range(ele_sam.shape[0]):
#     ele_sam[i] = ele[i * sample]
# np.save("electricity/electricity_sam.npy", ele_sam)

# # sales
# sample = 20
# sales = pd.read_csv("sales/sales data-set.csv")
# sale = sales.Weekly_Sales.values
# sale_sam = np.zeros(shape=int(sale.shape[0] / sample))
# for i in range(sale_sam.shape[0]):
#     sale_sam[i] = sale[i * sample]
# np.save("sales/sales_sam", sale_sam)

# # solar
# sample = 1
# solar = pd.read_csv("solar/solar.csv")
# solar = solar.TOTAL_YIELD.values
# solar_sam = np.zeros(shape=int(solar.shape[0] / sample))
# for i in range(solar_sam.shape[0]):
#     solar_sam[i] = solar[i * sample]
# np.save("solar/solar_sam.npy", solar_sam)

# # PM2.5
# sample = 1
# pm25 = pd.read_csv("pm25/pm25.csv")
# pm25 = pm25.Iws.values
# pm25_sam = np.zeros(shape=int(pm25.shape[0] / sample))
# for i in range(pm25_sam.shape[0]):
#     pm25_sam[i] = pm25[i * sample]
# np.save("pm25/pm25_sam.npy", pm25_sam)

# # ABEV
# abev = pd.read_csv("abev/abev.csv")
# abev = abev.close.values
# np.save("abev/abev_sam.npy", abev)

# # traffic
# sample = 50
# traffic = pd.read_csv("traffic/traffic.csv")
# traffic = traffic.median7.values
# traffic_sam = np.zeros(shape=int(traffic.shape[0] / sample))
# for i in range(traffic_sam.shape[0]):
#     traffic_sam[i] = traffic[i * sample]
# np.save("traffic/traffic_sam.npy", traffic_sam)

# # 油井数据
# well_info = NewWellInfo(name="new_WellInfo")
# well_info_data = well_info.data
# well_info_data_one = well_info_data[8]
# plt.figure(figsize=(12, 12))
# plt.plot(well_info_data_one)
# np.save("WellData/well_one.npy", well_info_data_one)

# HadCRUT5
had = df = pd.read_csv('HadCRUT/HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.csv')
anomaly = had.loc[:, "Anomaly (deg C)"].values.reshape(-1)
np.save("HadCRUT/HadCRUT0.npy", anomaly)
lower = had.loc[:, "Lower confidence limit (2.5%)"].values.reshape(-1)
np.save("HadCRUT/HadCRUT1.npy", lower)
upper = had.loc[:, "Upper confidence limit (97.5%)"].values.reshape(-1)
np.save("HadCRUT/HadCRUT2.npy", upper)

print()
plt.show()
print()
