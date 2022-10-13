"""
Testing SARIMAX
The paper is available in:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
sys.path.append("..")
import func.cal as cal
import statsmodels.api as sm


l_x = 60                   # Data sequence length
l_y = 1                    # Label sequence length
save_fig = True                  # Whether to save picture
save_np = True                  # Whether to save np file
save_model = True               # Whether to save network model
ratio_train = 0.5               # Proportion of training datasets
fig_size = (16, 12)
ts_name_all = ["cli_dash", "HadCRUT5", "temp_month", "elect", "traffic", "sales"]
ts_name_folder = "temp_month"    # Name of the folder where the data resides
ts_name = "ERSSTv3b"       # Name of the selected time series
iv = 1                          # sampling interval, used for plotting curves
way = "mean"                    # The style of plot curves of real data and predict results

# para = sys.argv
# ts_name_folder, ts_name = para[1], para[2]

x_address = osp.join("../datasets", ts_name_folder, ts_name + ".npy")
x = np.load(x_address)
num = x.shape[0]                # The length of time series

num_train = int(ratio_train * num)
train_true, test_true = x[:num_train], x[num_train:num]     # get training dataset and test dataset

result_address = osp.join("../result", ts_name, "SARIMAX")
if not(osp.exists(result_address)):
    os.makedirs(result_address)

saramax_model = sm.tsa.SARIMAX(train_true, order=(1, 0, 0), trend='c')
model_train = saramax_model.fit()
model_test = model_train.apply(test_true)

train_predict = model_train.predict()
test_predict = model_test.predict()
if save_np:
    np.save(osp.join(result_address, "train_true_SARIMAX.npy"), train_true)
    np.save(osp.join(result_address, "test_true_SARIMAX.npy"), test_true)
    np.save(osp.join(result_address, "train_predict_SARIMAX.npy"), train_predict)
    np.save(osp.join(result_address, "test_predict_SARIMAX.npy"), test_predict)

e_linear = test_true - test_predict
cal.plot_distribute(e_linear, 40, 4, x_name="e")
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_SARIMAX_error_distribution.png"))

cal.plot_result(train_true, test_true, train_predict, test_predict, iv, way, fig_size)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_SARIMAX.png"))

r2_train = cal.get_r2_score(train_predict, train_true, axis=1)
r2_test = cal.get_r2_score(test_predict, test_true, axis=1)
print("{}\nSARIMAX: r2_train: {:.5f}  r2_test: {:.5f}".format(ts_name, r2_train, r2_test))

#
# print()
# plt.show()
# print()
