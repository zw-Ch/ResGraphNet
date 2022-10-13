"""
Using ML Models to predict the time series
"""
import datetime
import numpy as np
import pandas as pd
import os
import os.path as osp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor

import sys
sys.path.append("..")
import func.cal as cal


l_x = 60                   # Data sequence length
l_y = 1                    # Label sequence length
ml_style_all = ["forest", "linear", "svr", "sgd"]
ml_style = "svr"
save_fig = True                  # Whether to save picture
save_txt = False                  # Whether to save txt
save_np = True                  # Whether to save np file
save_model = True               # Whether to save network model
ratio_train = 0.5               # Proportion of training datasets
fig_size = (8, 8)
ts_name_all = ["cli_dash", "HadCRUT5", "temp_month", "temp_year", "elect", "traffic", "sales"]
ts_name_folder = "HadCRUT5"    # Name of the folder where the data resides
ts_name = "HadCRUT5_global"       # Name of the selected time series
iv = 1                          # sampling interval, used for plotting curves
way = "mean"                    # The style of plot curves of real data and predict results

para = sys.argv
ts_name_folder, ts_name, ml_style = para[1], para[2], para[3]

x_address = osp.join("../datasets", ts_name_folder, ts_name + ".npy")
x = np.load(x_address)
num = x.shape[0]                # The length of time series

result_address = osp.join("../result", ts_name, "MLModel")
if not(osp.exists(result_address)):
    os.makedirs(result_address)

num_train = int(ratio_train * num)
data_train, data_test = x[:num_train], x[num_train:num]     # get training dataset and test dataset

# Using machine learning to predict time series
print("\nRunning, Machine Learning")
x_train, y_train = cal.create_inout_sequences(data_train, l_x, l_y, style="arr")         # 训练集序列
x_test, y_test = cal.create_inout_sequences(data_test, l_x, l_y, style="arr")           # 测试集序列

# Constructing Machine Learning Model (MLModel)
if ml_style == "forest":
    ml = RandomForestRegressor()
elif ml_style == "linear":
    ml = LinearRegression()
elif ml_style == "svr":
    ml = SVR()
elif ml_style == "sgd":
    ml = SGDRegressor()
else:
    raise TypeError("Unknown Type of ml_style!")

# Test and plot
start_time = datetime.datetime.now()
rmse_train, rmse_test, r2_train, r2_test, train_true, test_true, train_predict, test_predict = cal.eval_ml(
    ml, x_train, y_train, x_test, y_test, iv, way, fig_size)
print("{}:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(ml_style, rmse_train, rmse_test, r2_train, r2_test))
end_time = datetime.datetime.now()
run_time = end_time - start_time

if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_{}.png".format(ml_style)))
if save_np:
    np.save(osp.join(result_address, "train_true_{}.npy".format(ml_style)), train_true)
    np.save(osp.join(result_address, "test_true_{}.npy".format(ml_style)), test_true)
    np.save(osp.join(result_address, "train_predict_{}.npy".format(ml_style)), train_predict)
    np.save(osp.join(result_address, "test_predict_{}.npy".format(ml_style)), test_predict)
e_linear = test_true - test_predict
cal.plot_distribute(e_linear, 40, 4, x_name="e", fig_size=fig_size)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_{}_error_distribution.png".format(ml_style)))

# The output results of each model are appended to the file
if save_txt:
    info_txt_address = osp.join(result_address, "MLModel_result.txt")  # txt file address for saving parameter information
    info_df_address = osp.join(result_address, "MLModel_result.csv")  # csv file address for saving parameter information
    f = open(info_txt_address, 'a')
    if osp.getsize(info_txt_address) == 0:    # add the name of each feature in the first line of the text
        f.write("ml_style r2_test r2_train run_time l_x l_y\n")
    f.write(str(ml_style) + " ")
    f.write(str(r2_test) + " ")
    f.write(str(r2_train) + " ")
    f.write(str(run_time) + " ")
    f.write(str(l_x) + " ")
    f.write(str(l_y) + " ")

    f.write("\n")                           # Prepare for next running
    f.close()                               # close file

    info = np.loadtxt(info_txt_address, dtype=str)
    columns = info[0, :].tolist()
    values = info[1:, :]
    info_df = pd.DataFrame(values, columns=columns)
    info_df.to_csv(info_df_address)

# print()
# plt.show()
# print()
