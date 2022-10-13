"""
Testing ResGraphNet
"""
import datetime
import numpy as np
import pandas as pd
import torch
import os
import os.path as osp
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import func.cal as cal


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
l_x = 60                   # Data sequence length
l_y = 1                    # Label sequence length
lr = 0.0001                  # Learning rate
weight_decay = 5e-4
epochs = 4000
hidden_dim = 64
gnn_style = "ResGraphNet"
save_fig = True                  # Whether to save picture
save_txt = False                  # Whether to save txt
save_np = False                  # Whether to save np file
save_model = True               # Whether to save network model
ratio_train = 0.5               # Proportion of training datasets
fig_size = (12, 12)
ts_name_all = ["cli_dash", "HadCRUT5", "temp_month", "temp_year", "elect", "traffic", "sales"]
ts_name_folder = "HadCRUT5"    # Name of the folder where the data resides
ts_name = "HadCRUT5_global_full"       # Name of the selected time series

# para = sys.argv
# ts_name_folder, ts_name, epochs = para[1], para[2], int(para[3])

iv = 1                          # sampling interval, used for plotting curves
way = "mean"                    # The style of plot curves of real data and predict results

x_address = osp.join("../datasets", ts_name_folder, ts_name + ".npy")
x = np.load(x_address)
num = x.shape[0]                # The length of time series

result_address = osp.join("../result", ts_name, "ResGraphNet")
if not(osp.exists(result_address)):
    os.makedirs(result_address)

num_train = int(ratio_train * num)
data_train, data_test = x[:num_train], x[num_train:num]     # get training dataset and test dataset
data_test = np.array(data_test.tolist() + [data_test[-1]] * 9)

cal.compare_distribute(data_train, data_test, x_name="Temperature anomaly values", fig_size=fig_size)
if save_fig:
    plt.savefig(osp.join(result_address, "compare_distribute.png"))
cal.plot_temp(data_train, data_test, fig_size)
if save_fig:
    plt.savefig(osp.join(result_address, "temp_value.png"))

len_interp = l_y + 6
data_test_ = np.array(data_test[:-l_y].tolist() + data_test[-len_interp-l_y:-l_y].tolist() + data_test[-l_y:].tolist())

# Using Graph Neural network, prepare data information
x_train, y_train = cal.create_inout_sequences(data_train, l_x, l_y, style="arr")
x_test, y_test = cal.create_inout_sequences(data_test_, l_x, l_y, style="arr")

x_train = torch.from_numpy(x_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)
num_nodes = x_train.shape[0] + x_test.shape[0]
num_train = x_train.shape[0]

x = torch.cat((x_train, x_test), dim=0)
y = torch.cat((y_train, y_test), dim=0)

adm = cal.path_graph(num_nodes)
# adm = cal.ts_un(num_nodes, 6)
edge_index, edge_weight = cal.tran_adm_to_edge_index(adm)

train_index = torch.arange(num_train, dtype=torch.long)
test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
train_mask = cal.index_to_mask(train_index, num_nodes).to(device)
test_mask = cal.index_to_mask(test_index, num_nodes).to(device)

# Using ResGraphNet, predicting time series (The Proposed Network Model)
model = cal.GNNTime(l_x, hidden_dim, l_y, edge_weight, gnn_style, num_nodes).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
edge_index = edge_index.to(device)

start_time = datetime.datetime.now()
print("Running, {}".format(gnn_style))
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)
    output_train, y_train = output[train_mask], y[train_mask]
    train_loss = criterion(output_train[:, -1], y_train[:, -1])
    train_loss.backward()
    optimizer.step()

    model.eval()
    y_test_1 = y[test_mask][:-len_interp-l_y, :]
    y_test_2 = y[test_mask][-l_y:, :]
    y_test = torch.cat((y_test_1, y_test_2), dim=0)
    output_test = output[test_mask][:-len_interp, :]
    test_loss = criterion(output_test[:, -1], y_test[:, -1])

    train_true = y_train.detach().cpu().numpy()[:, -1]
    train_predict = output_train.detach().cpu().numpy()[:, -1]
    test_true = y_test.detach().cpu().numpy()[:, -1]
    test_predict = output_test.detach().cpu().numpy()[:, -1]

    r2_train = cal.get_r2_score(train_predict, train_true, axis=1)
    r2_test = cal.get_r2_score(test_predict, test_true, axis=1)

    if (epoch + 1) % 100 == 0:
        print("Epoch: {:05d}  Loss_Train: {:.5f}  Loss_Test: {:.5f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
              format(epoch + 1, train_loss.item(), test_loss.item(), r2_train, r2_test))

# predict and plot future time series
plot_predict = test_predict[-l_y:]
plot_true = test_true[-l_y:]
mse_plot = np.mean(np.square(plot_predict - plot_true))
print("mse_plot: {:.8f}".format(mse_plot))
cal.plot_spiral([plot_predict, plot_true[:3]], fig_size)           # predict results in the coming year
if save_fig:
    plt.savefig(osp.join(result_address, "future_predict_{}.png".format(ts_name)))

# calculate running time
end_time = datetime.datetime.now()
run_time = end_time - start_time              # The running time of program

# save model and numpy.file
if save_model:
    torch.save(model, osp.join(result_address, "{}.pkl".format(gnn_style)))
if save_np:
    np.save(osp.join(result_address, "train_true_{}.npy".format(gnn_style)), train_true)
    np.save(osp.join(result_address, "test_true_{}.npy".format(gnn_style)), test_true)
    np.save(osp.join(result_address, "train_predict_{}.npy".format(gnn_style)), train_predict)
    np.save(osp.join(result_address, "test_predict_{}.npy".format(gnn_style)), test_predict)

# plot the error and results
e_gnn = test_true - test_predict
cal.plot_distribute(e_gnn, 40, 4, x_name="e", fig_size=fig_size)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_" + gnn_style + "_error_distribution.png"))

cal.plot_result(train_true, test_true, train_predict, test_predict, iv, way, fig_size)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_" + gnn_style + ".png"))

# print indicators
rmse_train = cal.get_rmse(train_predict, train_true)
rmse_test = cal.get_rmse(test_predict, test_true)
r2_train = cal.get_r2_score(train_predict, train_true, axis=1)
r2_test = cal.get_r2_score(test_predict, test_true, axis=1)
print("{}: RMSE_Train={:.5f}  RMSE_Test={:.5f}  R2_Train={:.7f}  R2_Test={:.7f}".
      format(gnn_style, rmse_train, rmse_test, r2_train, r2_test))

# predict and plot future time series
mse_plot = np.mean(np.square(plot_predict - plot_true))
print("mse_plot: {:.8f}".format(mse_plot))
print("\nPredict: \n", np.round(plot_predict, 5), "\npredict annual mean: {:.5f}".format(np.mean(plot_predict)))
print("\nKnown: \n", np.round(plot_true, 5))

# The output results of each model are appended to the file
if save_txt:
    info_txt_address = osp.join(result_address, "ResGraphNet_result.txt")  # txt file address for saving parameter information
    info_df_address = osp.join(result_address, "ResGraphNet_result.csv")  # csv file address for saving parameter information
    f = open(info_txt_address, 'a')
    if osp.getsize(info_txt_address) == 0:    # add the name of each feature in the first line of the text
        f.write("gnn_style r2_test r2_train run_time l_x l_y hidden_dim lr epochs\n")
    f.write(str(gnn_style) + " ")
    f.write(str(r2_test) + " ")
    f.write(str(r2_train) + " ")
    f.write(str(run_time) + " ")
    f.write(str(l_x) + " ")
    f.write(str(l_y) + " ")
    f.write(str(hidden_dim) + " ")
    f.write(str(lr) + " ")
    f.write(str(epochs) + " ")

    f.write("\n")                           # Prepare for next running
    f.close()                               # close file

    info = np.loadtxt(info_txt_address, dtype=str)
    columns = info[0, :].tolist()
    values = info[1:, :]
    info_df = pd.DataFrame(values, columns=columns)
    info_df.to_csv(info_df_address)


print()
plt.show()
print()
