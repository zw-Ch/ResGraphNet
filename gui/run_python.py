import sys
import os
import os.path as osp
import matplotlib
import numpy as np
import pyqtgraph as pg
import torch
import datetime
import gui_func as gf
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets
from torch.utils.data import DataLoader
device = "cuda:1" if torch.cuda.is_available() else "cpu"


def get_model_(model):
    a = 0
    if model in ["ResGraphNet", "RESModel", "ARIMA", "SARIMAX"]:
        a = model
    elif model in ['forest', 'linear', 'svr', 'sgd']:
        a = "MLModel"
    elif model in ['GraphSage', 'UniMP', 'GCN', 'GIN']:
        a = "GNNModel"
    elif model in ["LSTM", "GRU"]:
        a = "RNNModel"
    return a


def get_folder(root):
    root_address_ = root.split('/')
    folder = "/"
    for i in range(len(root_address_) - 1):
        folder = osp.join(folder, root_address_[i])
    return folder


class OutputWidget(QWidget):
    def __init__(self):
        super(OutputWidget, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.result_text = QTextEdit()
        self.set_layout()

    def set_layout(self):
        vbox = QVBoxLayout()
        vbox.addWidget(self.result_text)
        self.setLayout(vbox)

    def run_model(self, para):
        ts_folder, ts_name, model_, model, root = para[0], para[1], para[3], para[4], para[5]
        epochs, ratio_train, l_x, l_y = para[2], float(para[6]), int(para[7]), int(para[8])
        model_name = model
        if epochs != ' ':
            epochs = int(epochs)
        x_address = osp.join(root, "datasets", ts_folder, ts_name + ".npy")
        x = np.load(x_address)
        num = x.shape[0]                        # The length of time series
        num_train = int(ratio_train * num)
        data_train, data_test = x[:num_train], x[num_train:num]  # get training dataset and test dataset
        if model_ == "ResGraphNet":
            len_interp = l_y + 6
            data_test_ = np.array(
                data_test[:-l_y].tolist() + data_test[-len_interp - l_y:-l_y].tolist() + data_test[-l_y:].tolist())
            # Using Graph Neural network, prepare data information
            x_train, y_train = gf.create_inout_sequences(data_train, l_x, l_y, style="arr")
            x_test, y_test = gf.create_inout_sequences(data_test_, l_x, l_y, style="arr")
            x_train = torch.from_numpy(x_train).float().to(device)
            x_test = torch.from_numpy(x_test).float().to(device)
            y_train = torch.from_numpy(y_train).float().to(device)
            y_test = torch.from_numpy(y_test).float().to(device)
            num_nodes = x_train.shape[0] + x_test.shape[0]
            num_train = x_train.shape[0]

            x = torch.cat((x_train, x_test), dim=0)
            y = torch.cat((y_train, y_test), dim=0)

            adm = gf.path_graph(num_nodes)
            edge_index, edge_weight = gf.tran_adm_to_edge_index(adm)

            train_index = torch.arange(num_train, dtype=torch.long)
            test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
            train_mask = gf.index_to_mask(train_index, num_nodes).to(device)
            test_mask = gf.index_to_mask(test_index, num_nodes).to(device)

            # Using ResGraphNet, predicting time series (The Proposed Network Model)
            model = gf.GNNTime(l_x, 64, l_y, edge_weight, "ResGraphNet", num_nodes).to(device)
            criterion = torch.nn.MSELoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
            edge_index = edge_index.to(device)

            start_time = datetime.datetime.now()
            pri = ""
            print("Running, {}".format("ResGraphNet"))
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                output = model(x, edge_index)
                output_train, y_train = output[train_mask], y[train_mask]
                train_loss = criterion(output_train[:, -1], y_train[:, -1])
                train_loss.backward()
                optimizer.step()
                QApplication.processEvents()

                model.eval()
                y_test_1 = y[test_mask][:-len_interp - l_y, :]
                y_test_2 = y[test_mask][-l_y:, :]
                y_test = torch.cat((y_test_1, y_test_2), dim=0)
                output_test = output[test_mask][:-len_interp, :]
                test_loss = criterion(output_test[:, -1], y_test[:, -1])

                train_true = y_train.detach().cpu().numpy()[:, -1]
                train_predict = output_train.detach().cpu().numpy()[:, -1]
                test_true = y_test.detach().cpu().numpy()[:, -1]
                test_predict = output_test.detach().cpu().numpy()[:, -1]

                r2_train = gf.get_r2_score(train_predict, train_true, axis=1)
                r2_test = gf.get_r2_score(test_predict, test_true, axis=1)

                if (epoch + 1) % 100 == 0:
                    pri_one = "Epoch: {:05d}  R2_Train: {:.7f}  R2_Test: {:.7f}\n".format(epoch + 1, r2_train, r2_test)
                    pri = pri + pri_one
                    print(pri_one)
                    self.result_text.setText(pri)
                self.result_text.moveCursor(QTextCursor.End)

        elif model_ == "MLModel":
            # Using machine learning to predict time series
            print("\nRunning, Machine Learning, {}".format(model_name))
            x_train, y_train = gf.create_inout_sequences(data_train, l_x, l_y, style="arr")
            x_test, y_test = gf.create_inout_sequences(data_test, l_x, l_y, style="arr")
            # Constructing Machine Learning Model (MLModel)
            if model == "forest":
                ml = RandomForestRegressor()
            elif model == "linear":
                ml = LinearRegression()
            elif model == "svr":
                ml = SVR()
            elif model == "sgd":
                ml = SGDRegressor()
            # Test and plot
            start_time = datetime.datetime.now()
            r2_train, r2_test, train_true, test_true, train_predict, test_predict = gf.eval_ml(
                ml, x_train, y_train, x_test, y_test)
            pri = ("{}:  R2_Train: {:.6f}  R2_Test: {:.6f}\n".format(model, r2_train, r2_test))
            print(pri)
            self.result_text.setText(pri)

        elif model_ == "GNNModel":
            # Using Graph Neural network, prepare data information
            x_train, y_train = gf.create_inout_sequences(data_train, l_x, l_y, style="arr")
            x_test, y_test = gf.create_inout_sequences(data_test, l_x, l_y, style="arr")
            x_train = torch.from_numpy(x_train).float().to(device)
            x_test = torch.from_numpy(x_test).float().to(device)
            y_train = torch.from_numpy(y_train).float().to(device)
            y_test = torch.from_numpy(y_test).float().to(device)
            num_nodes = x_train.shape[0] + x_test.shape[0]

            x = torch.cat((x_train, x_test), dim=0)
            y = torch.cat((y_train, y_test), dim=0)

            adm = gf.path_graph(num_nodes)
            edge_index, edge_weight = gf.tran_adm_to_edge_index(adm)

            train_index = torch.arange(num_train, dtype=torch.long)
            test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
            train_mask = gf.index_to_mask(train_index, num_nodes).to(device)
            test_mask = gf.index_to_mask(test_index, num_nodes).to(device)

            # Using GNN Model, predicting time series
            model = gf.GNNTime(l_x, 64, l_y, edge_weight, model, num_nodes).to(device)
            criterion = torch.nn.MSELoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
            edge_index = edge_index.to(device)

            start_time = datetime.datetime.now()
            print("Running, {}".format(model_name))
            pri = ""
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                output = model(x, edge_index)
                output_train, y_train = output[train_mask], y[train_mask]
                train_loss = criterion(output_train[:, -1], y_train[:, -1])
                train_loss.backward()
                optimizer.step()
                QApplication.processEvents()

                model.eval()
                output_test, y_test = output[test_mask], y[test_mask]
                test_loss = criterion(output_test[:, -1], y_test[:, -1])

                train_true = y_train.detach().cpu().numpy()[:, -1]
                train_predict = output_train.detach().cpu().numpy()[:, -1]
                test_true = y_test.detach().cpu().numpy()[:, -1]
                test_predict = output_test.detach().cpu().numpy()[:, -1]

                r2_train = gf.get_r2_score(train_predict, train_true, axis=1)
                r2_test = gf.get_r2_score(test_predict, test_true, axis=1)

                if (epoch + 1) % 100 == 0:
                    pri_one = "Epoch: {:05d}  R2_Train: {:.7f}  R2_Test: {:.7f}\n".format(epoch + 1, r2_train, r2_test)
                    pri = pri + pri_one
                    print(pri_one)
                    self.result_text.setText(pri)

        elif model_ == "RESModel":
            hidden_dim = 64
            batch_size = 32
            x_train, y_train = gf.create_inout_sequences(data_train, l_x, l_y, style="arr")
            x_test, y_test = gf.create_inout_sequences(data_test, l_x, l_y, style="arr")
            train_dataset = gf.MyData(x_train, y_train)
            test_dataset = gf.MyData(x_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model = gf.RESModel(hidden_dim, l_y, l_x).to(device)
            criterion = torch.nn.MSELoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

            print("\nRunning, RESModel")
            pri = ""
            for epoch in range(epochs):
                loss_train_all, loss_test_all = 0, 0
                train_true, train_predict, test_true, test_predict = [], [], [], []
                for idx, (x_train, y_train) in enumerate(train_loader):
                    x_train, y_train = x_train.to(device), y_train.to(device)
                    optimizer.zero_grad()
                    output_train = model(x_train)
                    loss_train = criterion(output_train, y_train)
                    loss_train.backward()
                    optimizer.step()
                    loss_train_all = loss_train_all + loss_train.item()
                    QApplication.processEvents()

                    train_predict_one = output_train.detach().cpu().numpy()[:, -1]
                    train_true_one = y_train.detach().cpu().numpy()[:, -1]
                    if idx == 0:
                        train_true = train_true_one
                        train_predict = train_predict_one
                    else:
                        train_true = np.concatenate((train_true, train_true_one), axis=0)
                        train_predict = np.concatenate((train_predict, train_predict_one), axis=0)

                for idx, (x_test, y_test) in enumerate(test_loader):
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    output_test = model(x_test)
                    loss_test = criterion(output_test, y_test)
                    loss_test_all = loss_test_all + loss_test.item()
                    QApplication.processEvents()

                    test_predict_one = output_test.detach().cpu().numpy()[:, -1]
                    test_true_one = y_test.detach().cpu().numpy()[:, -1]
                    if idx == 0:
                        test_true = test_true_one
                        test_predict = test_predict_one
                    else:
                        test_true = np.concatenate((test_true, test_true_one), axis=0)
                        test_predict = np.concatenate((test_predict, test_predict_one), axis=0)

                r2_train = gf.get_r2_score(train_predict, train_true, axis=1)
                r2_test = gf.get_r2_score(test_predict, test_true, axis=1)
                pri_one = "Epoch: {:04d}  R2_Train: {:.7f}  R2_Test: {:.7f}\n".format(epoch, r2_train, r2_test)
                pri = pri + pri_one
                print(pri_one)
                self.result_text.setText(pri)
                self.result_text.moveCursor(QTextCursor.End)

        elif model_ == "RNNModel":
            hidden_dim = 64
            batch_size = 32
            print("\nRunning, RNNModel, {}".format(model_name))
            x_train, y_train = gf.create_inout_sequences(data_train, l_x, l_y, style="arr")
            x_test, y_test = gf.create_inout_sequences(data_test, l_x, l_y, style="arr")
            train_dataset = gf.MyData(x_train, y_train)
            test_dataset = gf.MyData(x_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            model = gf.RNNTime(model_name, 1, hidden_dim, l_y, l_x, 1).to(device)
            criterion = torch.nn.MSELoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

            pri = ""
            for epoch in range(epochs):
                loss_train_all, loss_test_all = 0, 0
                train_true, train_predict, test_true, test_predict = [], [], [], []
                for idx, (x_train, y_train) in enumerate(train_loader):
                    x_train, y_train = x_train.to(device), y_train.to(device)
                    optimizer.zero_grad()
                    output_train = model(x_train)
                    loss_train = criterion(output_train, y_train)
                    loss_train.backward()
                    optimizer.step()
                    loss_train_all = loss_train_all + loss_train.item()
                    QApplication.processEvents()

                    train_predict_one = output_train.detach().cpu().numpy()
                    train_true_one = y_train.detach().cpu().numpy()
                    if idx == 0:
                        train_true = train_true_one
                        train_predict = train_predict_one
                    else:
                        train_true = np.concatenate((train_true, train_true_one), axis=0)
                        train_predict = np.concatenate((train_predict, train_predict_one), axis=0)

                for idx, (x_test, y_test) in enumerate(test_loader):
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    output_test = model(x_test)
                    loss_test = criterion(output_test, y_test)
                    loss_test_all = loss_test_all + loss_test.item()
                    QApplication.processEvents()

                    test_predict_one = output_test.detach().cpu().numpy()
                    test_true_one = y_test.detach().cpu().numpy()
                    if idx == 0:
                        test_true = test_true_one
                        test_predict = test_predict_one
                    else:
                        test_true = np.concatenate((test_true, test_true_one), axis=0)
                        test_predict = np.concatenate((test_predict, test_predict_one), axis=0)

                train_true = train_true[:, -1]
                train_predict = train_predict[:, -1]
                test_true = test_true[:, -1]
                test_predict = test_predict[:, -1]
                r2_train = gf.get_r2_score(train_predict, train_true, axis=1)
                r2_test = gf.get_r2_score(test_predict, test_true, axis=1)
                pri_one = "Epoch: {:04d}  R2_Train: {:.7f}  R2_Test: {:.7f}\n".format(epoch, r2_train, r2_test)
                print(pri_one)
                pri = pri + pri_one
                self.result_text.setText(pri)
                self.result_text.moveCursor(QTextCursor.End)
        # elif model_ in ["ARIMA", "SARIMAX"]:
        #     print("\nRunning, {}".format(model_name))
        #     train_true, test_true = x[:num_train], x[num_train:num]
        #     if model_name == "ARIMA":
        #         model = ARIMA(train_true, order=(1, 2, 2))
        #         model_train = model.fit()
        #         model_test = model_train.apply(test_true)
        #         train_predict = model_train.predict()
        #         test_predict = model_test.predict()
        #     elif model_name == "SARIMAX":
        #         model = sm.tsa.SARIMAX(train_true, order=(1, 0, 0), trend='c')
        #         model_train = model.fit()
        #         model_test = model_train.apply(test_true)
        #         train_predict = model_train.predict()
        #         test_predict = model_test.predict()
        #     r2_train = gf.get_r2_score(train_predict, train_true, axis=1)
        #     r2_test = gf.get_r2_score(test_predict, test_true, axis=1)
        #     print("{}:  r2_train: {:.5f}  r2_test: {:.5f}".format(model_name, r2_train, r2_test))
        result_address = osp.join(root, "result", ts_name, model_)
        np.save(osp.join(result_address, "train_true_{}.npy".format(model_name)), train_true)
        np.save(osp.join(result_address, "test_true_{}.npy".format(model_name)), test_true)
        np.save(osp.join(result_address, "train_predict_{}.npy".format(model_name)), train_predict)
        np.save(osp.join(result_address, "test_predict_{}.npy".format(model_name)), test_predict)


class TableWidget(QWidget):
    def __init__(self):
        super(TableWidget, self).__init__()
        self.ts_name_list = [
            'HadCRUT5_global', 'HadCRUT5_northern', 'HadCRUT5_southern', 'ERSSTv4', 'ERSSTv3b', 'NOAA',
            'Berkeley_Earth', 'HadSST3', 'ERA5_Global', 'ERA5_European', 'Electricity', 'Traffic', 'Sales']
        self.model_list = [
            'ResGraphNet', 'RESModel', 'GraphSage', 'UniMP', 'GCN', 'GIN', 'LSTM', 'GRU', 'forest', 'linear', 'svr', 'sgd',
            'ARIMA', 'SARIMAX']
        self.table = QStandardItemModel(len(self.ts_name_list), len(self.model_list))
        self.get_labels()

        self.table_view = QTableView()
        self.table_view.setModel(self.table)

        self.fill_btn, self.delete_btn, self.fill_all_btn, self.delete_all_btn = self.get_fill_delete_button()

        self.set_layout()

    def get_labels(self):
        self.table.setVerticalHeaderLabels(self.ts_name_list)
        self.table.setHorizontalHeaderLabels(self.model_list)

    def get_fill_delete_button(self):
        fill_btn = QPushButton('Fill')
        delete_btn = QPushButton('Delete')
        fill_all_btn = QPushButton('Fill All')
        delete_all_btn = QPushButton('Delete All')
        return fill_btn, delete_btn, fill_all_btn, delete_all_btn

    def set_layout(self):
        table_box = QVBoxLayout(self)
        table_box.addWidget(self.table_view)
        btn_box = QHBoxLayout(self)
        btn_box.addWidget(self.fill_btn)
        btn_box.addWidget(self.delete_btn)
        btn_box.addWidget(self.fill_all_btn)
        btn_box.addWidget(self.delete_all_btn)
        table_box.addLayout(btn_box)
        self.layout()

    def fill(self, ts_name, model, folder):
        ts_name_idx, model_idx = self.get_loc_in_table(ts_name, model)
        r2 = self.get_r2_test_result(ts_name, model, folder)
        item = QStandardItem(r2)
        self.table.setItem(ts_name_idx, model_idx, item)

    def fill_all(self, folder):
        for ts_name in self.ts_name_list:
            for model in self.model_list:
                self.fill(ts_name, model, folder)

    def delete_all(self):
        for ts_name in self.ts_name_list:
            for model in self.model_list:
                self.delete(ts_name, model)

    def delete(self, ts_name, model):
        ts_name_idx, model_idx = self.get_loc_in_table(ts_name, model)
        item = QStandardItem("")
        self.table.setItem(ts_name_idx, model_idx, item)

    def get_loc_in_table(self, ts_name, model):
        ts_name_idx = np.argwhere(ts_name == np.array(self.ts_name_list)).reshape(-1)[0]
        model_idx = np.argwhere(model == np.array(self.model_list)).reshape(-1)[0]
        return ts_name_idx, model_idx

    def get_r2_test_result(self, ts_name, model, folder):
        model_ = get_model_(model)
        result_address = osp.join(folder, "result", ts_name, model_)
        true_address = osp.join(result_address, "test_true_{}.npy".format(model))
        predict_address = osp.join(result_address, "test_predict_{}.npy".format(model))
        try:
            true, predict = np.load(true_address), np.load(predict_address)
            r2 = gf.get_r2_score(predict, true, axis=1)
            r2 = str(round(r2, 4))
        except:
            # self.setStatusTip("{} in {} haven't been calculated")
            r2 = "None"
        return r2


class VisualWidget(QWidget):
    def __init__(self):
        super(VisualWidget, self).__init__()
        pg.setConfigOption('background', '#f0f0f0')
        pg.setConfigOption('foreground', 'd')
        self.graph_wg_ori = pg.PlotWidget()
        self.graph_wg_result = pg.PlotWidget()
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.graph_wg_ori)
        vbox.addWidget(self.graph_wg_result)

    def plot_ori(self, ts_address):
        x = np.load(ts_address + ".npy")
        self.graph_wg_ori.clear()
        pen = pg.mkPen(color='k', width=2)
        self.graph_wg_ori.plot(x, pen=pen)
        ts_address_ = ts_address.split('/')
        ts_name = ts_address_[-2]
        self.set_style(ts_name)

    def plot_result(self, folder, ts_name, model_, model):
        result_address = osp.join(folder, "result", ts_name, model_)
        train_true_address = osp.join(result_address, "train_true_{}.npy".format(model))
        train_predict_address = osp.join(result_address, "train_predict_{}.npy".format(model))
        test_true_address = osp.join(result_address, "test_true_{}.npy".format(model))
        test_predict_address = osp.join(result_address, "test_predict_{}.npy".format(model))

        train_true = np.load(train_true_address)
        train_predict = np.load(train_predict_address)
        test_true = np.load(test_true_address)
        test_predict = np.load(test_predict_address)
        train_range = np.arange(train_true.shape[0])
        test_range = np.arange(train_true.shape[0], train_true.shape[0] + test_true.shape[0])

        self.graph_wg_result.clear()
        self.graph_wg_result.addLegend()
        pen = pg.mkPen(color='m', width=1)
        line1 = self.graph_wg_result.plot(train_range, train_true, pen=pen, name="Train True")
        pen = pg.mkPen(color='b', width=1)
        line2 = self.graph_wg_result.plot(train_range, train_predict, pen=pen, name="Train Predict")
        pen = pg.mkPen(color='g', width=1)
        line3 = self.graph_wg_result.plot(test_range, test_true, pen=pen, name="Test True")
        pen = pg.mkPen(color='r', width=1)
        line4 = self.graph_wg_result.plot(test_range, test_predict, pen=pen, name="Test Predict")

        line1.setAlpha(0.7, False)
        line2.setAlpha(0.99, False)
        line3.setAlpha(0.7, False)
        line4.setAlpha(0.99, False)

    def plot_result_one(self, style, folder, ts_name, model):
        model_ = get_model_(model)
        self.graph_wg_result.clear()
        result_address = osp.join(folder, "result", ts_name, model_)
        if style == "train":
            true_address = osp.join(result_address, "train_true_{}.npy".format(model))
            predict_address = osp.join(result_address, "train_predict_{}.npy".format(model))
        elif style == "test":
            true_address = osp.join(result_address, "test_true_{}.npy".format(model))
            predict_address = osp.join(result_address, "test_predict_{}.npy".format(model))
        else:
            raise TypeError("Unknown type of style!")
        true, predict = np.load(true_address), np.load(predict_address)
        ran = np.arange(true.shape[0])
        self.graph_wg_result.clear()
        self.graph_wg_result.addLegend()
        if style == "train":
            pen = pg.mkPen(color='m', width=1)
            line1 = self.graph_wg_result.plot(ran, true, pen=pen, name="Train True")
            pen = pg.mkPen(color='b', width=1)
            line2 = self.graph_wg_result.plot(ran, predict, pen=pen, name="Train Predict")
        elif style == "test":
            pen = pg.mkPen(color='g', width=1)
            line1 = self.graph_wg_result.plot(ran, true, pen=pen, name="Test True")
            pen = pg.mkPen(color='r', width=1)
            line2 = self.graph_wg_result.plot(ran, predict, pen=pen, name="Test Predict")
        line1.setAlpha(0.7, False)
        line2.setAlpha(0.99, False)

    def set_style(self, ts_name):
        if ts_name == 'HadCRUT5':
            self.graph_wg_ori.setLabel('left', 'Anomaly (deg C)')
            # self.graph_wg_ori.setLabel('bottom', 'year')
            self.graph_wg_result.setLabel('left', 'Anomaly (deg C)')
            # self.graph_wg_result.setLabel('bottom', 'year')
        elif ts_name == 'cli_dash':
            self.graph_wg_ori.setLabel('left', 'Anomaly')
            self.graph_wg_result.setLabel('left', 'Anomaly')
        elif ts_name == 'temp_month':
            self.graph_wg_ori.setLabel('left', 'Temperature')
            self.graph_wg_result.setLabel('left', 'Temperature')
        elif ts_name == 'elect':
            self.graph_wg_ori.setLabel('left', 'Electricity')
            self.graph_wg_result.setLabel('left', 'Electricity')
        elif ts_name == 'traffic':
            self.graph_wg_ori.setLabel('left', 'Traffic')
            self.graph_wg_result.setLabel('left', 'Traffic')
        elif ts_name == 'sales':
            self.graph_wg_ori.setLabel('left', 'Sales')
            self.graph_wg_result.setLabel('left', 'Sales')
        else:
            raise TypeError("Unknown Type of folder!")
        self.graph_wg_ori.plotItem.setTitle('Original Data')
        self.graph_wg_result.plotItem.setTitle('Predicted Results')


class TreeWidget(QTreeWidget):
    def __init__(self):
        super(TreeWidget, self).__init__()
        self.setColumnCount(6)
        self.setColumnWidth(2, 200)
        self.setColumnWidth(3, 250)
        self.setColumnWidth(4, 280)
        self.get_root_node()
        self.get_child1_nodes()
        self.get_child2_nodes()
        self.get_child3_nodes()
        self.get_child4_nodes()
        self.addTopLevelItem(self.root)
        self.expandAll()

    def get_root_node(self):
        self.root = QTreeWidgetItem(self)
        self.root.setText(0, 'root')
        self.root.setIcon(0, QIcon("文件夹.svg"))

    def get_child1_nodes(self):
        self.datasets = QTreeWidgetItem(self.root)
        self.datasets.setText(1, 'datasets')
        self.datasets.setIcon(1, QIcon("文件夹.svg"))
        self.func = QTreeWidgetItem(self.root)
        self.func.setText(1, 'func')
        self.func.setIcon(1, QIcon("文件夹.svg"))
        self.graph = QTreeWidgetItem(self.root)
        self.graph.setText(1, 'graph')
        self.graph.setIcon(1, QIcon("文件夹.svg"))
        self.plot = QTreeWidgetItem(self.root)
        self.plot.setText(1, 'plot')
        self.plot.setIcon(1, QIcon("文件夹.svg"))
        self.result = QTreeWidgetItem(self.root)
        self.result.setText(1, 'result')
        self.result.setIcon(1, QIcon("文件夹.svg"))
        self.run = QTreeWidgetItem(self.root)
        self.run.setText(1, 'run')
        self.run.setIcon(1, QIcon("文件夹.svg"))

    def get_child2_nodes(self):
        self.cli_dast = QTreeWidgetItem(self.datasets)
        self.cli_dast.setText(2, 'cli_dast')
        self.cli_dast.setIcon(2, QIcon("文件夹.svg"))
        self.elect = QTreeWidgetItem(self.datasets)
        self.elect.setText(2, 'elect')
        self.elect.setIcon(2, QIcon("文件夹.svg"))
        self.HadCRUT5 = QTreeWidgetItem(self.datasets)
        self.HadCRUT5.setText(2, 'HadCRUT5')
        self.HadCRUT5.setIcon(2, QIcon("文件夹.svg"))
        self.sales = QTreeWidgetItem(self.datasets)
        self.sales.setText(2, 'sales')
        self.sales.setIcon(2, QIcon("文件夹.svg"))
        self.temp_month = QTreeWidgetItem(self.datasets)
        self.temp_month.setText(2, 'temp_month')
        self.temp_month.setIcon(2, QIcon("文件夹.svg"))
        self.traffic = QTreeWidgetItem(self.datasets)
        self.traffic.setText(2, 'traffic')
        self.traffic.setIcon(2, QIcon("文件夹.svg"))

        self.init = QTreeWidgetItem(self.func)
        self.init.setText(2, '__init__.py')
        self.init.setIcon(2, QIcon("python.svg"))
        self.cal = QTreeWidgetItem(self.func)
        self.cal.setText(2, 'cal.py')
        self.cal.setIcon(2, QIcon("python.svg"))

        self.run_ARIMA = QTreeWidgetItem(self.run)
        self.run_ARIMA.setText(2, 'run_ARIMA.py')
        self.run_ARIMA.setIcon(2, QIcon("python.svg"))
        self.run_GNNModel = QTreeWidgetItem(self.run)
        self.run_GNNModel.setText(2, 'run_GNNModel.py')
        self.run_GNNModel.setIcon(2, QIcon("python.svg"))
        self.run_MLModel = QTreeWidgetItem(self.run)
        self.run_MLModel.setText(2, 'run_MLModel.py')
        self.run_MLModel.setIcon(2, QIcon("python.svg"))
        self.run_ResGraphNet = QTreeWidgetItem(self.run)
        self.run_ResGraphNet.setText(2, 'run_ResGraphNet.py')
        self.run_ResGraphNet.setIcon(2, QIcon("python.svg"))
        self.run_RESModel = QTreeWidgetItem(self.run)
        self.run_RESModel.setText(2, 'run_RESModel.py')
        self.run_RESModel.setIcon(2, QIcon("python.svg"))
        self.run_RNNModel = QTreeWidgetItem(self.run)
        self.run_RNNModel.setText(2, 'run_RNNModel.py')
        self.run_RNNModel.setIcon(2, QIcon("python.svg"))
        self.run_SARIMAX = QTreeWidgetItem(self.run)
        self.run_SARIMAX.setText(2, 'run_SARIMAX.py')
        self.run_SARIMAX.setIcon(2, QIcon("python.svg"))

        self.HadCRUT5_global = QTreeWidgetItem(self.result)
        self.HadCRUT5_global.setText(2, 'HadCRUT5_global')
        self.HadCRUT5_global.setIcon(2, QIcon("文件夹.svg"))
        self.Berkeley_Earth = QTreeWidgetItem(self.result)
        self.Berkeley_Earth.setText(2, 'Berkeley_Earth')
        self.Berkeley_Earth.setIcon(2, QIcon("文件夹.svg"))
        self.ERA5_European = QTreeWidgetItem(self.result)
        self.ERA5_European.setText(2, 'ERA5_European')
        self.ERA5_European.setIcon(2, QIcon("文件夹.svg"))

    def get_child3_nodes(self):
        self.HadCRUT5_global_npy = QTreeWidgetItem(self.HadCRUT5)
        self.HadCRUT5_global_npy.setText(3, 'HadCRUT5_global.npy')
        self.HadCRUT5_global_npy.setIcon(3, QIcon("file.svg"))
        self.HadCRUT5_northern_npy = QTreeWidgetItem(self.HadCRUT5)
        self.HadCRUT5_northern_npy.setText(3, 'HadCRUT5_northern.npy')
        self.HadCRUT5_northern_npy.setIcon(3, QIcon("file.svg"))
        self.HadCRUT5_southern_npy = QTreeWidgetItem(self.HadCRUT5)
        self.HadCRUT5_southern_npy.setText(3, 'HadCRUT5_southern.npy')
        self.HadCRUT5_southern_npy.setIcon(3, QIcon("file.svg"))

        self.ARIMA = QTreeWidgetItem(self.HadCRUT5_global)
        self.ARIMA.setText(3, 'ARIMA')
        self.ARIMA.setIcon(3, QIcon("文件夹.svg"))
        self.GNNModel = QTreeWidgetItem(self.HadCRUT5_global)
        self.GNNModel.setText(3, 'GNNModel')
        self.GNNModel.setIcon(3, QIcon("文件夹.svg"))
        self.MLModel = QTreeWidgetItem(self.HadCRUT5_global)
        self.MLModel.setText(3, 'MLModel')
        self.MLModel.setIcon(3, QIcon("文件夹.svg"))
        self.ResGraphNet = QTreeWidgetItem(self.HadCRUT5_global)
        self.ResGraphNet.setText(3, 'ResGraphNet')
        self.ResGraphNet.setIcon(3, QIcon("文件夹.svg"))
        self.RESModel = QTreeWidgetItem(self.HadCRUT5_global)
        self.RESModel.setText(3, 'RESModel')
        self.RESModel.setIcon(3, QIcon("文件夹.svg"))
        self.RNNModel = QTreeWidgetItem(self.HadCRUT5_global)
        self.RNNModel.setText(3, 'RNNModel')
        self.RNNModel.setIcon(3, QIcon("文件夹.svg"))
        self.SARIMAX = QTreeWidgetItem(self.HadCRUT5_global)
        self.SARIMAX.setText(3, 'SARIMAX')
        self.SARIMAX.setIcon(3, QIcon("文件夹.svg"))

    def get_child4_nodes(self):
        self.test_predict = QTreeWidgetItem(self.ResGraphNet)
        self.test_predict.setText(4, 'test_predict_ResGraphNet.npy')
        self.test_predict.setIcon(4, QIcon("file.svg"))
        self.test_true = QTreeWidgetItem(self.ResGraphNet)
        self.test_true.setText(4, 'test_true_ResGraphNet.npy')
        self.test_true.setIcon(4, QIcon("file.svg"))
        self.train_predict = QTreeWidgetItem(self.ResGraphNet)
        self.train_predict.setText(4, 'train_predict_ResGraphNet.npy')
        self.train_predict.setIcon(4, QIcon("file.svg"))
        self.train_true = QTreeWidgetItem(self.ResGraphNet)
        self.train_true.setText(4, 'train_true_ResGraphNet.npy')
        self.train_true.setIcon(4, QIcon("file.svg"))


class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()
        self.init()

    def init(self):
        menu_bar = QMenuBar(self)
        self.vis_wg = VisualWidget()
        file_menu = menu_bar.addMenu('File')

        self.ts_name = QLabel('ts name')
        self.ts_folder = QLabel('ts folder')
        self.epochs = QLabel('epochs')
        self.model = QLabel('model')
        self.root = QLabel('root')
        self.ts_address = QLabel('ts address')

        self.model_cb = self.get_model_cb()
        self.model_cb.currentTextChanged.connect(self.change_epochs)
        self.ts_folder_cb = self.get_ts_folder_cb()
        self.ts_name_cb = self.get_ts_name_cb()
        self.ts_folder_cb.currentTextChanged.connect(self.change_ts_name)
        self.epochs_edit = QLineEdit('1000')
        self.root_edit = QLineEdit('/home/chenziwei2021/py_envs_pc/paper/ResGraphNet')
        self.datasets_address = osp.join(self.root_edit.text(), "datasets")
        self.folder = get_folder(self.datasets_address)
        self.ts_address_edit = QLineEdit()
        self.ts_address_button = self.get_ts_address_button()
        self.model_button = self.get_model_button()
        self.lx_ql = QLabel('lx')
        self.lx_edit = QLineEdit('60')
        self.ly_ql = QLabel('ly')
        self.ly_edit = QLineEdit('1')
        self.ratio_train_ql = QLabel("ratio train")
        self.ratio_train_edit = QLineEdit("0.5")

        self.tab = TableWidget()
        self.tree = TreeWidget()
        self.output = OutputWidget()

        self.tab.fill_btn.clicked.connect(
            lambda: self.tab.fill(self.ts_name_cb.currentText(), self.model_cb.currentText(), self.folder))
        self.tab.delete_btn.clicked.connect(
            lambda: self.tab.delete(self.ts_name_cb.currentText(), self.model_cb.currentText()))
        self.tab.fill_all_btn.clicked.connect(
            lambda: self.tab.fill_all(self.folder))
        self.tab.delete_all_btn.clicked.connect(
            lambda: self.tab.delete_all())
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.vis_wg)
        self.stack.addWidget(self.tab)
        self.stack.addWidget(self.tree)
        self.stack.addWidget(self.output)

        self.ts_address_edit.textChanged.connect(self.vis_wg.plot_ori)
        self.set_layout()

        file_menu.addAction(self.get_exit_act())
        file_menu.addAction(self.get_run_act())
        result_menu = menu_bar.addMenu("Display")
        result_menu.addAction(self.get_vis_stack())
        result_menu.addAction(self.get_tab_stack())
        result_menu.addAction(self.get_tree_stack())
        result_menu.addAction(self.get_output_stack())

        curve_menu = menu_bar.addMenu('Curve')
        curve_menu.addAction(self.get_plot_train_act())
        curve_menu.addAction(self.get_plot_test_act())
        curve_menu.addAction(self.get_plot_all_act())

        table_menu = menu_bar.addMenu('Table')
        table_menu.addAction(self.get_table_fill_act())
        table_menu.addAction(self.get_table_delete_act())

        self.setGeometry(100, 100, 1600, 800)
        self.setWindowTitle('Drawing ResGraphNet')
        self.center()
        self.show()

    def set_layout(self):
        grid_1 = QGridLayout()
        grid_1.addWidget(self.ts_folder, 1, 0)
        grid_1.addWidget(self.ts_folder_cb, 1, 1)
        grid_1.addWidget(self.ts_name, 2, 0)
        grid_1.addWidget(self.ts_name_cb, 2, 1)
        grid_1.addWidget(self.root, 3, 0)
        grid_1.addWidget(self.root_edit, 3, 1)
        grid_1.addWidget(self.ts_address, 4, 0)
        grid_1.addWidget(self.ts_address_edit, 4, 1)
        grid_1.addWidget(self.ts_address_button, 4, 2)
        grid_2 = QGridLayout()
        grid_2.addWidget(self.epochs, 0, 0)
        grid_2.addWidget(self.epochs_edit, 0, 1)
        grid_2.addWidget(self.model, 0, 2)
        grid_2.addWidget(self.model_cb, 0, 3)
        grid_2.addWidget(self.model_button, 0, 4)
        grid_2.addWidget(self.lx_ql, 2, 0)
        grid_2.addWidget(self.lx_edit, 2, 1)
        grid_2.addWidget(self.ly_ql, 2, 2)
        grid_2.addWidget(self.ly_edit, 2, 3)
        grid_2.addWidget(self.ratio_train_ql, 3, 0)
        grid_2.addWidget(self.ratio_train_edit, 3, 1)
        left_box = QVBoxLayout()
        left_box.addLayout(grid_1)
        left_box.addLayout(grid_2)
        all_box = QHBoxLayout()
        all_box.addLayout(left_box)
        all_box.addWidget(self.stack)
        self.setLayout(all_box)

    def get_exit_act(self):
        exit_act = QAction(QIcon('toggle2-off.svg'), 'Exit', self)
        exit_act.setShortcut('Ctrl+w')
        exit_act.triggered.connect(qApp.quit)
        return exit_act

    def get_run_act(self):
        run_act = QAction(QIcon('run.svg'), 'run', self)
        run_act.setShortcut('Ctrl+q')
        # run_act.triggered.connect(self.run_program)
        self.vis_wg.graph_wg_result.clear()
        QApplication.processEvents()
        run_act.triggered.connect(lambda: self.output.run_model(self.get_para()))
        run_act.triggered.connect(
            lambda: self.vis_wg.plot_result(self.folder, self.ts_name_cb.currentText(),
                                            get_model_(self.model_cb.currentText()), self.model_cb.currentText()))
        return run_act

    def get_plot_train_act(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        plot_train_act = QAction('Plot Train', self)
        plot_train_act.setShortcut('Ctrl+z')
        plot_train_act.setToolTip('Only Plot True and Predict of Training Dataset')
        plot_train_act.triggered.connect(lambda: self.vis_wg.plot_result_one(
            "train", self.folder, self.ts_name_cb.currentText(), self.model_cb.currentText()))
        return plot_train_act

    def get_plot_test_act(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        plot_test_act = QAction('Plot Test', self)
        plot_test_act.setShortcut('Ctrl+x')
        plot_test_act.setToolTip('Only Plot True and Predict of Test Dataset')
        plot_test_act.triggered.connect(lambda: self.vis_wg.plot_result_one(
            "test", self.folder, self.ts_name_cb.currentText(), self.model_cb.currentText()))
        return plot_test_act

    def get_plot_all_act(self):
        plot_all_act = QAction('Plot All', self)
        plot_all_act.setShortcut('Ctrl+c')
        plot_all_act.triggered.connect(
            lambda: self.vis_wg.plot_result(self.folder, self.ts_name_cb.currentText(),
                                            get_model_(self.model_cb.currentText()), self.model_cb.currentText()))
        return plot_all_act

    def get_table_fill_act(self):
        table_fill_act = QAction('Fill', self)
        table_fill_act.setShortcut('Ctrl+f')
        table_fill_act.triggered.connect(
            lambda: self.tab.fill(self.ts_name_cb.currentText(), self.model_cb.currentText(), self.folder))
        return table_fill_act

    def get_table_delete_act(self):
        table_delete_act = QAction('Delete', self)
        table_delete_act.setShortcut('Ctrl+d')
        table_delete_act.triggered.connect(
            lambda: self.tab.delete(self.ts_name_cb.currentText(), self.model_cb.currentText()))
        return table_delete_act

    def get_vis_stack(self):
        vis_stack = QAction('Curve', self)
        vis_stack.setShortcut('Ctrl+e')
        vis_stack.triggered.connect(lambda: self.display_stack(0))
        return vis_stack

    def get_tab_stack(self):
        tab_stack = QAction('Table', self)
        tab_stack.setShortcut('Ctrl+r')
        tab_stack.triggered.connect(lambda: self.display_stack(1))
        return tab_stack

    def get_tree_stack(self):
        tree_stack = QAction('Tree', self)
        tree_stack.setShortcut('Ctrl+t')
        tree_stack.triggered.connect(lambda: self.display_stack(2))
        return tree_stack

    def get_output_stack(self):
        output_stack = QAction('Output', self)
        output_stack.setShortcut('Ctrl+y')
        output_stack.triggered.connect(lambda: self.display_stack(3))
        return output_stack

    def display_stack(self, i):
        self.stack.setCurrentIndex(i)

    def run_program(self):
        run_folder = osp.join(self.folder, "run")
        ts_folder = self.ts_folder_cb.currentText()
        ts_name = self.ts_name_cb.currentText()
        model = self.model_cb.currentText()
        epochs = self.epochs_edit.text()
        os.chdir(run_folder)
        if model == 'ResGraphNet':
            run_address = osp.join(run_folder, "run_ResGraphNet.py")
            self.model_ = model
            command = "python {} {} {} {}".format(run_address, ts_folder, ts_name, epochs)
        elif model == "RESModel":
            run_address = osp.join(run_folder, "run_RESModel.py")
            self.model_ = model
            command = "python {} {} {} {}".format(run_address, ts_folder, ts_name, epochs)
        elif model in ['forest', 'linear', 'svr', 'sgd']:
            run_address = osp.join(run_folder, "run_MLModel.py")
            self.model_ = "MLModel"
            ml_style = model
            command = 'python {} {} {} {}'.format(run_address, ts_folder, ts_name, ml_style)
        elif model in ['GraphSage', 'UniMP', 'GCN', 'GIN']:
            run_address = osp.join(run_folder, "run_GNNModel.py")
            self.model_ = "GNNModel"
            gnn_style = model
            command = "python {} {} {} {} {}".format(run_address, ts_folder, ts_name, gnn_style, epochs)
        elif model in ["LSTM", "GRU"]:
            run_address = osp.join(run_folder, "run_RNNModel.py")
            self.model_ = "RNNModel"
            rnn_style = model
            command = "python {} {} {} {} {}".format(run_address, ts_folder, ts_name, rnn_style, epochs)
        elif model == "ARIMA":
            run_address = osp.join(run_folder, "run_ARIMA.py")
            self.model_ = "ARIMA"
            command = "python {} {} {}".format(run_address, ts_folder, ts_name)
        elif model == "SARIMAX":
            run_address = osp.join(run_folder, "run_SARIMAX.py")
            self.model_ = "SARIMAX"
            command = "python {} {} {}".format(run_address, ts_folder, ts_name)
        else:
            raise TypeError("Unknown Type of model!")
        os.system(command)

        # plot result
        self.vis_wg.plot_result(self.folder, ts_name, self.model_, model)

        os.chdir(os.getcwd())

    def get_ts_address_button(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        ts_address_button = QPushButton('get')
        ts_address_button.setToolTip('Get the Data Address and Plot Original Data')
        ts_address_button.clicked.connect(self.ts_address_click)
        return ts_address_button

    def get_model_button(self):
        QToolTip.setFont(QFont('SansSerif', 10))
        model_button = QPushButton('run')
        model_button.setToolTip('Training the Models and Plot Predicted Results')
        # model_button.clicked.connect(self.run_program)
        self.vis_wg.graph_wg_result.clear()
        QApplication.processEvents()
        model_button.clicked.connect(lambda: self.output.run_model(self.get_para()))
        model_button.clicked.connect(
            lambda: self.vis_wg.plot_result(self.folder, self.ts_name_cb.currentText(),
                                            get_model_(self.model_cb.currentText()), self.model_cb.currentText()))
        return model_button

    def ts_address_click(self):
        ts_folder = self.ts_folder_cb.currentText()
        ts = self.ts_name_cb.currentText()
        datasets_address = self.datasets_address
        ts_address = osp.join(datasets_address, ts_folder, ts)
        self.ts_address_edit.text = ts_address
        self.ts_address_edit.setText(ts_address)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2))

    def get_ts_folder_cb(self):
        ts_folder_cb = QComboBox()
        ts_folder_cb.addItems(['HadCRUT5', 'cli_dash', 'temp_month', 'elect', 'traffic', 'sales'])
        return ts_folder_cb

    def get_ts_name_cb(self):
        ts_name_cb = QComboBox()
        ts_name_cb.addItems(['HadCRUT5_global', 'HadCRUT5_northern', 'HadCRUT5_southern'])
        return ts_name_cb

    def get_model_cb(self):
        model_cb = QComboBox()
        model_cb.addItems(['ResGraphNet', 'RESModel', 'GraphSage', 'UniMP', 'GCN', 'GIN', 'forest', 'linear', 'svr',
                           'sgd', 'LSTM', 'GRU', 'ARIMA', 'SARIMAX'])
        return model_cb

    def change_ts_name(self, text):
        ts_folder_all = ['HadCRUT5', 'cli_dash', 'temp_month', 'elect', 'traffic', 'sales']
        ts_HadCRUT5 = ['HadCRUT5_global', 'HadCRUT5_northern', 'HadCRUT5_southern']
        ts_cli_dash = ['Berkeley_Earth', 'ERA5_European', 'ERA5_Global', 'HadSST3']
        ts_temp_month = ['ERSSTv3b', 'ERSSTv4', 'NOAA']
        ts_elect = ['elect']
        ts_traffic = ['traffic']
        ts_sales = ['sales']
        ts_lists = ts_HadCRUT5, ts_cli_dash, ts_temp_month, ts_elect, ts_traffic, ts_sales
        self.ts_name_cb.clear()
        self.ts_name_cb.addItems(ts_lists[ts_folder_all.index(text)])

    def get_para(self):
        ts_name = self.ts_name_cb.currentText()
        ts_folder = self.ts_folder_cb.currentText()
        epochs = self.epochs_edit.text()
        model = self.model_cb.currentText()
        model_ = get_model_(model)
        root = self.root_edit.text()
        ratio_train = self.ratio_train_edit.text()
        l_x = self.lx_edit.text()
        l_y = self.ly_edit.text()
        para = [ts_folder, ts_name, epochs, model_, model, root, ratio_train, l_x, l_y]
        return para

    def change_epochs(self, text):
        if text in ['RESModel', 'LSTM', 'GRU']:
            self.epochs_edit.setText('50')
        elif text in ['ResGraphNet', 'GraphSage', 'UniMP', 'GCN', 'GIN']:
            self.epochs_edit.setText('1000')
        elif text in ['forest', 'linear', 'svr', 'sgd', 'ARIMA', 'SARIMAX']:
            self.epochs_edit.setText(' ')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    font = QFont()
    font.setPointSize(13)
    ex.setFont(font)
    sys.exit(app.exec_())
