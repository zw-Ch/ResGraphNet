import os
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def tran_adm_to_edge_index(adm):
    if isinstance(adm, np.ndarray):
        u, v = np.nonzero(adm)
        num_edges = u.shape[0]
        edge_index = np.vstack([u.reshape(1, -1), v.reshape(1, -1)])
        edge_weight = np.zeros(shape=u.shape)
        for i in range(num_edges):
            edge_weight_one = adm[u[i], v[i]]
            edge_weight[i] = edge_weight_one
        edge_index = torch.from_numpy(edge_index.astype(np.int64))
        edge_weight = torch.from_numpy(edge_weight.astype(np.float32))
        return edge_index, edge_weight
    elif torch.is_tensor(adm):
        u, v = torch.nonzero(adm, as_tuple=True)
        num_edges = u.shape[0]
        edge_index = torch.cat((u.view(1, -1), v.view(1, -1)))
        edge_weight = torch.zeros(size=u.shape)
        for i in range(num_edges):
            edge_weight[i] = adm[u[i], v[i]]
        return edge_index, edge_weight


def get_r2_score(y1, y2, axis):
    if (type(y1) is np.ndarray) & (type(y2) is np.ndarray):      # numpy数组类型
        pass
    elif (torch.is_tensor(y1)) & (torch.is_tensor(y2)):          # pytorch张量类型
        y1 = y1.detach().cpu().numpy()
        y2 = y2.detach().cpu().numpy()
    else:
        raise TypeError("type of y1 and y must be the same, but got {} and {}".format(type(y1), type(y2)))
    if y1.shape != y2.shape:
        raise ValueError("shape of y1 and y2 must be the same, but got {} and {}".format(y1.shape, y2.shape))
    if y1.ndim == 1:
        y1 = np.expand_dims(y1, axis=1)
        y2 = np.expand_dims(y2, axis=1)
    elif y1.ndim == 2:
        pass
    else:
        raise ValueError("y1 and y2 must be 1d or 2d, but got {}d".format(y1.ndim))
    if axis == 0:
        num_col = y1.shape[0]
    elif axis == 1:
        num_col = y1.shape[1]
    else:
        raise TypeError("axis must be equal as 0 or 1, but got {}".format(axis))
    r2_all = 0
    for i in range(num_col):
        if axis == 0:
            y1_one = y1[i, :]
            y2_one = y2[i, :]
        elif axis == 1:
            y1_one = y1[:, i]
            y2_one = y2[:, i]
        else:
            raise TypeError("axis must be equal as 0 or 1, but got {}".format(axis))
        r2_one = r2_score(y1_one, y2_one)
        r2_all = r2_all + r2_one
    r2 = r2_all / num_col
    return r2


def create_inout_sequences(input_data, x_length=32, y_length=4, style="list", ml_dim=0, ld1=False):
    seq_list, seq_arr, label_arr = [], None, None
    data_length = len(input_data)
    x_y_length = x_length + y_length
    if style == "list":
        for i in range(data_length - x_y_length + 1):
            if input_data.ndim == 2:
                seq = input_data[i: (i + x_length), :]
                if ld1:
                    label = input_data[(i + x_length): (i + x_length + y_length), ml_dim]
                else:
                    label = input_data[(i + x_length): (i + x_length + y_length), :]
            elif input_data.ndim == 1:
                seq = input_data[i: (i + x_length)]
                label = input_data[(i + x_length): (i + x_length + y_length)]
            elif input_data.ndim == 3:
                seq = input_data[i: (i + x_length), :, :]
                if ld1:
                    label = input_data[(i + x_length): (i + x_length + y_length), :, ml_dim]
                else:
                    label = input_data[(i + x_length): (i + x_length + y_length), :, :]
            seq_list.append((seq, label))
        return seq_list

    elif style == "arr":
        for i in range(data_length - x_y_length + 1):
            if input_data.ndim == 2:
                seq = input_data[i: (i + x_length), :]
                label = input_data[(i + x_length): (i + x_length + y_length), ml_dim].reshape(1, -1)
                seq = np.expand_dims(seq, 0)
            elif input_data.ndim == 1:
                seq = input_data[i: (i + x_length)]
                label = input_data[(i + x_length): (i + x_length + y_length)]
                seq, label = seq.reshape(1, -1), label.reshape(1, -1)
            if (seq_arr is None) & (label_arr is None):
                seq_arr, label_arr = seq, label
            else:
                seq_arr, label_arr = np.vstack([seq_arr, seq]), np.vstack([label_arr, label])
        return seq_arr, label_arr


def path_graph(m):
    adm = np.zeros(shape=(m, m))
    for i in range(m - 1):
        adm[i, i + 1] = 1
    return adm


class GNNTime(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_weight, gnn_style, num_nodes):
        super(GNNTime, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.edge_weight = nn.Parameter(edge_weight)
        self.gnn_style = gnn_style
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.gcn1 = gnn.GCNConv(input_dim, hidden_dim)
        self.gcn2 = gnn.GCNConv(hidden_dim, output_dim)
        self.che1 = gnn.ChebConv(input_dim, hidden_dim, K=3)
        self.che2 = gnn.ChebConv(hidden_dim, output_dim, K=3)
        self.sage1 = gnn.SAGEConv(input_dim, hidden_dim)
        self.sage2 = gnn.SAGEConv(hidden_dim, output_dim)
        self.gin1 = gnn.GraphConv(input_dim, hidden_dim)
        self.gin2 = gnn.GraphConv(hidden_dim, output_dim)
        self.tran1 = gnn.TransformerConv(input_dim, hidden_dim)
        self.tran2 = gnn.TransformerConv(hidden_dim, output_dim)
        self.tag1 = gnn.TAGConv(input_dim, hidden_dim)
        self.tag2 = gnn.TAGConv(hidden_dim, output_dim)
        self.gat1 = gnn.GATConv(input_dim, hidden_dim)
        self.gat2 = gnn.GATConv(hidden_dim, output_dim)
        self.cnn1 = nn.Conv2d(1, hidden_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn5 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn6 = nn.Conv2d(hidden_dim, 1, kernel_size=(5, 5), padding=(2, 2))
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.linear1 = nn.Linear(output_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        if self.gnn_style == "GCN":             # Graph Convolution Network Model
            h = self.gcn1(x, edge_index, self.edge_weight)
            h = self.drop(h)
            h = self.gcn2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "Cheb":           # Chebyshev Network
            h = self.che1(x, edge_index)
            h = self.drop(h)
            h = self.che2(h, edge_index)
        elif self.gnn_style == "GraphSage":          # GraphSAGE Model
            h = self.sage1(x, edge_index)
            h = self.drop(h)
            h = self.sage2(h, edge_index)
        elif self.gnn_style == "GIN":           # Graph Isomorphic Network Model
            h = self.gin1(x, edge_index, self.edge_weight)
            h = self.drop(h)
            h = self.gin2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "UniMP":
            h = self.tran1(x, edge_index)
            h = self.drop(h)
            h = self.tran2(h, edge_index)
        elif self.gnn_style == "TAGCN":
            h = self.tag1(x, edge_index, self.edge_weight)
            h = self.drop(h)
            h = self.tag2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "GAT":
            h = self.gat1(x, edge_index)
            h = self.drop(h)
            h = self.gat2(h, edge_index)
        elif self.gnn_style == "ResGraphNet":
            h = self.sage1(x, edge_index)
            h = h.unsqueeze(0).unsqueeze(0)
            out = self.cnn1(h)

            out_0 = out
            out = self.cnn2(self.drop(out))
            out = self.cnn3(self.drop(out))
            out = out + out_0

            out_1 = out
            out = self.cnn4(self.drop(out))
            out = self.cnn5(self.drop(out))
            out = out + out_1

            out = self.cnn6(self.drop(out))
            h = out.squeeze(0).squeeze(0)

            h = self.sage2(h, edge_index)
        else:
            raise TypeError("{} is unknown for gnn style".format(self.gnn_style))
        return h


def eval_ml(regress, x_train, y_train, x_test, y_test):
    num_train, num_test = x_train.shape[0], x_test.shape[0]
    num = num_train + num_test
    y_length = y_train.shape[1]
    for i in range(y_length):
        y_train_one = y_train[:, i]
        regress.fit(x_train, y_train_one)
        y_train_pred_one = regress.predict(x_train).reshape(-1, 1)
        y_test_pred_one = regress.predict(x_test).reshape(-1, 1)
        if i == 0:
            y_train_pred = y_train_pred_one
            y_test_pred = y_test_pred_one
        else:
            y_train_pred = np.concatenate((y_train_pred, y_train_pred_one), axis=1)
            y_test_pred = np.concatenate((y_test_pred, y_test_pred_one), axis=1)

    r2_train = get_r2_score(y_train_pred, y_train, axis=1)
    r2_test = get_r2_score(y_test_pred, y_test, axis=1)
    return r2_train, r2_test, y_train[:, -1], y_test[:, -1], y_train_pred[:, -1], y_test_pred[:, -1]


class MyData(Dataset):
    def __init__(self, x, y):
        super(MyData, self).__init__()
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_one = self.x[idx, :]
        y_one = self.y[idx, :]
        return x_one, y_one


class RESModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, x_length):
        super(RESModel, self).__init__()
        self.lin_pre = nn.Linear(1, hidden_dim)
        self.cnn1 = nn.Conv2d(1, hidden_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn5 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn6 = nn.Conv2d(hidden_dim, 1, kernel_size=(5, 5), padding=(2, 2))
        self.last1 = nn.Linear(hidden_dim, 1)
        self.last2 = nn.Linear(x_length, output_dim)
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()

    def forward(self, x):
        h = x.unsqueeze(1).unsqueeze(3)
        h = self.lin_pre(h)
        h = self.cnn1(h)
        h_0 = h
        h = self.cnn2(self.drop(h))
        h = self.cnn3(self.drop(h))
        h = h + h_0
        h_1 = h
        h = self.cnn4(self.drop(h))
        h = self.cnn5(self.drop(h))
        h = h + h_1
        h = self.cnn6(h)
        h = self.last1(h).squeeze(3).squeeze(1)
        h = self.last2(h)
        return h


class RNNTime(nn.Module):
    def __init__(self, rnn_style, in_dim, hid_dim, out_dim, l_x, num_layers):
        super(RNNTime, self).__init__()
        self.rnn_style = rnn_style
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.relu = nn.ReLU()
        self.linear_pre = nn.Linear(in_dim, hid_dim)
        self.lstm1 = nn.LSTM(hid_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.gru1 = nn.GRU(hid_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.last1 = nn.Linear(hid_dim, 1)
        self.last2 = nn.Linear(l_x, out_dim)

    def forward(self, x):
        h = x.unsqueeze(2)
        h = self.linear_pre(h)
        if self.rnn_style == "LSTM":
            h, (_, _) = self.lstm1(self.pre(h))
        elif self.rnn_style == "GRU":
            h, (_) = self.gru1(self.pre(h))
        else:
            raise TypeError("Unknown Type of rnn_style!")
        h = self.last1(h).squeeze(2)
        h = self.last2(h)
        return h
