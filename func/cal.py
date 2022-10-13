import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import pandas as pd
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
from matplotlib.collections import LineCollection
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# 两层全连接层
class Linear_Pre(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Linear_Pre, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(input_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        h = self.linear1(x)
        h = F.relu(h)
        h = F.dropout(h)
        h = self.linear2(h)
        return h


def eval_ml(regress, x_train, y_train, x_test, y_test, iv, way, fig_size=(12, 8), alpha=0.5):
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

    rmse_train = get_rmse(y_train_pred, y_train)
    rmse_test = get_rmse(y_test_pred, y_test)
    r2_train = get_r2_score(y_train_pred, y_train, axis=1)
    r2_test = get_r2_score(y_test_pred, y_test, axis=1)

    plt.figure(figsize=fig_size)
    # plt.title(title, fontsize=30)
    length_train = y_train_pred[:, -1].shape[0]
    length_test = y_test_pred[:, -1].shape[0]
    lim_train = np.arange(length_train // iv)
    lim_test = np.arange(length_train // iv, length_train // iv + length_test // iv)
    plt.plot(lim_train, sam(y_train[:, -1], iv, way), label="$y^{train}$", alpha=alpha)
    plt.plot(lim_train, sam(y_train_pred[:, -1], iv, way), label="$\hat{y}^{train}$", alpha=alpha)
    plt.plot(lim_test, sam(y_test[:, -1], iv, way), label="$y^{test}$", alpha=alpha)
    plt.plot(lim_test, sam(y_test_pred[:, -1], iv, way), label="$\hat{y}^{test}$", alpha=alpha)
    plt.legend(fontsize=25, loc=1)
    plt.ylabel("Value", fontsize=30)
    plt.xlabel("Time", fontsize=30)
    return rmse_train, rmse_test, r2_train, r2_test, y_train[:, -1], y_test[:, -1], y_train_pred[:, -1], y_test_pred[:, -1]


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


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


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
        self.relu = nn.ReLU()
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
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, output_dim)

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
            h_0 = h
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

            h = self.w2(h + h_0)
            h = self.sage2(h, edge_index)
        else:
            raise TypeError("{} is unknown for gnn style".format(self.gnn_style))
        return h


def path_graph(m):
    adm = np.zeros(shape=(m, m))
    for i in range(m - 1):
        adm[i, i + 1] = 1
    return adm


def cyclic_graph(m):
    adm = np.zeros(shape=(m, m))
    for i in range(m - 1):
        adm[i, i + 1] = 1
    adm[m - 1, 0] = 1
    return adm


def ts_un(n, k):
    adm = np.zeros(shape=(n, n))
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    else:
        for i in range(n):
            if i < (n - k):
                for k_one in range(1, k + 1):
                    adm[i, i + k_one] = 1.
            else:
                for k_one in range(1, k + 1):
                    if (k_one + i) >= n:
                        mod = (k_one + i) % n
                        adm[i, mod] = 1.
                    else:
                        adm[i, i + k_one] = 1.
    adm = (adm + adm.T) / 2
    return adm


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


def get_rmse(a1, a2):
    a1, a2 = a1.reshape(-1), a2.reshape(-1)
    m_1, m_2 = a1.shape[0], a2.shape[0]
    if m_1 < m_2:
        m = m_1
    else:
        m = m_2
    a1_m, a2_m = a1[:m], a2[:m]
    result = np.sqrt(np.sum(np.square(a1_m - a2_m)) / m)
    return result


def plot_2curve(a1, a2, l1=None, l2=None, title=None, x_name=None, y_name=None, alpha=0.5, fig_size=(12, 12),
                font_size=20, font_size_le=15, style='plot', s=10, marker=False):
    plt.figure(figsize=fig_size)
    if style == 'stem':
        plt.stem(a1, label=l1, alpha=alpha)
        plt.stem(a2, label=l2, alpha=alpha)
    elif style == 'plot':
        if marker:
            plt.plot(a1, label=l1, marker="o", markersize=10, alpha=alpha)
            plt.plot(a2, label=l2, marker="o", markersize=10, alpha=alpha)
        else:
            plt.plot(a1, label=l1, alpha=alpha)
            plt.plot(a2, label=l2, alpha=alpha)
    elif style == 'scatter':
        plt.scatter(x=np.linspace(1, a1.shape[0], a1.shape[0]), y=a1, label=l1, s=s)
        plt.scatter(x=np.linspace(1, a2.shape[0], a2.shape[0]), y=a2, label=l2, s=s)
    plt.xlabel(x_name, fontsize=font_size)
    plt.ylabel(y_name, fontsize=font_size)
    if (l1 is None) & (l2 is None):
        pass
    else:
        plt.legend(fontsize=font_size_le)
    plt.title(title, fontsize=font_size)
    return None


def plot_distribute(x, bins, jump, title=None, x_name=None, y_name="Frequency", fig_size=(12, 12)):
    if x.ndim == 1:
        pass
    elif (x.ndim == 2) & (x.shape[1] == 1):
        x = x.reshape(-1)
    else:
        raise TypeError("x must be 1d or row vector of 2d, but got {}d".format(x.shape[1]))
    x_label, x_bins = pd.cut(x, bins=bins, retbins=True)
    x_label = pd.DataFrame(x_label)
    x_label_vc = pd.DataFrame(x_label).value_counts()
    interval = x_label_vc.index.tolist()
    interval_sum = x_label_vc.values
    mid_all = []
    for i in range(bins):
        interval_one = interval[i][0]
        mid = (interval_one.left + interval_one.right) / 2
        mid_all.append(mid)
    mid_all = np.array(mid_all)
    sort_index = np.argsort(mid_all)
    mid_all_sort = mid_all[sort_index]
    mid_all_sort = np.around(mid_all_sort, 2)
    interval_sum_sort = interval_sum[(sort_index)]

    plt.figure(figsize=fig_size)
    if title is not None:
        plt.title(title, fontsize=30)
    plt.bar(x=np.arange(bins), height=interval_sum_sort)
    labels = mid_all_sort[np.arange(0, bins, jump)]
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    plt.xticks(ticks=ticks, labels=labels)
    if x_name is not None:
        plt.xlabel(x_name, fontsize=30)
    if y_name is not None:
        plt.ylabel(y_name, fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    return None


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


def sam(arr, iv, way="one"):
    num = arr.shape[0] // iv
    arr_sam = []
    for i in range(num):
        if way == "one":
            arr_one = arr[i * iv]
        elif way == "mean":
            arr_one_range = arr[i * iv: (i + 1) * iv]
            arr_one = np.mean(arr_one_range)
        else:
            raise TypeError("Unknown type of way!")
        arr_sam.append(arr_one)
    arr_sam = np.array(arr_sam)
    return arr_sam


def plot_spiral(x_list, cmap, fig_size=(12, 12), length=12, r=7):
    fig, ax = plt.subplots(figsize=fig_size)
    # x = x + 1.5
    radius = r + 0.4
    r_factor = r / 3.6

    idx = [2, 1, 0, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    idx_str = ["Mar", "Feb", "Jan", "Dec", "Nov", "Oct", "Sep", "Aug", "Jul", "Jun", "May", "Apr"]  # 标题
    idx_points = segment_circle(len(idx_str))
    lc_list = []
    linestyle_list = [(0, (5, 10)), 'solid']
    color = ['white', 'yellow']
    for j in range(len(x_list)):
        x = x_list[j] + 1.5
        x_vals = []
        y_vals = []
        for i in range(0, len(x)):
            r_pos = x[i] * r_factor
            x_unit_r, y_unit_r = idx_points[idx[i % length], :2]
            x_r, y_r = (r_pos * x_unit_r, r_pos * y_unit_r)
            x_vals.append(x_r)
            y_vals.append(y_r)
        # segments = [np.column_stack([x, y]) for x, y in zip(x_vals, y_vals)]
        pts = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)

        if not cmap:
            lc = LineCollection(segments, norm=plt.Normalize(0, 3.6), linestyle=linestyle_list[j],
                                colors=color[j])
        else:
            lc = LineCollection(segments, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0, 3.6), linestyle=linestyle_list[j])
        lc.set_array(np.asarray(x))
        lc_list.append(lc)

    fig.patch.set_facecolor('grey')
    ax.axis('equal')
    ax.set(xlim=(-10, 10), ylim=(-10, 10))
    circle = plt.Circle((0, 0), r, fc='#000000')
    ax.add_patch(circle)
    circle_2 = plt.Circle((0, 0), r_factor * 2.5, ec='red', fc=None, fill=False, lw=3.0)
    ax.add_patch(circle_2)
    circle_1_5 = plt.Circle((0, 0), r_factor * 3.0, ec='red', fc=None, fill=False, lw=3.0)
    ax.add_patch(circle_1_5)
    props_months = {'ha': 'center', 'va': 'center', 'fontsize': 24, 'color': 'white'}
    props_year = {'ha': 'center', 'va': 'center', 'fontsize': 36, 'color': 'white'}
    props_temp = {'ha': 'center', 'va': 'center', 'fontsize': 32, 'color': 'red'}
    ax.text(0, r_factor * 2.5, '1.5°C', props_temp, bbox={'facecolor': 'black', 'edgecolor': 'none'}, fontsize=40)
    ax.text(0, r_factor * 3.0, '2.0°C', props_temp, bbox={'facecolor': 'black', 'edgecolor': 'none'}, fontsize=40)
    # ax.text(0, r + 2.0, 'Global temperature change (1850-2021)\n{}'.format(title), props_year)

    for j in range(0, len(idx_str)):
        x_unit_r, y_unit_r, angle = idx_points[j]
        x_radius, y_radius = (radius * x_unit_r, radius * y_unit_r)
        angle = angle - 0.5 * np.pi
        ax.text(x_radius, y_radius, idx_str[j], props_months, rotation=np.rad2deg(angle), fontsize=60)
    for k in range(len(lc_list)):
        lc = lc_list[k]
        plt.gca().add_collection(lc)
    # plt.gca().add_collection(lc)
    ax.autoscale()
    ax.axis("off")
    return None


def segment_circle(num_segments):
    segment_rad = 2 * np.pi / num_segments
    segment_rads = segment_rad * np.arange(num_segments)
    coordX = np.cos(segment_rads)
    coordY = np.sin(segment_rads)
    return np.c_[coordX, coordY, segment_rads]


def plot_result(train_true, test_true, train_predict, test_predict, iv, way, fig_size):
    plt.figure(figsize=fig_size)
    length_train = train_true.shape[0]
    length_test = test_true.shape[0]
    lim_train = np.arange(length_train // iv)
    lim_test = np.arange(length_train // iv, length_train // iv + length_test // iv)
    plt.plot(lim_train, sam(train_true, iv, way), label="$y^{train}$", alpha=0.5)
    plt.plot(lim_train, sam(train_predict, iv, way), label="$\hat{y}^{train}$", alpha=0.5)
    plt.plot(lim_test, sam(test_true, iv, way), label="$y^{test}$", alpha=0.5)
    plt.plot(lim_test, sam(test_predict, iv, way), label="$\hat{y}^{test}$", alpha=0.5)
    plt.xlabel("Time", fontsize=30)
    plt.ylabel("Value", fontsize=30)
    plt.legend(fontsize=25)


def compare_distribute(x1, x2, bins=50, jump=5, title=None, x_name=None, y_name="Frequency", fig_size=(12, 12)):
    x = np.concatenate((x1, x2), axis=0)
    len_x, len_x1, len_x2 = x.shape[0], x1.shape[0], x2.shape[0]
    train_index = np.arange(len_x1)
    test_index = np.arange(len_x1, len_x)
    train_mask = np.zeros((len_x, ), dtype=bool)
    train_mask[train_index] = 1
    test_mask = np.zeros((len_x, ), dtype=bool)
    test_mask[test_index] = 1

    x_label, x_bins = pd.cut(x, bins=bins, retbins=True)
    x1_label, x2_label = x_label[train_mask], x_label[test_mask]
    x1_label, x2_label = pd.DataFrame(x1_label), pd.DataFrame(x2_label)
    x1_label_vc, x2_label_vc = x1_label.value_counts(), x2_label.value_counts()
    interval = x1_label_vc.index.tolist()

    interval_sum_x1, interval_sum_x2 = x1_label_vc.values, x2_label_vc.values
    mid_all = []
    for i in range(bins):
        interval_one = interval[i][0]
        mid = (interval_one.left + interval_one.right) / 2
        mid_all.append(mid)
    mid_all = np.array(mid_all)
    sort_index = np.argsort(mid_all)
    mid_all_sort = mid_all[sort_index]
    mid_all_sort = np.around(mid_all_sort, 2)
    interval_sum_x1_sort = interval_sum_x1[(sort_index)] / len_x1
    interval_sum_x2_sort = interval_sum_x2[(sort_index)] / len_x2

    plt.figure(figsize=fig_size)
    if title is not None:
        plt.title(title, fontsize=30)
    plt.bar(x=np.arange(bins), height=interval_sum_x1_sort, label="Before 1936", alpha=0.5)
    plt.bar(x=np.arange(bins), height=interval_sum_x2_sort, label=" After 1936", alpha=0.5)
    labels = mid_all_sort[np.arange(0, bins, jump)]
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    plt.xticks(ticks=ticks, labels=labels)
    if x_name is not None:
        plt.xlabel(x_name, fontsize=30)
    if y_name is not None:
        plt.ylabel(y_name, fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=35)
    return None


def plot_temp(data_train, data_test, fig_size):
    plt.figure(figsize=fig_size)
    num_train, num_test = data_train.shape[0], data_test.shape[0]
    num = num_train + num_test
    train_range = np.arange(num_train)
    test_range = np.arange(num_train, num)
    plt.plot(train_range, data_train, alpha=0.5, label="Before 1936")
    plt.plot(test_range, data_test, alpha=0.5, label=" After 1936")
    plt.legend(fontsize=35)
    plt.vlines((1936 - 1850) * 12, -0.75, 0.5, color='r', linewidth=4)
    plt.text((1936 - 1850) * 9.7, 0.6, '1936.Jan', fontsize=30)
    x_tick = [0, 360, 720, 1080, 1440, 1800]
    x_label = ["1850", "1880", "1910", "1940", "1970", "2000"]
    plt.xticks(x_tick, x_label, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Year", fontsize=30)
    plt.ylabel("Anomaly ($^{\circ}$C)", fontsize=30)
    return None

