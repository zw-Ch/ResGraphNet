import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_geometric.nn as gnn
import xlrd
import pickle


# 油井静态数据 and 相关信息
class NewWellInfo(object):
    def __init__(self, name, root="WellData"):
        self.root = root            # 相关数据文件的存放路径
        self.name = name            # 文件名
        self.info = pd.read_csv(osp.join(self.root, self.name + ".csv"))
        self.update_index()
        self.data, self.time, self.cp, self.op = self.pickle_load()
        self.well_name = self.info.loc[:, "井名"].values.reshape(-1)
        self.depth = self.info.loc[:, "校深"].values.reshape(-1)
        self.res = self.info.loc[:, "电阻率"].values.reshape(-1)
        self.lag = self.info.loc[:, "声波时差"].values.reshape(-1)
        self.density = self.info.loc[:, "岩石密度"].values.reshape(-1)
        self.cni = self.info.loc[:, "补偿中子"].values.reshape(-1)
        self.mud = self.info.loc[:, "泥质含量"].values.reshape(-1)
        self.porosity = self.info.loc[:, "总孔隙度"].values.reshape(-1)
        self.perm = self.info.loc[:, "渗透率"].values.reshape(-1)
        self.hyd = self.info.loc[:, "含油气饱和度"].values.reshape(-1)
        self.xy = self.info.loc[:, ["井口纵坐标X", "井口横坐标Y"]].values
        self.data_fill_time = None          # 对非采集时间的日产气量补零
        self.cp_fill_time = None            # 对非采集时间的套压补零
        self.op_fill_time = None            # 对非采集时间的油压补零
        self.time_all_list = None           # 所有采集时间的list格式
        self.time_all_arr = None            # 所有采集时间的array格式
        self.time_all_list_month = None     # 所有采集时间的list格式，仅考虑年、月
        self.time_all_arr_month = None      # 所有采集时间的array格式，仅考虑年、月

    def pickle_load(self):              # 读取data和time，pkl格式
        data_address = osp.join(self.root, self.name + "data.pkl")
        time_address = osp.join(self.root, self.name + "time.pkl")
        cp_address = osp.join(self.root, self.name + "cp.pkl")
        op_address = osp.join(self.root, self.name + "op.pkl")
        with open(data_address, "rb") as f_data:
            data = pickle.load(f_data)
        with open(time_address, "rb") as f_time:
            time = pickle.load(f_time)
        with open(cp_address, "rb") as f_cp:
            cp = pickle.load(f_cp)
        with open(op_address, "rb") as f_op:
            op = pickle.load(f_op)
        return data, time, cp, op

    def remain_by_xy_range(self, x_range, y_range):  # 仅保留一定范围内的节点及其数据
        x_min, x_max, y_min, y_max = x_range[0], x_range[1], y_range[0], y_range[1]
        remain_index = []
        num_well = self.xy.shape[0]      # 井的数量
        for i in range(num_well):
            xy_one = self.xy[i, :]           # 一个井的坐标
            x_one, y_one = xy_one[0], xy_one[1]
            if (x_one > x_min) & (x_one < x_max) & (y_one > y_min) & (y_one < y_max):
                remain_index.append(i)                  # 保留在范围内的节点索引
        self.delete_well_index(index=remain_index)      # 保留数据
        self.update_index()
        return None

    def delete_well_index(self, index):         # 仅保留index内的节点，index为节点索引
        self.cni = self.cni[index]
        self.density = self.density[index]
        self.depth = self.depth[index]
        self.hyd = self.hyd[index]
        self.info = self.info.iloc[index, :]
        self.lag = self.lag[index]
        self.mud = self.mud[index]
        self.perm = self.perm[index]
        self.porosity = self.porosity[index]
        self.res = self.res[index]
        self.well_name = self.well_name[index]
        self.xy = self.xy[index, :]
        data, time = [], []
        for i in range(len(index)):
            index_one = index[i]
            data_one = self.data[index_one]
            time_one = self.time[index_one]
            data.append(data_one)
            time.append(time_one)
        self.data = data
        self.time = time
        return None

    def remain_by_data(self, data_name, v_max=None, v_min=None):         # 根据数据，选择保留节点的方式
        data = self.select_data(data_name=data_name)
        if v_max is not None:
            index = np.argwhere(data < v_max).reshape(-1).tolist()
            self.delete_well_index(index=index)
        if v_min is not None:
            index = np.argwhere(data > v_min).reshape(-1).tolist()
            self.delete_well_index(index=index)
        self.update_index()
        return None

    def select_data(self, data_name):            # 挑选某个类型的静态数据
        if data_name == "cni":
            data = self.cni
        elif data_name == "density":
            data = self.density
        elif data_name == "depth":
            data = self.depth
        elif data_name == "hyd":
            data = self.hyd
        elif data_name == "lag":
            data = self.lag
        elif data_name == "mud":
            data = self.mud
        elif data_name == "perm":
            data = self.perm
        elif data_name == "porosity":
            data = self.porosity
        elif data_name == "res":
            data = self.res
        return data

    def update_index(self):         # 让节点的索引为[0, 1, ..., N-1]，N为节点数量
        info_index = np.arange(self.info.shape[0]).tolist()
        self.info.index = info_index
        return None

    def select_data_list(self, name_list):         # 根据静态数据的类型列表，挑选多个静态数据
        num_name = len(name_list)              # 静态数据的类型的数量
        data_all = []
        for i in range(num_name):
            name_one = name_list[i]
            data_one = self.select_data(data_name=name_one).reshape(-1, 1)
            if i != 0:
                data_all = np.hstack([data_all, data_one])
            else:
                data_all = data_one
        return data_all

    def merge_data(self, name, style="month"):        # 按照某种方式合并日产气量
        if style == "month":               # 将一个月的日产气量加起来
            if self.data_fill_time is not None:
                data_fill_time, time_all_list, time_all_arr = self.data_fill_time, self.time_all_list, self.time_all_arr
            else:
                data_fill_time, time_all_list, time_all_arr = self.fill_time(name)
            num_all_time = time_all_arr.shape[0]        # 所有采集时间的数量
            month_last, day_last, data_month_all, time_month_all = 0, 0, None, []
            time_all_arr_month, time_all_list_month = None, []
            for i in range(num_all_time):
                data_one = data_fill_time[:, i].reshape(-1, 1)         # 某一采集时间，所有井的日产气量
                time_one_arr = time_all_arr[i, :]                      # 某一采集时间，年月日为数组形式
                year, month, day = int(time_one_arr[0]), int(time_one_arr[1]), int(time_one_arr[2])   # 年、月、日
                if (month != month_last) | (i == num_all_time - 1):     # 采集时间与上一个采集时间不是同一个月，或运行到最后一次
                    if i != 0:      # 不是第一次运行，拼接上一个月份的日产气量
                        if data_month_all is None:      # 是第一个月
                            data_month_all = data_month
                        else:                           # 不是第一个月
                            data_month_all = np.hstack([data_month_all, data_month])
                    data_month = data_one                   # 当前月份第一天的日产气量
                    month_last = month

                    time_one_arr_month = np.array([year, month]).reshape(1, -1)    # 仅考虑年月，array格式
                    if time_all_arr_month is None:
                        time_all_arr_month = time_one_arr_month
                    else:
                        time_all_arr_month = np.vstack([time_all_arr_month, time_one_arr_month])

                    time_one_list_month = str(year) + "/" + str(month)             # 仅考虑年月，list格式存储为字符串
                    time_all_list_month.append(time_one_list_month)

                else:           # 采集时间与上一个采集时间为同一个月
                    data_month = data_month + data_one
                self.time_all_list_month = time_all_list_month
                self.time_all_arr_month = time_all_arr_month
        else:
            raise TypeError("style must be 'month', but got {}".format(style))
        return data_month_all, time_all_list_month, time_all_arr_month

    def choice_dynamic_data(self, well_list, rz_style, all_well, features):   # 由名称，挑选油井的日产气量or套压or油压
        if all_well:       # 考虑所有的油井
            if rz_style:
                raise ValueError("If all_well is True, rz_style must be None, otherwise the minimum length of"
                                 " data is too small")
            well_list = self.well_name
        num_choice = len(well_list)                 # 挑选的油井的数量
        data_all_well, cp_all_well, op_all_well = None, None, None      # 初始化动态数据
        length_all = []  # 记录下除零后，各个油井的日产气量的长度
        for i in range(num_choice):
            well_one = well_list[i]                 # 读取一口油井
            well_location = np.argwhere(self.well_name == well_one)     # 该井在数据集中的位置
            if well_location.shape[0] == 0:         # 若该井在数据集中不存在，报错
                raise ValueError("{} is not in the {}".format(well_one, self.name))
            else:
                well_location = well_location.reshape(-1)[0]
                data_one_well = self.data_fill_time[well_location, :].reshape(-1, 1)       # 该井在所有采集时间上的日产气量
                cp_one_well = self.cp_fill_time[well_location, :].reshape(-1, 1)        # 该井在所有采集时间上的套压
                op_one_well = self.op_fill_time[well_location, :].reshape(-1, 1)        # 该井在所有采集时间上的油压
            if (data_all_well is None) & (cp_all_well is None) & (op_all_well is None):         # 拼接所有井的动态数据
                data_all_well = data_one_well
                cp_all_well = cp_one_well
                op_all_well = op_one_well
            else:
                data_all_well = np.hstack([data_all_well, data_one_well])
                cp_all_well = np.hstack([cp_all_well, cp_one_well])
                op_all_well = np.hstack([op_all_well, op_one_well])

        if rz_style is not None:            # 需要基于日产气量数据，进行除零操作
            data_all, cp_all, op_all = None, None, None           # 除零后，各个油井日产气量、套压的拼接
            if rz_style == "rz_one":        # 各油井的日产气量除零后，拼接，长度为除零后某井日产气量的最小长度
                length_min = float('inf')       # 初始化最小长度
                for i in range(num_choice):
                    data_one, cp_one, op_one = data_all_well[:, i], cp_all_well[:, i], op_all_well[:, i]
                    index = np.argwhere(data_one).reshape(-1)         # 零值在日产气量中的位置
                    data_one, cp_one, op_one = data_one[index], cp_one[index], op_one[index]      # 除去各自的零值
                    length_one = data_one.shape[0]          # 除零后，日产气量的长度
                    length_all.append(length_one)
                    if length_one < length_min:
                        length_min = length_one             # 选择所有井中日产气量的最小长度
                    data_one = data_one[:length_min].reshape(-1, 1)        # 仅保留前n个值，n为最小长度
                    cp_one = cp_one[:length_min].reshape(-1, 1)
                    op_one = op_one[:length_min].reshape(-1, 1)
                    if data_all is None:            # 拼接除零后的日产气量
                        data_all, cp_all, op_all = data_one, cp_one, op_one
                    else:
                        if data_all.shape[0] > length_min:        # 若当前总日产气量的长度大于最小长度n，仅保留前n个值
                            data_all = data_all[:length_min, :]
                            cp_all = cp_all[:length_min, :]
                            op_all = op_all[:length_min, :]
                        data_all = np.hstack([data_all, data_one])
                        cp_all = np.hstack([cp_all, cp_one])
                        op_all = np.hstack([op_all, op_one])
                data_all_well, cp_all_well, op_all_well = data_all, cp_all, op_all
        if features == ["日产气量(10⁴m³/d)"]:            # 使用的数据集仅考虑日产气量
            return data_all_well, length_all
        elif features == ["日产气量(10⁴m³/d)", "套压(MPa)"]:
            data_all_well = np.expand_dims(data_all_well, axis=2)
            cp_all_well = np.expand_dims(cp_all_well, axis=2)
            data_used = np.concatenate((data_all_well, cp_all_well), axis=2)
        elif features == ["日产气量(10⁴m³/d)", "油压(MPa)"]:
            data_all_well = np.expand_dims(data_all_well, axis=2)
            op_all_well = np.expand_dims(op_all_well, axis=2)
            data_used = np.concatenate((data_all_well, op_all_well), axis=2)
        elif features == ["日产气量(10⁴m³/d)", "套压(MPa)", "油压(MPa)"]:
            data_all_well = np.expand_dims(data_all_well, axis=2)
            cp_all_well = np.expand_dims(cp_all_well, axis=2)
            op_all_well = np.expand_dims(op_all_well, axis=2)
            data_used = np.concatenate((data_all_well, cp_all_well, op_all_well), axis=2)
        return data_used, length_all

    def get_well_data(self, well, rz=False):                      # 找到某口油井的日产气量
        index = np.argwhere(self.well_name == well)
        if index.shape[0] == 0:         # 未找到该井
            raise ValueError("{} is not in the {}".format(well, self.name))
        else:
            index = index.reshape(-1)[0]
        well_data = self.data[index]
        if rz:          # 选择对日产气量除零
            index = np.argwhere(well_data).reshape(-1)
            well_data = well_data[index]
        return well_data

    def get_well_list_data(self, well_list, rz=False):                      # 找到一些油井的日产气量
        num_well = len(well_list)                   # 油井的数量
        data_all = []                               # 日产气量list
        for i in range(num_well):
            well_one = well_list[i]                         # 单口油井
            data_one = self.get_well_data(well_one, rz)     # 单口油井的日产气量
            data_all.append(data_one)
        return data_all

    def select_well_in_new(self, well_list):            # 输入list，挑选出位于new中的油井
        num = len(well_list)                # 油井数量
        well_list_selected = []
        for i in range(num):
            well_name_one = well_list[i]
            index = np.argwhere(self.well_name == well_name_one)
            if index.shape[0] == 0:         # 该油井不在new中
                continue
            else:                           # 该油井在new中
                well_list_selected.append(well_name_one)
        return well_list_selected

    def sum_data_fill_time(self, node_index=None):        # 计算不同的油井，所有日产气量的和
        if self.data_fill_time is None:
            data_fill_time, _, _, _, _ = self.fill_time(name="new")
        else:
            data_fill_time = self.data_fill_time
        if node_index is None:      # 考虑所有油井
            node_index = np.arange(data_fill_time.shape[0])
        data = data_fill_time[node_index, :]        # 选出这些油井的日产气量（在所有采集时间）
        data_sum = np.sum(data, axis=1)
        return data_sum

    def select_static_data_well_list(self, well_list, static_style=None):          # 挑选一些油井的静态数据
        num = len(well_list)            # 油井的数量
        if static_style is None:        # 若为None，默认提取所有静态特征
            static_style = ["cni", "density", "depth", "hyd", "lag", "mud", "perm", "porosity", "res"]
        static_data = self.select_data_list(static_style)           # 挑选这些静态特征
        static_data_well_all = None
        for i in range(num):
            well_one = well_list[i]
            index = np.argwhere(self.well_name == well_one)
            if index.shape[0] == 0:
                raise ValueError("{} is not in the {}".format(well_one, self.name))
            else:
                index = index.reshape(-1)[0]
                static_data_well_one = static_data[index, :].reshape(1, -1)         # 提该油井的静态特征
                if static_data_well_all is None:
                    static_data_well_all = static_data_well_one
                else:
                    static_data_well_all = np.vstack([static_data_well_all, static_data_well_one])
        return static_data_well_all

    def plot_well_in_block(self, block_name, block_well_name, legend=False, font_size_le=8):
        """
        按照不同的区块，绘制油井
        :param block_name: 所有区块的名称数组，类型array
        :param block_well_name: 各个区块包含的油井名称列表，类型list
        :param legend: 是否显示标注，类型bool
        :param font_size_le: 标注字体大小，类型int
        :return:
        """
        num_blocks = block_name.shape[0]            # 区块数量
        xy_all = []             # 所有区块的油井坐标
        for i in range(num_blocks):
            block_one = block_name[i]               # 当前区块
            block_well_one = block_well_name[i]     # 当前区块的油井
            num_well = len(block_well_one)          # 当前区块的油井数量
            xy = None
            for j in range(num_well):
                well_one = block_well_one[j]        # 当前油井
                index = np.argwhere(self.well_name == well_one)
                if index.shape[0] == 0:             # 该油井在NewWellInfo中不存在
                    continue
                else:
                    index = index.reshape(-1)[0]                # 当前油井在NewWellInfo中的索引
                    xy_one = self.xy[index, :].reshape(1, -1)
                if xy is None:
                    xy = xy_one
                else:
                    xy = np.vstack([xy, xy_one])                # 一个区块内，油井的坐标
            xy_all.append(xy)

        plt.figure(figsize=(12, 8))
        for i in range(len(xy_all)):
            plt.scatter(xy_all[i][:, 0], xy_all[i][:, 1], label="{}".format(block_name[i]))
        if legend:
            plt.legend(fontsize=font_size_le)
        plt.xlabel("x", fontsize=20)
        plt.ylabel("y", fontsize=20)
        return xy_all
