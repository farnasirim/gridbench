#!/usr/bin/env python
import os
import sys
import numpy as np
import json
import collections
import math
import pickle

from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
import pylab as pl
from matplotlib import collections  as mc

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import matplotlib.animation as animation


MOTION_DETECTION_THRESHOLD = 0.04

Vec3 = collections.namedtuple('Vec3', ['x', 'y', 'z'])

def rss(vec):
    sum = 0
    for component in vec:
        sum += component * component
    return math.sqrt(sum)

def vec_diff(a, b):
    return Vec3(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def new_start_datapoint(device_id, timestamp):
    return dict(device_id=device_id, timestamp=int(timestamp))


def start_key_func(s):
    return (s["timestamp"], s["device_id"])


def battery_key_func(b):
    return (b["timestamp"], b["device_id"])


def connected_key_func(c):
    return (c["timestamp"], c["device_id"])


def acc_key_func(a):
    return (a["timestamp"], a["device_id"])


def select(list_of_items, key):
    ret = [item[key] for item in list_of_items]
    return ret


def where(list_of_items, filter_func):
    return [item for item in list_of_items if filter_func(item)]


def new_battery_stat(s):
    battery_data = s["battery"].replace("charge", "'charge'").replace("voltage", "'voltage'")
    battery_data = eval(battery_data)
    return dict(timestamp=int(s["time"]) - 3600 * 4, device_id=int(s["sensor_id"]), charge=int(battery_data["charge"]), voltage=int(battery_data["voltage"]))


def new_acc_stat(s):
    try:
        acc_data = s["acc_record"].replace("x", "'x'").replace("y", "'y'").replace("z", "'z'")
        acc_data = eval(acc_data)
        return dict(timestamp=float(s["device_timestamp"]), device_id=int(s["sensor_id"]), point=(float(acc_data["x"]), float(acc_data["y"]), float(acc_data["z"])))
    except Exception as e:
        return None

def new_connected_stat(device_id, timestamp):
    return dict(device_id=device_id, timestamp=timestamp)


def uniqued(item_list, key_func):
    items = sorted(item_list, key=key_func)
    uniq_items = []
    for item in items:
        if len(uniq_items) == 0 or key_func(uniq_items[-1]) != key_func(item):
            uniq_items.append(item)
    return uniq_items


def get_chains(l, thresh):
    chains = []
    this_chain = []
    last_time = -123
    for data_point in l:
        if data_point["timestamp"] - last_time > thresh:
            if len(this_chain) > 1:
                chains.append(this_chain)
            this_chain = []

        this_chain.append(data_point)
        last_time = data_point["timestamp"]

    if len(this_chain) > 1:
        chains.append(this_chain)

    return chains

def diff_chain(ch):
    x, y = [], []

    last_x = 0
    for dt in ch:
        x.append(dt["timestamp"])
        y.append(dt["point"][0] - last_x)
        last_x = dt["point"][0]

    return x, y


def get_interarrivals(chain):
    ret = []

    for i in range(len(chain) - 1):
        ret.append((chain[i + 1]["timestamp"] - chain[i]["timestamp"]) * 60 * 1000)

    return ret


def get_exclusive_of(baseline, follower, dim, time_rad):
    it = follower.__iter__()
    xs, ys = [], []
    x, y = [], []
    f = None
    prev = -1
    for b in baseline:
        if f is None or (b["timestamp"] - f["timestamp"]) > time_rad:
            try:
                f = it.__next__()
                while (b["timestamp"] - f["timestamp"]) > time_rad:
                    f = it.__next__()
            except:
                f = None

        while len(y) > 0 and rss(vec_diff(y[-1], b["point"])) < MOTION_DETECTION_THRESHOLD:
            x = x[:-1]
            y = y[:-1]

        if (f is not None and abs(b["timestamp"] - f["timestamp"]) < time_rad) or (len(x) and b["timestamp"] - x[-1] > time_rad * 2):
            if len(x):
                if len(x) > 3:
                    xs.append(x)
                    ys.append(y)
                x, y = [], []

        if f is None or abs(b["timestamp"] - f["timestamp"]) > time_rad:
            x.append(b["timestamp"])
            y.append(b["point"])


    if len(x) > 3:
        xs.append(x)
        ys.append(y)

    return xs, ys


# def get_exclusive_of(baseline, follower, d, dim):
#     a = [[(1, 4), (1, -2)], [(2, -3), (4, 5)]]
#     for s in a:
#         s[0] = (s[0][0] * dim + baseline, s[0][1] * dim + follower)
#         s[1] = (s[1][0] + follower, s[1][1] * dim + baseline)
#     return a

def lines_from_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        return [line.strip() for line in lines if len(line.strip())]

def stat_from_line(stat_line):
    try:
        json_stat = json.loads(stat_line[stat_line.find("[STAT]") + len("[STAT]"):])
        return json_stat
    except Exception as e:
        print(e)
        return None



def stats_from_file(filename):
    stat_lines = [line for line in lines_from_file(filename) if "[STAT]" in line]
    info = {}
    info["device_exp"] = dict()
    with Pool(5) as p:
        json_stats = p.map(stat_from_line, stat_lines)

    json_stats = [j for j in json_stats if j is not None]

    for json_stat in json_stats:
        if "sensor_id" in json_stat and int(json_stat["sensor_id"]) not in info["device_exp"] and "code-branch" in json_stat:
            info["device_exp"][int(json_stat["sensor_id"])] = json_stat["code-branch"]
    return info, json_stats


class Dataset():
    def group_by_device_id(self, stat_name, stat_list, key_func):

        for data in stat_list:
            self.by_device[data["device_id"]][stat_name].append(data)

        for device_id in self.by_device:
            if stat_name in self.by_device[device_id]:
                self.by_device[device_id][stat_name] = sorted(self.by_device[device_id][stat_name], key=key_func)

        to_del = []
        for key in self.by_device.keys():
            # print(key, self.by_device[key].keys(), len(self.by_device[key]["battery"]))
            if len(self.by_device[key]["battery"]) == 0:
                to_del.append(key)

        for key in to_del:
            del self.by_device[key]

    def fix_device_timescale(self):
        starting_point = 1e10
        for device_id in self.by_device.keys():
            min_timestamps_bat = min(select(self.by_device[device_id]["battery"], "timestamp"))
            min_timestamps_acc = min(select(self.by_device[device_id]["acc"], "timestamp"))
            try:
                min_timestamps_connected = min(select(self.by_device[device_id]["connected"], "timestamp"))
            except:
                min_timestamps_connected = 1e18
            min_timestamps = min(min_timestamps_bat, min_timestamps_acc, min_timestamps_connected)
            starting_point = min(starting_point, min_timestamps)

        for device_id in self.by_device.keys():
            for item_list in self.by_device[device_id].values():
                for item in item_list:
                    item["timestamp"] -= starting_point
                    item["timestamp"] /= 60
                    self.mxtime = max(self.mxtime, item["timestamp"])


    def do_starts(self, metadata):
        for start_file in self.start_files:
            last_part = start_file.split("/")[-1]
            device_id = int(last_part[:last_part.find("-")])
            start_keys = list([int(timestamp.strip()) for timestamp in lines_from_file(start_file)])
            for start_key in start_keys:
                self.starts_data.append(new_start_datapoint(device_id, start_key))
        self.starts_data = uniqued(self.starts_data, start_key_func)
        self.group_by_device_id("starts", self.starts_data, start_key_func)

    def get_label(self, dev):
        add = ""
        if self.device_exp[dev] is not None:
            add = self.device_exp[dev]

        return "{}-{}".format(dev, add)


    def do_connected(self, metadata):
        for filename in self.general_files:
            lines = lines_from_file(filename)
            last_time = 0
            for (i, line) in enumerate(lines):
                if line.startswith("error 1"):
                    try:
                        last_time = float(line.split(" ")[1][:-1])
                    except:
                        pass
                if line.startswith("Connected "):
                    if last_time > self.metadata["start_timestamp"]:
                        self.connected_data.append(new_connected_stat(int(line.split(" ")[1]), last_time))

        self.group_by_device_id("connected", self.connected_data, connected_key_func)

    def time_func(self, stat):
        return int(stat["time"]) > self.metadata["start_timestamp"]

    def do_battery_acc(self, metadata):
        for filename in self.general_files:
            info, stats = stats_from_file(filename)
            for dev in info["device_exp"].keys():
                self.device_exp[dev] = info["device_exp"][dev]
            if "start_timestamp" in metadata:
                stats = where(stats, self.time_func)
            battery_stats = []
            acc_stats = []

            for stat in stats:
                if "battery" in stat or ("event" in stat and stat["event"] == "battery_handler_called"):
                    battery_stats.append(stat)
                elif "event" in stat and stat["event"] == "send_data_upstream":
                    acc_stats.append(stat)
            print("acc len:", len(acc_stats))
            for battery_stat in battery_stats:
                self.battery_data.append(new_battery_stat(battery_stat))

            for acc_stat in acc_stats:
                acc_stat = new_acc_stat(acc_stat)
                if acc_stat is not None:
                    self.acc_data.append(acc_stat)

            print("acc json len:", len(self.acc_data))

        self.battery_data = uniqued(self.battery_data, battery_key_func)
        self.acc_data = uniqued(self.acc_data, acc_key_func)

        self.group_by_device_id("battery", self.battery_data, battery_key_func)
        self.group_by_device_id("acc", self.acc_data, acc_key_func)

    def default_dict_lambda(self):
        return collections.defaultdict(list)

    def __init__(self, dataset_dir):
        self.general_files = []
        self.start_files = []

        self.device_exp = {}

        self.mxtime = 0

        self.starts_data = []
        self.battery_data = []
        self.acc_data = []
        self.connected_data = []

        self.name = dataset_dir.split("/")[-1]

        self.by_device = collections.defaultdict(self.default_dict_lambda)

        for dirname, _, filenames in os.walk(dataset_dir):
            for filename in filenames:
                filename = os.path.join(dirname, filename)
                if filename.endswith(".gz"):
                    continue
                if filename.endswith("-starts"):
                    self.start_files.append(filename)
                else: # filename.endswith("-err"):
                    self.general_files.append(filename)

        metadata = {}
        try:
            with open(os.path.join(dataset_dir, "meta.json"), "r") as f:
                metadata = json.load(f)
        except Exception as e:
            pass

        self.metadata = metadata

        self.do_starts(metadata)
        self.do_battery_acc(metadata)

        self.do_connected(metadata)

        self.fix_device_timescale()


def arg_from(name, args_list):
    for arg in args_list:
        if arg.startswith("--" + name):
            return arg.split("=")[1]
    return None



def main():
    cwd = os.path.dirname(os.path.abspath(sys.argv[0]))
    runs_data_dir = sys.argv[1]
    plot = "--plot" in sys.argv
    double_devices = "--doubles" in sys.argv
    save = arg_from("save", sys.argv)
    load = arg_from("load", sys.argv)
    dataset_names = os.listdir(runs_data_dir)
    dataset_dirs = [os.path.join(cwd, runs_data_dir, dataset_name) for dataset_name in dataset_names]

    datasets = []
    if load:
        with open(load, "rb") as f:
            datasets = pickle.load(f)
    else:
        print("about to start")
        with ThreadPool(10) as p:
            datasets = p.map(Dataset, dataset_dirs)
        datasets = [ds for ds in datasets if not double_devices or len(ds.by_device.keys()) == 2]
        print(len(datasets))

        # for dataset_dir in dataset_dirs:
        #     d = Dataset(dataset_dir)
        #     if double_devices:
        #         if len(d.by_device.keys()):
        #             datasets.append(d)
        #     else:
        #         datasets.append(d)

    if double_devices:
        correct_datasets = []
        for ds in datasets:
            if len(ds.by_device.keys()) == 2:
                correct_datasets.append(ds)
        datasets = correct_datasets


    if save:
        with open(save, "wb") as f:
            pickle.dump(datasets, f)

    print("datasets ready")

    if not plot:
        return


#     ### acc data
# 
#     device_lines = {}
#     all_timestamps = []
# 
#     dataset = datasets[0]
# 
#     devices = dataset.by_device.keys()
# 
#     for i, dataset in enumerate(datasets):
#         for d in devices:
#             all_timestamps.extend(select(dataset.by_device[d]["acc"], "timestamp"))
# 
#     all_timestamps = sorted(all_timestamps)
# 
#     def fl_cmp(a, b):
#         if abs(a - b) < 1e-8:
#             return 0;
#         if a < b:
#             return -1
#         return 1
# 
#     def get_line(acc_data_list, all_timestamps):
#         acc_ret = np.empty((3, len(all_timestamps)))
#         data_pointer = 0
#         last_data = (0.0, 0.0, 0.0)
#         for (i, tm) in enumerate(all_timestamps):
#             now = last_data
#             while data_pointer < len(acc_data_list) and fl_cmp(acc_data_list[data_pointer]["timestamp"], tm) < 0:
#                 now = acc_data_list[data_pointer]["point"]
#                 data_pointer += 1
# 
#             if data_pointer < len(acc_data_list) and fl_cmp(acc_data_list[data_pointer]["timestamp"], tm) <= 0:
#                 now = acc_data_list[data_pointer]["point"]
# 
#             acc_ret[:, i] = now
#             last_data = now
#         return acc_ret
# 
#     for d in devices:
#         device_lines[d] = get_line(dataset.by_device[d]["acc"], all_timestamps)
# 
#     def plot(data):
#         for line in data:
#             print(line.shape)
# 
#         fig = plt.figure()
#         ax = p3.Axes3D(fig)
#         lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
# 
#         ax.set_xlim3d([-1.0, 1.0])
#         ax.set_xlabel('X')
# 
#         ax.set_ylim3d([-1.0, 1.0])
#         ax.set_ylabel('Y')
# 
#         ax.set_zlim3d([-1.0, 1.0])
#         ax.set_zlabel('Z')
# 
#         ax.set_title('acc')
# 
#         def update_lines(num, dataLines, lines):
#             for line, data in zip(lines, dataLines):
#                 # NOTE: there is no .set_data() for 3 dim data...
#                 line.set_data(data[0:2, :num])
#                 line.set_3d_properties(data[2, :num])
#             return lines
# 
#         line_ani = animation.FuncAnimation(fig, update_lines, data[0].shape[1], fargs=(data, lines),
#                                            interval=0.01, blit=False)
# 
#         plt.show()


#   fr = 10000
#   # to = 10000
#     plot([list(device_lines.values())[0][:, fr:], list(device_lines.values())[1][:, fr:]])
    ### acc data



     # time_buckets = [0.1, 0.25, 0.5]
     # for (nplot, time_radius) in enumerate(time_buckets):
     #     fig, ax = pl.subplots(len(datasets), 1)
     #     for i, dataset in enumerate(datasets):
     #         devices = list(dataset.by_device.keys())

     #         plt.xlabel("timestamp (hours)")
     #         plt.ylabel("data reads")
     #         plt.title(dataset.name + " - data sent excluseively by each acc")
     #         plots = []
     #         d0, d1 = devices[0], devices[1]
     #         line_segments = []
     #         line_colors = []

     #         dim_color = [(0, 1.)]
     #         for acc_dim, color_coefficient in dim_color:
     #             segs = get_exclusive_of(d0, d1, dataset.by_device, acc_dim)
     #             colors = [(color_coefficient, 0, 0, 1)] * len(segs)
     #             line_segments.extend(segs)
     #             line_colors.extend(colors)
     #             segs = get_exclusive_of(d1, d0, dataset.by_device, acc_dim)
     #             colors = [(0, 0, color_coefficient, 1)] * len(segs)
     #             line_segments.extend(segs)
     #             line_colors.extend(colors)

     #         line_segments.append([(10, 10), (10, 10)])
     #         line_colors.append((0, 1, 0, 1))
     #         lc = mc.LineCollection(line_segments, colors=np.array(line_colors), linewidths=2)
     #         ax[i].add_collection(lc)
     #         lc = mc.PathCollection()
     #         ax[i].add_collection(lc)
     #         ax[i].autoscale()
     #         ax[i].margins(0.1)

    plot_x_read(datasets)
    plot_battery(datasets)
    # plot_data_sent(datasets)
    plot_data_diff(datasets)
    plot_chains_hist(datasets)
    plot_packets_per_minute(datasets)
    plot_connections_per_minute(datasets)

    plt.show()


def plot_x_read(datasets):
    print("plotting reported data")
    plt.figure()
    for i, dataset in enumerate(datasets):
        plt.subplot(len(datasets), 1, i + 1)
        plt.axis(ymin=-4.2, ymax=4.2, xmin=-0.2, xmax=dataset.mxtime + 1)

        devices = dataset.by_device.keys()

        plt.xlabel("timestamp (minutes)")
        plt.ylabel("x axis data read")
        plt.title(dataset.name + " - data read on the x axis of acc")
        plots = []
        for device_number in devices:
            d = dataset.by_device[device_number]
            plt.plot(select(d["acc"], "timestamp"), select(select(d["acc"], "point"), 0), ".-", label="{}".format(dataset.get_label(device_number)))
        plt.legend()


def plot_battery(datasets):
    print("plotting batter data")
    plt.figure()
    for i, dataset in enumerate(datasets):
        plt.subplot(len(datasets), 2, i * 2 + 1)
        plt.axis(xmin=-0.2, xmax=dataset.mxtime + 1)

        devices = dataset.by_device.keys()

        plt.xlabel("timestamp (minutes)")
        plt.ylabel("voltage")
        plt.title(dataset.name + " voltages")
        plots = []
        for device_number in devices:
            d = dataset.by_device[device_number]
            plt.plot(select(d["battery"], "timestamp"), select(d["battery"], "voltage"), ".-", label="{}".format(dataset.get_label(device_number)))
        plt.legend()

        plt.subplot(len(datasets), 2, i * 2 + 2)
        plt.xlabel("timestamp (minutes)")
        plt.ylabel("charge")
        plt.title(dataset.name + " charges")
        for device_number in devices:
            d = dataset.by_device[device_number]
            plt.plot(select(d["battery"], "timestamp"), select(d["battery"], "charge"), ".-", label="{}".format(dataset.get_label(device_number)))
        plt.legend()


def plot_data_sent(datasets):
    print("plotting exclusive data")
    time_buckets = [0.5/60]
    for (nplot, time_radius) in enumerate(time_buckets):
        plt.figure()
        for i, dataset in enumerate(datasets):
            plt.subplot(len(datasets), 1, i + 1)
            plt.axis(ymin=-4.2, ymax=4.2, xmin=-0.2, xmax=dataset.mxtime + 1)
            devices = list(dataset.by_device.keys())

            plt.xlabel("timestamp (minutes)")
            plt.ylabel("data reads")
            plt.title(dataset.name + " - data sent exclusively by each acc")
            plots = []
            d0, d1 = devices[0], devices[1]
            line_segments = []
            line_colors = []

            acc_dim = 0

            xs, ys = get_exclusive_of(dataset.by_device[d1]["acc"], dataset.by_device[d0]["acc"], acc_dim, time_radius)
            colors = ['r', 'maroon', 'tomato']
            axes = [0, 1, 2]

            has_legend = False
            for x, data in zip(xs, ys):
                for (axis, color) in list(zip(axes, colors))[:1]:
                    if not has_legend:
                        has_legend = True
                        plt.plot(x, select(data, axis), c=color, label="{}'s data".format(dataset.get_label(d1)))
                    plt.plot(x, select(data, axis), c=color)
                    plt.scatter(x, select(data, axis), c=color, s=5)

            has_legend = False
            xs, ys = get_exclusive_of(dataset.by_device[d0]["acc"], dataset.by_device[d1]["acc"], acc_dim, time_radius)
            colors = ['steelblue', 'darkblue', 'aqua']
            for x, data in zip(xs, ys):
                for (axis, color) in list(zip([0, 1, 2], colors))[:1]:
                    if not has_legend:
                        has_legend = True
                        plt.plot(x, select(data, axis), c=color, label="{}'s data".format(dataset.get_label(d0)))
                    else:
                        plt.plot(x, select(data, axis), c=color)
                    plt.scatter(x, select(data, axis), c=color, s=5)
            plt.legend()


def plot_data_diff(datasets):
    print("plotting differences data")
    chain_start_threshold = 1/60

    plt.figure()

    colors = ["r", "steelblue"]

    for (i, dataset) in enumerate(datasets):
        plt.subplot(len(datasets), 1, i + 1)
        plt.axis(ymin=-4.2, ymax=4.2, xmin=-0.2, xmax=dataset.mxtime + 1)
        devices = list(dataset.by_device.keys())
        plt.xlabel("timestamp (minutes)")
        plt.ylabel("data reads diff")
        plt.title(dataset.name + "data diffs")

        for (i, d) in enumerate(devices):
            has_legend = False
            xs, ys = [], []
            for chain in get_chains(dataset.by_device[d]["acc"], chain_start_threshold):
                x, y = diff_chain(chain)
                if not has_legend:
                    has_legend = True
                    plt.plot(x, y, c=colors[i], label="{}'s data".format(dataset.get_label(d)))
                else:
                    plt.plot(x, y, c=colors[i])
                plt.scatter(x, y, c=colors[i], s=5)

        plt.legend()


def plot_chains_hist(datasets):
    print("plotting interarrivals hist")
    chain_start_threshold = 1/60

    plt.figure()
    bins = 200

    for (i, dataset) in enumerate(datasets):
        devices = list(dataset.by_device.keys())
        times = {}
        for col_pos, d in zip([1, 2], devices):
            plt.subplot(len(datasets), 3, i * 3 + col_pos)

            plt.xlabel("inter-arrival times (milliseconds)")
            plt.title(dataset.name + " - inter-arrival times histogram")
            times[d] = []
            for chain in get_chains(dataset.by_device[d]["acc"], chain_start_threshold):
                times[d].extend(get_interarrivals(chain))
            plt.hist(times[d], bins=bins)
            plt.legend()

        plt.subplot(len(datasets), 3, i * 3 + 3)
        plt.xlabel("inter-arrival times (milliseconds)")
        plt.title(dataset.name + " - inter-arrival times overlay histogram")
        colors = [[1, 0, 0, 0.25], [0, 0, 1, 0.25]]
        for (i, d) in enumerate(times.keys()):
            plt.hist(times[d], bins=bins, color=colors[i], label=dataset.get_label(d))
        plt.legend()

def plot_packets_per_minute(datasets):
    print("plotting packets per minute")
    plt.figure()
    for (i, dataset) in enumerate(datasets):
        mn_time, mx_time = 1e18, 0
        devices = list(dataset.by_device.keys())
        timestamps = {}
        for col_pos, d in zip([1, 2], devices):
            timestamps[d] = select(dataset.by_device[d]["acc"], "timestamp")
            mn_time = min(mn_time, min(timestamps[d]))
            mx_time = max(mx_time, max(timestamps[d]))
        plt.subplot(len(datasets), 1, i + 1)
        plt.title(dataset.name + "arrivals per minute")
        colors = [[1, 0, 0, 0.25], [0, 0, 1, 0.25]]
        bins = list(np.arange(float(mn_time), float(mx_time), 60))
        for (i, d) in enumerate(timestamps.keys()):
            plt.hist(timestamps[d], bins=bins, color=colors[i], label=dataset.get_label(d))
        plt.legend()


def plot_connections_per_minute(datasets):
    print("plotting connections per minute")
    plt.figure()
    for (i, dataset) in enumerate(datasets):
        mn_time, mx_time = 1e18, 0
        devices = list(dataset.by_device.keys())
        timestamps = {}
        for col_pos, d in zip([1, 2], devices):
            timestamps[d] = select(dataset.by_device[d]["connected"], "timestamp")
            if len(timestamps[d]) == 0:
                timestamps[d] = [0, 1, 2]
            mn_time = min(mn_time, min(timestamps[d]))
            mx_time = max(mx_time, max(timestamps[d]))
        plt.subplot(len(datasets), 1, i + 1)
        plt.title(dataset.name + "connected events")
        colors = [[1, 0, 0, 0.25], [0, 0, 1, 0.25]]
        bins = list(np.arange(float(mn_time), float(mx_time), 60))
        for (i, d) in enumerate(timestamps.keys()):
            plt.hist(timestamps[d], bins=bins, color=colors[i], label=dataset.get_label(d))
        plt.legend()

    return
    # plt.figure()
    # for (i, dataset) in enumerate(datasets):
    #     devices = list(dataset.by_device.keys())
    #     timestamps = {}
    #     for col_pos, d in zip([1, 2], devices):
    #         timestamps[d] = select(dataset.by_device[d]["connected"], "timestamp")
    #     plt.subplot(len(datasets), 1, i + 1)
    #     plt.title(dataset.name + "connected events")
    #     colors = [[1, 0, 0, 0.25], [0, 0, 1, 0.25]]
    #     for (i, d) in enumerate(timestamps.keys()):
    #         plt.plot(timestamps[d], [i + 1] * len(timestamps[d]), color=colors[i], label=dataset.get_label(d))
    #         plt.scatter(timestamps[d], [i + 1] * len(timestamps[d]), color=colors[i], label=dataset.get_label(d))
    #     plt.legend()


if __name__ == "__main__":
    main()
