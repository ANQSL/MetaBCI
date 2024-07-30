import os

import numpy
import numpy as np
import scipy.io
import torch
from torch.utils.data import TensorDataset


def transform_database(path, name):
    files = os.listdir(path)
    data = []
    data_index = []
    for file in files:
        file_data = scipy.io.loadmat(os.path.join(path, file), struct_as_record=False, squeeze_me=True)
        if "filter" in file:
            data.append(file_data["EEG"].data)
        else:
            events = file_data["EEG"].event
            if type(events) == numpy.ndarray:
                for i, event in enumerate(events):
                    event_index = event.latency
                    if event_index != 0:
                        if i < len(events) - 1:
                            if events[i].type != events[i + 1].type:
                                data_index.append(event_index)
                        else:
                            data_index.append(event_index)
            else:
                event_index = events.latency
                data_index.append(event_index)
    data = np.concatenate(data, axis=-1)
    data_index = np.array(data_index)
    data_index = np.reshape(data_index, (12, 2))
    # eeg_data = []
    # for index in data_index:
    #     eeg_data.append(data[:, _label[0]:_label[1]])
    label = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    if not os.path.exists("../database/{}".format(name)):
        os.makedirs("../database/{}".format(name))
    np.save("../database/{}/data.npy".format(name), data)
    np.save("../database/{}/data_index.npy".format(name), data_index)
    np.save("../database/{}/label.npy".format(name), label)


def slide_window(data_list, labels, window_size, stride):
    samples = []
    mark = []
    for i, data in enumerate(data_list):
        data = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=1)
        data = data[:, ::stride]
        data = np.swapaxes(data, 0, 1)
        # data = np.reshape(data, (-1, 62, 5 * window_size))
        label = np.full(data.shape[0], labels[i])
        samples.append(data)
        mark.append(label)
    samples = np.concatenate(samples, axis=0)
    mark = np.concatenate(mark, axis=0)
    return samples, mark


def standardization(data):
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    data = (data - mean) / std
    return data


# transform_database("C:/Users/20291/Desktop/fear_data/wsm-1/2024-07-27/1", "wsm")

def get_database1(path, i, data_len, stride):
    subjects = os.listdir(path)
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for subject_id, subject in enumerate(subjects):
        if "rar" not in subject:
            subject_path = os.path.join(path, subject)
            data = np.load(subject_path + "/data.npy")
            label = np.load(subject_path + "/label.npy")
            label = label.astype(np.int64)
            data_index = np.load(subject_path + "/data_index.npy")
            eeg_data = []
            for index in data_index:
                eeg_data.append(data[:, index[0] + 10000:index[1]])
            data, label = slide_window(eeg_data, label, data_len * 1000, stride * 1000)

            if i != subject_id:
                train_data.append(data)
                train_label.append(label)
            else:
                test_data.append(data)
                test_label.append(label)

    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)

    test_data = np.concatenate(test_data, axis=0)
    test_label = np.concatenate(test_label, axis=0)

    train_data = standardization(train_data)
    test_data = standardization(test_data)

    # train_tensor = torch.tensor(train_data, dtype=torch.float)
    # test_tensor = torch.tensor(test_data, dtype=torch.float)
    # train_label = torch.tensor(train_label, dtype=torch.int64)
    # test_label = torch.tensor(test_label, dtype=torch.int64)
    #
    # # 转化为TensorDataset
    # train_dataset = TensorDataset(train_tensor, train_label)
    # test_dataset = TensorDataset(test_tensor, test_label)
    # 返回训练集和测试集
    return train_data, train_label,test_data,test_label


def get_database(path, data_len, stride):
    subjects = os.listdir(path)
    train_data = []
    train_label = []
    for subject_id, subject in enumerate(subjects):
        if "rar" not in subject:
            subject_path = os.path.join(path, subject)
            data = np.load(subject_path + "/data.npy")
            label = np.load(subject_path + "/label.npy")
            label = label.astype(np.int64)
            data_index = np.load(subject_path + "/data_index.npy")
            eeg_data = []
            for index in data_index:
                eeg_data.append(data[:, index[0] + 10000:index[1]])
            data, label = slide_window(eeg_data, label, data_len * 1000, stride * 1000)

            train_data.append(data)
            train_label.append(label)

    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)

    train_data = standardization(train_data)

    # train_tensor = torch.tensor(train_data, dtype=torch.float)
    # test_tensor = torch.tensor(test_data, dtype=torch.float)
    # train_label = torch.tensor(train_label, dtype=torch.int64)
    # test_label = torch.tensor(test_label, dtype=torch.int64)
    #
    # # 转化为TensorDataset
    # train_dataset = TensorDataset(train_tensor, train_label)
    # test_dataset = TensorDataset(test_tensor, test_label)
    # 返回训练集和测试集
    return train_data, train_label
