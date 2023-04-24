# -*- coding: utf-8 -*-
# 3w dataset generator for a binary classification task with multivariable input
import numpy as np
import pandas as pd
import os

import torch

normal_path = "/home/shida/3W/dataset/0"
inst_path = "/home/shida/3W/dataset/4"  # instability data
normal_name = os.listdir(normal_path)
inst_name = os.listdir(inst_path)

# for test
# normal_name = normal_name[:10]
# inst_name = inst_name[:10]


def split_dataset(filelist, train_test_ratio):
    split = int(len(filelist) * train_test_ratio)
    np.random.shuffle(filelist)
    trainlist = filelist[:split]
    testlist = filelist[split:]
    return trainlist, testlist


def overlap_window(data, length, overlap_ratio=0, sliding_window=False):
    if overlap_ratio != 0 and sliding_window == True:
        raise ValueError("Use overlap_ratio OR sliding_window")
    start = 0
    dataset = []
    if sliding_window:
        while start + length < data.shape[0]:
            dataset.append(data[int(start) : int(start + length)])
            start += 1
    else:
        while start + length < data.shape[0]:
            dataset.append(data[int(start) : int(start + length)])
            start = int(start + length * (1 - overlap_ratio))
    dataset = np.array(dataset)
    return dataset


def read_files(filepath, filelist, length, overlap_ratio=0, sliding_window=False):
    valid_file_count = 0
    for idx, file in enumerate(filelist):
        path = os.path.join(filepath, file)
        data = pd.read_csv(path, sep=",")
        nan_idx = []

        p_tpt = data["P-TPT"]
        t_tpt = data["T-TPT"]
        p_ckp = data["P-MON-CKP"]

        if (
            np.isnan(p_tpt).sum() > 0
            or np.isnan(t_tpt).sum() > 0
            or np.isnan(p_ckp).sum() > 0
        ):
            # nan_idx.append(idx) # if want to print idx for file with nan values
            pass  # if have nan, skip this file

        else:
            p_tpt = overlap_window(p_tpt, length, overlap_ratio, sliding_window)
            t_tpt = overlap_window(t_tpt, length, overlap_ratio, sliding_window)
            p_ckp = overlap_window(p_ckp, length, overlap_ratio, sliding_window)
            valid_file_count += 1
            if valid_file_count == 1:
                train = np.stack([p_tpt, t_tpt, p_ckp], axis=2)

            else:
                temp = np.stack([p_tpt, t_tpt, p_ckp], axis=2)
                train = np.concatenate((train, temp), axis=0)

    return train


def w3_generator(length, train_test_ratio, overlap_ratio=0, sliding_window=False):
    # split normal dataset
    trainlist_normal, testlist_normal = split_dataset(normal_name, train_test_ratio)
    # split instable dataset
    trainlist_inst, testlist_inst = split_dataset(inst_name, train_test_ratio)

    # train dataset
    # build normal train dataset
    train_normal = read_files(
        normal_path, trainlist_normal, length, overlap_ratio, sliding_window
    )
    train_normal_output = np.zeros(train_normal.shape[0])
    # build instable train dataset
    train_inst = read_files(
        inst_path, trainlist_inst, length, overlap_ratio, sliding_window
    )
    train_inst_output = np.ones(train_inst.shape[0])
    # combine
    train = np.concatenate((train_normal, train_inst), axis=0)
    train_output = np.concatenate((train_normal_output, train_inst_output), axis=0)
    # shffle
    state1 = np.random.get_state()
    np.random.shuffle(train)
    np.random.set_state(state1)
    np.random.shuffle(train_output)

    # test dataset
    # build normal test dataset
    test_normal = read_files(
        normal_path, testlist_normal, length, overlap_ratio, sliding_window
    )
    test_normal_output = np.zeros(test_normal.shape[0])
    # build instable test dataset
    test_inst = read_files(
        inst_path, testlist_inst, length, overlap_ratio, sliding_window
    )
    test_inst_output = np.ones(test_inst.shape[0])
    # combine
    test = np.concatenate((test_normal, test_inst), axis=0)
    test_output = np.concatenate((test_normal_output, test_inst_output), axis=0)
    # shffle
    state2 = np.random.get_state()
    np.random.shuffle(test)
    np.random.set_state(state2)
    np.random.shuffle(test_output)

    return train, train_output, test, test_output


# default length = 900 for a 15 mins time window
if __name__ == "__main__":
    train_input, train_output, test_input, test_output = w3_generator(
        900, 0.7, overlap_ratio=0
    )
    # print(train_input.shape) # (9014, 900, 3), remove nan to (8146, 900, 3)
    has_nan = torch.any(torch.isnan(torch.Tensor(train_input)))
    print(has_nan)  # expect to be False
