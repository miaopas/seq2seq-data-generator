# -*- coding: utf-8 -*-
# This script is used to load climate dataset
# 1980-2022 daily total column water (kg/m^2)
import numpy as np


def dataset_generator(
    data, length, train_test_ratio, overlap_ratio=0, sliding_window=False
):
    """

    Parameters
    ----------
    data : nparray, univariable series
    length : int, length of each series
    train_test_ratio : float, percent of training data to total
    overlap_ratio : float, optional, overlap ratio between two neighboring windows
    sliding_window : bool, optional, if True, use a sliding window instead of overlapping data

    Returns
    -------
    train : nparray, (batch, length, n_dim = 1)
    test : nparray, (batch, length, n_dim = 1)

    """
    if overlap_ratio != 0 and sliding_window == True:
        raise ValueError("Use overlap_ratio OR sliding_window")
    n_dim = data.ndim

    # splite train and test, data[:split] for train, data[split:] for test
    split = int(train_test_ratio * data.shape[0])
    train_data = data[:split]
    test_data = data[split:]
    train = []
    test = []

    # overlap in train dataset
    start = 0
    if sliding_window:
        while start + length < train_data.shape[0]:
            train.append(data[int(start) : int(start + length)])
            start += 1
    else:
        while start + length < train_data.shape[0]:
            train.append(data[int(start) : int(start + length)])
            start = int(start + length * (1 - overlap_ratio))
    train = np.array(train)

    # overlap in test dataset
    start = 0
    if sliding_window:
        while start + length < test_data.shape[0]:
            test.append(data[int(start) : int(start + length)])
            start += 1
    else:
        while start + length < test_data.shape[0]:
            test.append(data[int(start) : int(start + length)])
            start = int(start + length * (1 - overlap_ratio))
    test = np.array(test)

    # shuffle
    np.random.shuffle(train)
    np.random.shuffle(test)

    # expand dim
    train = np.expand_dims(train, axis=-1)
    test = np.expand_dims(test, axis=-1)
    return train, test


if __name__ == "__main__":
    data = np.load("tcw_1980_2022.npy")
    # generate dataset
    train, test = dataset_generator(
        data, length=100, train_test_ratio=0.7, sliding_window=True
    )
    # print(train.shape, test.shape)  # (10638, 100, 1) (4503, 100, 1)
    # print(train.dtype, test.dtype)
