from pytorch_lightning import Trainer
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import os
import pickle
from libs.seq2seq_model import RNNModel, LinearRNNModel, ComplexLinearRNNModel
from libs.lfgenerator import Exponential
from math import floor
from datetime import datetime
from ml_collections import FrozenConfigDict
from libs.lfgenerator import Shift
import numpy as np
from torch.utils.data import ConcatDataset

# from libs.seq2seq_model import S4DModel, S4Model




def train_model(
    name,
    model,
    input,
    output,
    train_test_split,
    epochs=300,
    batch_size=128,
    check_point_monitor="valid_loss",
    devices=4,
    call_backs=None,
):
    """_summary_

    Args:
        name (str): Name of this run
        model (Model):The model
        input (ndarray): input array
        output (ndarray): output array
        train_test_split (float): ratio of train test split
    """
    if input is not None:
        # If input not provided then skip this part
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, dtype=torch.float64)
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output, dtype=torch.float64)

        dataset = torch.utils.data.TensorDataset(input, output)
        total = len(dataset)
        train_size = floor(total * train_test_split)
        test_size = total - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_dataset = ConcatDataset([train_dataset]*100)
        valid_dataset = ConcatDataset([valid_dataset]*100)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )
    else:
        train_loader = None
        valid_loader = None
    import itertools

   

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    now = datetime.now().strftime("%H:%M:%S__%m-%d")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor=check_point_monitor,
        filename=name + "-{epoch:02d}-{valid_loss:.2e}",
    )

    if call_backs:
        default_callbacks = [checkpoint_callback, lr_monitor] + call_backs
    else:
        default_callbacks = [checkpoint_callback, lr_monitor]



    if devices == 1:
        trainer = Trainer(
            accelerator="gpu",
            devices=[3],
            max_epochs=epochs,
            precision=64,
            logger=TensorBoardLogger("runs", name=name),
            callbacks=default_callbacks,
        )
    else:
        trainer = Trainer(
            accelerator="gpu",
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=False),
            max_epochs=epochs,
            precision=64,
            logger=TensorBoardLogger("runs", name=name),
            callbacks=default_callbacks,
        )

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )

    return trainer.validate(model=model, dataloaders=train_loader)


def shift_rnn():
    shifts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    res = []
    for s in shifts:
        hid_dim = 32
        model = ComplexLinearRNNModel(hid_dim, 1, 1)
        generator = Shift(
            {"input_dim": 1, "path_len": 128, "shift": [s], "data_num": 100000}
        )

        x, y = generator.generate()
        early_stop_callback = EarlyStopping(
            monitor="train_loss", min_delta=1e-10, patience=3, verbose=False, mode="min"
        )
        res.append(
            train_model(
                "rnn_shift",
                model,
                x,
                y,
                0.8,
                epochs=400,
                devices=4,
                call_backs=[early_stop_callback],
            )
        )

    import pickle

    with open("res.pickle", "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


def shift_rnn_complex():
    shifts = [100]
    res = []
    for s in shifts:
        hid_dim = 32
        model = LinearRNNModel(hid_dim, 1, 1)
        generator = Shift(
            {"input_dim": 1, "path_len": 128, "shift": [s], "data_num": 100000}
        )

        x, y = generator.generate()
        # early_stop_callback = EarlyStopping(monitor="train_loss",min_delta=1e-10, patience=3, verbose=False, mode="min")
        # res.append(train_model('rnn_shift', model, x, y, 0.8, epochs=400, devices=4, call_backs=[early_stop_callback]) )
        res.append(
            train_model(
                "rnn_shift", model, x, y, 0.8, epochs=400, devices=4, call_backs=[]
            )
        )

    import pickle

    with open("res.pickle", "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_exp():
    hid_dim = 12
    model = LinearRNNModel(hid_dim, 1, 1)
    generator = Exponential({"input_dim": 1, "path_len": 128, "data_num": 10000})

    x, y = generator.generate()

    permuted = x.copy()
    permuted[:5] = x[-5:]
    permuted[-5:] = x[:5]

    x = permuted
    early_stop_callback = EarlyStopping(
        monitor="train_loss", min_delta=1e-10, patience=3, verbose=False, mode="min"
    )
    train_model("exp", model, x, y, 0.8, epochs=400, devices=4, call_backs=[])


def shift_s4d():
    shifts = [1]
    res = []
    for s in shifts:
        hid_dim = 32
        model = S4DModel(hid_dim=hid_dim, output_dim=1)
        generator = Shift(
            {"input_dim": 1, "path_len": 5, "shift": [s], "data_num": 100000}
        )

        x, y = generator.generate()
        # early_stop_callback = EarlyStopping(monitor="train_loss",min_delta=1e-10, patience=3, verbose=False, mode="min")
        # res.append(train_model('rnn_shift', model, x, y, 0.8, epochs=400, devices=4, call_backs=[early_stop_callback]) )
        res.append(
            train_model(
                "s4d_shift", model, x, y, 0.8, epochs=400, devices=1, call_backs=[]
            )
        )

    import pickle

    with open("res.pickle", "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


def shift_s4():
    shifts = [30]
    res = []
    for s in shifts:
        hid_dim = 32
        model = S4Model(hid_dim=hid_dim, output_dim=1)
        generator = Shift(
            {"input_dim": 1, "path_len": 128, "shift": [s], "data_num": 100000}
        )

        x, y = generator.generate()
        # early_stop_callback = EarlyStopping(monitor="train_loss",min_delta=1e-10, patience=3, verbose=False, mode="min")
        # res.append(train_model('rnn_shift', model, x, y, 0.8, epochs=400, devices=4, call_backs=[early_stop_callback]) )
        res.append(
            train_model(
                "s4d_shift", model, x, y, 0.8, epochs=400, devices=1, call_backs=[]
            )
        )

    import pickle

    with open("res.pickle", "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_3w(
    name,
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=300,
    batch_size=128,
    check_point_monitor="valid_loss",
    devices=1,
    call_backs=None,
    dtype=torch.float64,
):
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train, dtype=dtype)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=dtype)
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test, dtype=dtype)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=dtype)

    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    now = datetime.now().strftime("%H:%M:%S__%m-%d")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor=check_point_monitor,
        filename=name + "-{epoch:02d}-{valid_loss:.2e}",
    )

    default_callbacks = [checkpoint_callback, lr_monitor] + call_backs

    if devices == 1:
        trainer = Trainer(
            accelerator="gpu",
            devices=[3],
            max_epochs=epochs,
            precision=64,
            logger=TensorBoardLogger("runs", name=name),
            callbacks=default_callbacks,
        )
    else:
        trainer = Trainer(
            accelerator="gpu",
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=False),
            max_epochs=epochs,
            precision=64,
            logger=TensorBoardLogger("runs", name=name),
            callbacks=default_callbacks,
        )

    print("Start training")

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )

    return trainer.validate(model=model, dataloaders=valid_loader)


def train_tcw(
    name,
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=300,
    batch_size=128,
    check_point_monitor="valid_loss",
    devices=1,
    call_backs=None,
    dtype=torch.float64,
):
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train, dtype=dtype)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=dtype)
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test, dtype=dtype)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=dtype)

    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    now = datetime.now().strftime("%H:%M:%S__%m-%d")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor=check_point_monitor,
        filename=name + "-{epoch:02d}-{valid_loss:.2e}",
    )

    default_callbacks = [checkpoint_callback, lr_monitor] + call_backs

    if devices == 1:
        trainer = Trainer(
            accelerator="gpu",
            devices=[3],
            max_epochs=epochs,
            precision=64,
            logger=TensorBoardLogger("runs", name=name),
            callbacks=default_callbacks,
        )
    else:
        trainer = Trainer(
            accelerator="gpu",
            devices=devices,
            strategy=DDPStrategy(find_unused_parameters=False),
            max_epochs=epochs,
            precision=64,
            logger=TensorBoardLogger("runs", name=name),
            callbacks=default_callbacks,
        )

    print("Start training")

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )

    return trainer.validate(model=model, dataloaders=valid_loader)



