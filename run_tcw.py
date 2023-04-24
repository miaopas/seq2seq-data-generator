import os
import time
from pathlib import Path
from functools import partial

from libs.train import *
from code_3w.data_3w_generator import w3_generator
from tcw.tcw_generator import dataset_generator


# from libs.train_with_tune import *
def truncatedloss(y, y_hat, loss=torch.nn.MSELoss(), t=10):
    truncated_loss = 0
    length = y.shape[-2]

    for i in range(length - t, length):
        truncated_loss += loss(y[:, i, :], y_hat[:, i, :])

    return truncated_loss


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    # tune_linear_cnn()
    # train_linear_cnn()

    # According to the default parameters in 3w_generator.py, the sequence length is 900
    # First time to run this code, the data will be generated and saved in the data folder

    T = 90
    data = np.load("tcw_1980_2022.npy")
    train, test = dataset_generator(
        data, length=100, train_test_ratio=0.7, sliding_window=True
    )

    train_in = np.expand_dims(train[:, :T, :], -1)
    train_output = np.expand_dims(train[:, -T:, :], -1)
    test_in = np.expand_dims(test[:, :T, :], -1)
    test_output = np.expand_dims(test[:, -T:, :], -1)

    activation = "tanh"  # Tanh RNN
    hid_dim = 128
    num_layers = 1
    input_dim = 1
    output_dim = 1
    config = {}
    config["loss"] = partial(truncatedloss, loss=torch.nn.MSELoss(), t=10)
    config["optim"] = "Adam"
    config["lr"] = 0.0001

    dtype = torch.float32

    train_tcw(
        f"{activation}RNN_tcw",
        RNNModel(
            config=config,
            hid_dim=hid_dim,
            num_layers=num_layers,
            input_dim=input_dim,
            output_dim=output_dim,
            return_sequence=True,
            dtype=32,
        ),
        train_in,
        train_output,
        test_in,
        test_output,
        call_backs=[
            EarlyStopping(
                monitor="valid_loss",
                min_delta=1e-10,
                patience=5,
                verbose=True,
                mode="min",
            )
        ],
        devices=1,
        dtype=dtype,
    )
