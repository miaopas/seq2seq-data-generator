import os
import time
from pathlib import Path

from libs.train import *
from dataprepare_3w.data_3w_generator import w3_generator

# from libs.train_with_tune import *


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

    # tune_linear_cnn()
    # train_linear_cnn()

    # According to the default parameters in 3w_generator.py, the sequence length is 900
    # First time to run this code, the data will be generated and saved in the data folder

    T = 900
    data_dir = Path(f"./data/3w_T_{T}/")
    if os.path.exists(data_dir):
        train = torch.load(f"{data_dir}/x_train.pt")
        train_output = torch.load(f"{data_dir}/y_train.pt")
        test = torch.load(f"{data_dir}/x_test.pt")
        test_output = torch.load(f"{data_dir}/y_test.pt")
    else:
        data_dir.mkdir(exist_ok=True, parents=True)
        train, train_output, test, test_output = w3_generator(900, 0.7, overlap_ratio=0)
        train_output = np.expand_dims(train_output, -1)
        test_output = np.expand_dims(test_output, -1)

        torch.save(train, data_dir / "x_train.pt")
        torch.save(train_output, data_dir / "y_train.pt")
        torch.save(test, data_dir / "x_test.pt")
        torch.save(test_output, data_dir / "y_test.pt")

    # print(train.shape, train_output.shape, test.shape, test_output.shape)
    # print(train.dtype, train_output.dtype, test.dtype, test_output.dtype)
    # exit()

    activation = "tanh"  # Tanh RNN
    hid_dim = 128
    num_layers = 1
    input_dim = 3
    output_dim = 1
    config = {}
    config["loss"] = torch.nn.MSELoss()
    config["optim"] = "Adam"
    config["lr"] = 0.0001

    dtype = torch.float32

    train_3w(
        f"{activation}RNN",
        RNNModel(
            config=config,
            hid_dim=hid_dim,
            num_layers=num_layers,
            input_dim=input_dim,
            output_dim=output_dim,
            return_sequence=False,
            dtype=32,
        ),
        train,
        train_output,
        test,
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
