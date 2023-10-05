from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
import torch
from pytorch_lightning import Trainer
import torch.nn as nn
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import os
from libs.seq2seq_model import RNNModel, LinearRNNModel, TCNModel, LinearTCNModel
from math import floor
from libs.lfgenerator import Shift, ExpPeak, TwoPart, LorenzRandFGenerator


def data_process(name, input, output, train_test_split, batch_size=128):
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

    # train_loader.__class__.__repr__ = lambda _: name

    from typing import Tuple

    class Tuple_with_Name(Tuple):
        def __repr__(self) -> str:
            return name

    return Tuple_with_Name((train_loader, valid_loader))


def train_model(config, model, epochs, tune=True):
    model = model(config["model"])
    train_loader, valid_loader = config["data"]

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5, monitor="valid_loss", filename="{epoch:02d}-{valid_loss:.2e}"
    )

    tune_report = TuneReportCallback(
        {
            "train_loss": "train_loss",
            "valid_loss": "valid_loss",
        },
        on="validation_end",
    )
    if tune:
        default_callbacks = [lr_monitor, tune_report, checkpoint_callback]
    else:
        default_callbacks = [lr_monitor, checkpoint_callback]

    # wandb_logger = WandbLogger(name=config['name'], save_dir='runs')
    tfboard_logger = TensorBoardLogger("runs", name=config["name"])
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        precision=64,
        logger=tfboard_logger,
        callbacks=default_callbacks,
        enable_progress_bar=not tune,
    )

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )


def tune_model(name, model, config, parameter_columns, epochs, resources_per_trial):
    """_summary_

    Args:
        name (str): Name of this run
        model (Model):The model
        data (tuple): (input_loader, output_loader)
        train_test_split (float): ratio of train test split
    """

    reporter = CLIReporter(
        parameter_columns=parameter_columns,
        metric_columns=["train_loss", "valid_loss", "training_iteration"],
    )

    train_fn_with_parameters = tune.with_parameters(
        train_model, model=model, epochs=epochs
    )
    trainable_with_gpu = tune.with_resources(train_fn_with_parameters, {"gpu": 1})

    tuner = tune.Tuner(
        tune.with_resources(trainable_with_gpu, resources=resources_per_trial),
        run_config=air.RunConfig(
            name=name, progress_reporter=reporter, local_dir="ray_results", verbose=1
        ),
        param_space=config,
    )
    results = tuner.fit()
    import pickle

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    # print("Best hyperparameters found were: ", results.get_best_result().config)


def tune_linear_rnn():
    # shifts = [10,20,30,40]
    # dataloaders = []
    # for shift in shifts:
    #     x, y = Shift({'input_dim':1, 'path_len':128 ,'shift':[shift], 'data_num':25600*2}).generate()
    #     data_loader = data_process(f'Shift_{shift}', x,y,train_test_split=0.8, batch_size=256)
    #     dataloaders.append(data_loader)

    # centers = [2,4,6,8]
    # sigmas =  [8,8,8,8]
    # dataloaders = []
    # for center, sigma in zip(centers, sigmas):
    #     x, y = TwoPart({'input_dim':1, 'path_len':128 ,'centers':[[center]], 'sigmas': [[sigma]], 'data_num':25600*2}).generate()
    #     data_loader = data_process(f'TwoPart_{center}', x,y,train_test_split=0.8, batch_size=256)
    #     dataloaders.append(data_loader)

    model = LinearRNNModel

    # Set the configurations
    config = {
        "name": "tune_linear_rnn",
        "data": tune.grid_search(dataloaders),
        "model": {
            "input_dim": 1,
            "output_dim": 1,
            "loss": nn.MSELoss(),  # This should be fixed
            "optim": "Adam",
            "lr": 1e-3,
            "hid_dim": tune.grid_search([8, 16, 32, 64]),
        },  # This is a parameter list for the model
    }
    parameter_columns = ["model/hid_dim", "data"]

    # The devices to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # Number of GPU per trial. Can set to one to see usage first.  Since for this set up the usage is about 35%, so we have set can run two experiments on a GPU at the same time.
    gpus_per_trial = 0.5
    resources_per_trial = {"cpu": 4, "gpu": gpus_per_trial}

    tune_model(
        "tune_linear_rnn",
        model,
        config,
        parameter_columns,
        epochs=300,
        resources_per_trial=resources_per_trial,
    )


def tune_cnn():
    # Set the configurations
    config = {
        "input_dim": 1,
        "output_dim": 1,
        "loss": nn.MSELoss(),  # This should be fixed
        "optim": "Adam",
        "lr": 1e-3,
        "kernel_size": 2,
        "channels": tune.grid_search([1, 2, 3, 4]),
        "layers": tune.grid_search(
            [1, 2, 3, 4, 5]
        ),  # This is a parameter list for the model
    }
    parameter_columns = ["layers", "channels"]

    model = TCNModel
    x, y = Shift(
        {"input_dim": 1, "path_len": 128, "shift": [10], "data_num": 12800}
    ).generate()

    data_loaders = data_process(x, y, train_test_split=0.8, batch_size=128)

    # The devices to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

    # Number of GPU per trial. Can set to one to see usage first.  Since for this set up the usage is about 35%, so we have set can run two experiments on a GPU at the same time.
    gpus_per_trial = 0.3
    resources_per_trial = {"cpu": 4, "gpu": gpus_per_trial}

    tune_model(
        "tune_cnn",
        model,
        config,
        parameter_columns,
        data_loaders,
        epochs=300,
        resources_per_trial=resources_per_trial,
    )


def train_linear_cnn():
    config = {
        "name": "train_linear_cnn",
        "input_dim": 1,
        "output_dim": 1,
        "loss": nn.MSELoss(),  # This should be fixed
        "optim": "Adam",
        "lr": 1e-3,
        "kernel_size": 2,
        "channels": 4,
        "layers": 6,  # This is a parameter list for the model
    }

    model = LinearTCNModel
    x, y = Shift(
        {"input_dim": 1, "path_len": 128, "shift": [10], "data_num": 25600}
    ).generate()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
    data_loaders = data_process(x, y, train_test_split=0.8, batch_size=128)
    train_model(config, model, data_loaders, 400, tune=False)


def tune_linear_cnn():
    # Set the configurations
    config = {
        "name": "tune_linear_cnn",
        "input_dim": 1,
        "output_dim": 1,
        "loss": nn.MSELoss(),  # These should be fixed
        "optim": "Adam",
        "lr": 1e-3,
        "kernel_size": 2,
        "channels": tune.grid_search([1, 2, 3]),
        "layers": tune.grid_search(
            [1, 2, 3, 4, 5, 6, 7]
        ),  # This is a parameter list for the model
    }
    parameter_columns = ["layers", "channels"]

    model = LinearTCNModel
    x, y = Shift(
        {"input_dim": 1, "path_len": 128, "shift": [50], "data_num": 12800}
    ).generate()
    # x, y = ExpPeak({'lambda':[0.5], 'centers': [10],'sigmas':[20],'path_len':128,'data_num':12800}).generate()
    data_loaders = data_process(x, y, train_test_split=0.8, batch_size=128)

    # The devices to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

    # Number of GPU per trial. Can set to one to see usage first.  Since for this set up the usage is about 35%, so we have set can run two experiments on a GPU at the same time.
    gpus_per_trial = 0.3
    resources_per_trial = {"cpu": 3, "gpu": gpus_per_trial}

    tune_model("tune_linear_cnn", model, config, parameter_columns, data_loaders, epochs=300, resources_per_trial=resources_per_trial)


def train_linear_rnn():
    model = LinearRNNModel
    x, y = Shift({'input_dim':1, 'path_len':128 ,'shift':[10], 'data_num':25600}).generate()
    data_loaders = data_process(f'TwoPart_10', x,y,train_test_split=0.8, batch_size=128)


    config = {'name': 'train_linear_rnn',
        

        'data': data_loaders,

        'model':{'input_dim': 1, 'output_dim':1, 'loss': nn.MSELoss(), # This should be fixed

        'optim': 'Adam', 'lr': 1e-3, 'hid_dim': 64} # This is a parameter list for the model
          
    }

    os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2"
    train_model(config, model, 400, tune=False)


def train_rnn():
    model = RNNModel
    x, y = LorenzRandFGenerator({'n_init':512, 'K':1, 'J':10,'path_len':128, 'only_terminal':False, 'data_num':25600}).generate()
    data_loaders = data_process(f'lorenz_10', x,y,train_test_split=0.8, batch_size=128)


    config = {'name': 'train_rnn',
        

        'data': data_loaders,

        'model':{'input_dim': 1, 'output_dim':1, 'num_layers':1, 'loss': nn.MSELoss(), # This should be fixed

        'optim': 'Adam', 'lr': 1e-3, 'hid_dim': 128} # This is a parameter list for the model
          
    }

    os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"
    train_model(config, model, 400, tune=False)