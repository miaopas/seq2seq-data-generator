from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import torch
from pytorch_lightning import Trainer
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
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

def data_process(input, output, train_test_split, batch_size=128):
    if input is not None:
    # If input not provided then skip this part
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, dtype=torch.float64)
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output, dtype=torch.float64)

        dataset = torch.utils.data.TensorDataset(input, output)
        total = len(dataset)
        train_size = floor(total*train_test_split)
        test_size = total - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,drop_last=True, num_workers=os.cpu_count(), pin_memory=True)
    else:
        train_loader = None
        valid_loader = None
    
    return train_loader, valid_loader

def train_model(config, model, data_loaders, epochs):
    
    model = model(config)
    train_loader, valid_loader = data_loaders
    

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
                                        save_top_k=5, 
                                        monitor="valid_loss",
                                        filename="{epoch:02d}-{valid_loss:.2e}") 

    tune_report = TuneReportCallback(
                {
                    "loss": "valid_loss",
                },
                on="validation_end")
    default_callbacks = [ lr_monitor, tune_report, checkpoint_callback]

    
    trainer = Trainer(accelerator="gpu", 
                devices=1,
                max_epochs=epochs,
                precision=64,
                logger=TensorBoardLogger("runs"),
                callbacks=default_callbacks,
                enable_progress_bar=False
                )
   


    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


def tune_model(name, model, config, data_loaders, epochs, resources_per_trial):
    """_summary_

    Args:
        name (str): Name of this run
        model (Model):The model
        data (tuple): (input_loader, output_loader)
        train_test_split (float): ratio of train test split
    """
    
    
    reporter = CLIReporter(
        parameter_columns=["hid_dim"],
        metric_columns=["loss", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_model,
                                                    model=model, data_loaders=data_loaders, epochs=epochs)
    trainable_with_gpu = tune.with_resources(train_fn_with_parameters, {"gpu":1})
   
    tuner = tune.Tuner(
        tune.with_resources(
            trainable_with_gpu,
            resources=resources_per_trial
        ),
        run_config=air.RunConfig(
            name=name,
            progress_reporter=reporter,
            local_dir='ray_results',
            verbose=1
        ),
        param_space=config,
    )
    results = tuner.fit()
    # print("Best hyperparameters found were: ", results.get_best_result().config)


def tune_linear_rnn():

    # Set the configurations 
    config = {
        'input_dim': 1, 'output_dim':1, 'loss': nn.MSELoss(), # This should be fixed

         'optim': 'Adam', 'lr': 1e-3,

        'hid_dim': tune.grid_search([8, 16, 32, 64])   # This is a parameter list for the model
    }
    

    model = LinearRNNModel
    x, y = Shift({'input_dim':1, 'path_len':128 ,'shift':[10], 'data_num':25600}).generate()

    data_loaders = data_process(x,y,train_test_split=0.8, batch_size=128)
    
    # The devices to be used
    os.environ["CUDA_VISIBLE_DEVICES"] ="0,2,3"

    # Number of GPU per trial. Can set to one to see usage first.  Since for this set up the usage is about 35%, so we have set can run two experiments on a GPU at the same time.
    gpus_per_trial = 0.5
    resources_per_trial = {"cpu": 4, "gpu": gpus_per_trial}

    tune_model("tune_linear_rnn", model, config, data_loaders, epochs=300, resources_per_trial=resources_per_trial)
   