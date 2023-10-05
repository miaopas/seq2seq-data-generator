from typing import List, Tuple

import hydra
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pyrootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #


import torch

from src import utils
from src.data.components.lf.datagen import LF_generate, rhos

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting memory testing!")

    # Load the model
    model_ckpt = torch.load(cfg.ckpt_path)["state_dict"]
    model.load_state_dict(model_ckpt)

    # All one inputs
    inputs = torch.ones(1, cfg.data.seq_length * 4, cfg.data.input_dim, dtype=torch.float64) * 0.1

    # Evaluate the predictions
    outputs = model(inputs)
    max_outputs, _ = torch.max(outputs, dim=0)
    print(max_outputs.shape)  # T * D
    # torch.save(max_outputs, cfg.paths.output_dir + "/max_outputs.pt")
    max_outputs = max_outputs.detach().numpy()

    # outputs_memory = np.squeeze(np.abs(max_outputs[1:] - max_outputs[:-1]))  # (T-1) *D
    outputs_memory = np.squeeze(
        np.abs(max_outputs[2:] - 2 * max_outputs[1:-1] + max_outputs[:-2])
    )  # (T-2) *D
    print("outputs_memory.shape", outputs_memory.shape)
    print("outputs_memory datatype", outputs_memory.dtype)
    plt.plot(outputs_memory, label="model memory")

    # Draw the target memory for toy case
    if cfg.memory_type is not None:
        targets = LF_generate(
            None,
            None,
            None,
            1,
            0.1,
            rhos[cfg.memory_type],
            None,
            Gaussian_input=False,
            evaluate_inputs=inputs.detach().numpy(),
        )
        targets_memory = np.squeeze(np.abs(targets[0, 1:] - targets[0, :-1]))
        plt.plot(targets_memory, label="target memory")

    # plt.ylim([1e-4, 0.05])
    plt.yscale("log")
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Memory $M(H)$")
    plt.legend()
    plt.savefig(cfg.paths.output_dir + "/memory_function.pdf")
    plt.close()

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="memory.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
