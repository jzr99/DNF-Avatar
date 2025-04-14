import sys
import os
import time
import logging
import torch
import hydra
import wandb

from datetime import datetime
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig


@hydra.main(version_base=None, config_path="configs", config_name="config_peoplesnapshot")
def main(config_explicit_implicit):
    print(OmegaConf.to_yaml(config_explicit_implicit), flush=True)
    # hydra_config = HydraConfig.get()
    config = config_explicit_implicit.implicit_branch

    if config.mode not in ["train", "validate", "test", "predict"]:
        raise ValueError("invalid mode: {}".format(config.mode))

    n_gpus = len(config.gpu.split(","))

    import datasets
    import systems
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import WandbLogger#, CSVLogger
    from utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
    )

    # trial_name = config.tag + datetime.now().strftime("@%Y%m%d-%H%M%S")
    trial_name = config.tag
    config.exp_dir = os.path.join(config.exp_dir, config.name)
    config.log_dir = os.path.join(config.exp_dir, trial_name, "log")
    config.save_dir = os.path.join(config.exp_dir, trial_name, "save")
    config.ckpt_dir = os.path.join(config.exp_dir, trial_name, "ckpt")
    config.code_dir = os.path.join(config.exp_dir, trial_name, "code")
    config.config_dir = os.path.join(config.exp_dir, trial_name, "config")

    logger = logging.getLogger("pytorch_lightning")
    if config.verbose:
        logger.setLevel(logging.DEBUG)

    if "seed" not in config:
        config.seed = int(time.time() * 1000) % 1000
    pl.seed_everything(config.seed)

    dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(
        config.system.name,
        config,
        config_explicit_implicit,
        load_from_checkpoint=None if not config.resume_weights_only else config.resume,
    )

    callbacks = []
    if config.mode == "train":
        callbacks += [
            ModelCheckpoint(dirpath=config.ckpt_dir, **config.checkpoint),
            LearningRateMonitor(logging_interval="step"),
            # CodeSnapshotCallback(config.code_dir, use_version=False),
            ConfigSnapshotCallback(config, config.config_dir, use_version=False),
            CustomProgressBar(refresh_rate=1),
        ]

    loggers = []
    if config.mode == "train":
        loggers += [
            # CSVLogger(config.exp_dir, name=trial_name, version="csv_logs"),
            WandbLogger(
                name=config.name + "_" + config.tag,
                project=config.logger.project,
                entity=config.logger.entity,
                id=config.logger.id,
                save_dir=config.log_dir,
                config=config,
                offline=config.logger.offline,
                settings=wandb.Settings(start_method="fork"),
            ),
        ]

    if sys.platform == "win32":
        # does not support multi-gpu on windows
        strategy = "dp"
        assert n_gpus == 1
    else:
        strategy = "ddp"

    trainer = Trainer(
        devices=n_gpus,
        accelerator="gpu",
        callbacks=callbacks,
        logger=loggers,
        strategy=strategy,
        **config.trainer
    )

    if config.mode == "train":
        if config.resume and not config.resume_weights_only:
            # FIXME: different behavior in pytorch-lighting>1.9 ?
            checkpoint = torch.load(config.resume, map_location="cpu")
            keys_to_remove = []
            # For testing we do not load occupancy grids
            for k, v in checkpoint["state_dict"].items():
                if "occupancy_grid" in k:
                    keys_to_remove.append(k)
                if "pose_correction" in k:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                checkpoint["state_dict"].pop(k)
            system.load_state_dict(checkpoint["state_dict"], strict=False)

            trainer.fit(system, datamodule=dm, ckpt_path=None)
            # trainer.fit(system, datamodule=dm, ckpt_path=config.resume)
        else:
            trainer.fit(system, datamodule=dm)
        trainer.test(system, datamodule=dm)
    elif config.mode == "validate":
        trainer.validate(system, datamodule=dm, ckpt_path=config.resume)
    elif config.mode == "test":
        checkpoint = torch.load(config.resume, map_location="cpu")
        keys_to_remove = []
        # For testing we do not load occupancy grids
        for k, v in checkpoint["state_dict"].items():
            if "occupancy_grid" in k:
                keys_to_remove.append(k)
            if "pose_correction" in k:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            checkpoint["state_dict"].pop(k)
        system.load_state_dict(checkpoint["state_dict"], strict=False)
        trainer.test(system, datamodule=dm, ckpt_path=None)
    elif config.mode == "predict":
        trainer.predict(system, datamodule=dm, ckpt_path=config.resume)


if __name__ == "__main__":
    main()
