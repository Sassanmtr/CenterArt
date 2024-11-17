import os
import tyro
import random
import numpy as np
from typing import Optional

# pyright: reportGeneralTypeIssues=false


def main(
    log_wandb: bool = False,
    gpu_num: int = 0,
    seed: int = 12345,
    resume_ckpt: Optional[str] = None,
):
    # Need to do this before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from centerart_model.utils.configs import Directories
    from centerart_model.rgb.training_centerart import LitCenterArtModel, load_rgb_config
    from centerart_model.rgb.rgb_data import RGBDataset

    # Seeds
    os.environ["PYTHONHASHSEED"] = str(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get args
    specs, _config = load_rgb_config()

    # Data
    train_set = RGBDataset(specs["EmbeddingCkptPath"], mode="train")
    valid_set = RGBDataset(specs["EmbeddingCkptPath"], mode="valid")
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=_config["num_workers"],
        batch_size=_config["batch_size"],
        shuffle=True,
        persistent_workers=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        num_workers=_config["num_workers"],
        batch_size=_config["batch_size"],
        shuffle=False,
        persistent_workers=True,
    )

    # Model
    lit_model = LitCenterArtModel(**specs)

    # Loggers
    logger = False
    enable_checkpointing = False
    callbacks = []
    if log_wandb:
        logger = WandbLogger(project="[CenterArt] RGB")
        logger.watch(lit_model)  # type: ignore
        enable_checkpointing = True
        checkpoints_path = Directories.ROOT / "ckpt_rgb" / logger.version
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoints_path, every_n_epochs=1
        )
        callbacks.append(checkpoint_callback)
        logger.experiment.config.update(specs)
        logger.experiment.config.update(_config)
        logger.experiment.config.update({"seed": seed})

    # Resume Ckpt
    resume_ckpt_path = (
        Directories.ROOT / "ckpt_rgb" / resume_ckpt if resume_ckpt is not None else None
    )

    # Training
    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=1,
        max_epochs=specs["NumEpochs"],
        logger=logger,
        enable_checkpointing=enable_checkpointing,
        callbacks=callbacks,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=resume_ckpt_path,
    )


tyro.cli(main)
