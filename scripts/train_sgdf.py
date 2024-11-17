import os
import random
import numpy as np
import pathlib
import tyro

# pyright: reportGeneralTypeIssues=false


def main(log_wandb: bool = False, gpu_num: int = 0):
    # Need to do this before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    import torch
    import pytorch_lightning as pl
    from centerart_model.sdf.sdf_dataset import SGDFDataset
    from centerart_model.sdf.training_deep_sdf import (
        load_data_dir_config,
        load_object_config,
        load_split_config,
    )
    from centerart_model.sgdf.training_deep_sgdf import (
        LitSGDFModel,
        load_sgdf_config,
        EmbeddingLogger,
    )

    # Seeds
    seed = 12345
    os.environ["PYTHONHASHSEED"] = str(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Configs
    specs, _config = load_sgdf_config()
    dataset_config = load_object_config()
    data_dir_config = load_data_dir_config()
    split_config = load_split_config()

    # Data
    train_set = SGDFDataset(
        points_per_obj=_config["points_per_obj"],
        dataset_dir=data_dir_config["train_data_dir"],
        dataset_config=dataset_config,
        split_config=split_config,
        mode="train",
    )
    valid_set = SGDFDataset(
        points_per_obj=_config["points_per_obj"],
        dataset_dir=data_dir_config["valid_data_dir"],
        dataset_config=dataset_config,
        split_config=split_config,
        mode="valid",
    )

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
    specs["points_per_obj"] = _config["points_per_obj"]
    # specs["num_train_objects"] = train_set.get_num_objects()
    specs["num_train_objects"] = _config["num_train_objects"]
    lit_model = LitSGDFModel(**specs)

    # Loggers
    logger = None
    callbacks = []
    if log_wandb:
        logger = pl.loggers.WandbLogger(project="[CenterArt] SGDFDecoder")
        logger.watch(lit_model)  # type: ignore
        checkpoints_path = (
            pathlib.Path(__file__).parent.parent.resolve()
            / "ckpt_sgdf"
            / logger.version
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoints_path,
            every_n_epochs=specs["ckpt_every_n_epochs"],
            mode="max",
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(EmbeddingLogger(logger))

        logger.experiment.config.update(specs)
        logger.experiment.config.update(_config)

    # Training
    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=1,
        max_epochs=specs["NumEpochs"],
        logger=logger,
        log_every_n_steps=5,
        callbacks=callbacks,
    )

    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )


tyro.cli(main)
