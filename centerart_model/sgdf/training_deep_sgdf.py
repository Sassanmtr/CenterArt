import json
import pathlib
import numpy as np
from typing import List
import wandb
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import centerart_model.gdf.gdf_utils as gdf_utils
from centerart_model.sdf.deep_sdf_decoder import Decoder
from centerart_model.utils.data_utils import get_checkpoint_path
from centerart_model.utils.lr_schedules import adjust_learning_rate, get_learning_rate_schedules
from centerart_model import DEVICE


def load_sgdf_config():
    specs_path = (
        pathlib.Path(__file__).parents[2].resolve() / "configs/sgdf_train_specs.json"
    )
    specs = json.load(open(specs_path))
    _config_path = (
        pathlib.Path(__file__).parents[2].resolve() / "configs/sgdf_train_config.json"
    )
    if not _config_path.is_file():
        _config_path = (
            pathlib.Path(__file__).parents[2].resolve()
            / "configs/sgdf_train_config_default.json"
        )
    _config = json.load(open(_config_path))
    return specs, _config


def load_sgdf_model(
    ckpt_name: str, ckpt_folder_name: str = "ckpt_sgdf", overwrite_specs: bool = False
):
    ckpt_path = get_checkpoint_path(ckpt_name, ckpt_folder_name)
    sgdf_specs, _ = load_sgdf_config()
    lit_model = LitSGDFModel.load_from_checkpoint(ckpt_path, map_location="cpu").to(
        DEVICE
    )
    if overwrite_specs:
        lit_model.specs = sgdf_specs
    return lit_model, sgdf_specs


def load_embeddings(embedding_ckpt_path, ckpt_folder_name: str = "ckpt_sgdf"):
    with torch.no_grad():
        lit_model, _ = load_sgdf_model(embedding_ckpt_path, ckpt_folder_name=ckpt_folder_name)  # type: ignore
        embeddings_matrix = lit_model.embeddings.weight.clone()
    return embeddings_matrix


class LitSGDFModel(pl.LightningModule):
    def __init__(self, **specs):
        super().__init__()
        self.save_hyperparameters()
        self.decoder = Decoder(
            latent_size=specs["CodeLength"], out_dim=10, **specs["NetworkSpecs"]
        )
        self.embeddings = torch.nn.Embedding(
            num_embeddings=specs["num_train_objects"],
            embedding_dim=specs["CodeLength"],
            max_norm=specs["CodeBound"],
        )
        # Initialize Embeddings with zero mean and std based on latent size
        torch.nn.init.normal_(
            self.embeddings.weight.data,
            0.0,
            1.0 / np.sqrt(specs["CodeLength"]),
        )
        self.specs = specs
        return

    def clamp_sdf(self, sdf):
        clamp_dist = self.specs["ClampingDistance"]
        if clamp_dist == 0.0:
            return sdf
        minT = -clamp_dist
        maxT = clamp_dist
        clamped_sdf = torch.clamp(sdf, minT, maxT)
        return clamped_sdf

    def on_train_epoch_start(self):
        # print("New Epoch started!")
        optimizers = self.optimizers()
        if not isinstance(optimizers, List):
            optimizers = [optimizers]
        for optimizer in optimizers:
            adjust_learning_rate(optimizer, self.current_epoch)
            for param_group in optimizer.param_groups:
                self.log(f"lr/{param_group['target']}", param_group["lr"])
        return

    def get_codesize_loss(self, embeddings):
        if not self.specs["CodeRegularization"]:
            return 0.0
        code_reg_order = self.specs["CodeRegularizationOrder"]
        l2_codesize_loss = torch.mean(
            torch.linalg.vector_norm(embeddings, dim=1, ord=code_reg_order)
        )
        reg_loss = min(1, self.current_epoch / 5) * l2_codesize_loss
        return reg_loss

    def vec_projection(self, v, e):
        proj = torch.sum(v * e, dim=-1, keepdim=True) * e
        return proj

    def grahm_schmidt_torch(self, v1, v2):
        ## Obtain rotation matrix from two vectors (grasp orientation from z_1, and z_2)
        u1 = v1
        e1 = u1 / torch.norm(u1, dim=1, keepdim=True)
        u2 = v2 - self.vec_projection(v2, e1)
        e2 = u2 / torch.norm(u2, dim=1, keepdim=True)
        ## Previous version
        e3 = torch.cross(e1, e2, dim=1)
        rot_matrix = torch.cat(
            [e1.unsqueeze(dim=-1), e2.unsqueeze(dim=-1), e3.unsqueeze(dim=-1)], dim=2
        )
        ## Currect version
        # e3 = torch.cross(e2, e1, dim=1)
        # rot_matrix = torch.cat(
        #     [e1.unsqueeze(dim=-1), e3.unsqueeze(dim=-1), e2.unsqueeze(dim=-1)], dim=2
        # )
        # return rot_matrix
        ## Working version
        # e3 = torch.cross(e1, e2, dim=1)
        # rot_matrix = torch.cat(
        #     [e2.unsqueeze(dim=-1), e1.unsqueeze(dim=-1), e3.unsqueeze(dim=-1)], dim=2
        # )
        return rot_matrix

    def postprocess_out(self, sgdf_out_raw, points):
        sdf = sgdf_out_raw[:, 0]  # sdf value
        delta_xyz = sgdf_out_raw[:, 1:4]  # grasp position offset
        z_1 = sgdf_out_raw[:, 4:7]  # grasp orientation z_1
        z_2 = sgdf_out_raw[:, 7:10]  # grasp orientation z_2
        grasp_position = points + delta_xyz
        # Grahm Schmidt
        grasp_rot = self.grahm_schmidt_torch(z_1, z_2)
        grasp_pose = torch.cat([grasp_rot, grasp_position.unsqueeze(dim=-1)], dim=2)
        homog_vector = (
            torch.tensor([0, 0, 0, 1])
            .to(grasp_pose.device)
            .expand((grasp_pose.shape[0], 1, 4))
        )
        grasp_pose = torch.cat([grasp_pose, homog_vector], dim=1)
        return sdf, grasp_pose

    def v_vec_from_grasp(self, grasp_pose):
        # v in grasp frame
        gf_v = torch.tensor(gdf_utils.get_gf_v_vec(), dtype=torch.float32)
        gf_v = torch.cat([gf_v, torch.ones((gf_v.shape[0], 1))], dim=1).to(DEVICE)
        # v in world frame
        wf_v = torch.matmul(grasp_pose.unsqueeze(1), gf_v.unsqueeze(-1)).squeeze(-1)
        return wf_v[..., :3]

    def get_sgdf_loss(self, embeddings, sdf, grasp_poses, sdf_gt, gdf_gt):
        sdf_gt = torch.reshape(sdf_gt, (-1,))
        gdf_gt = torch.reshape(gdf_gt, (-1, 5, 3))
        v_vec = self.v_vec_from_grasp(grasp_poses)
        v_vec_alternative = torch.clone(v_vec)
        v_vec_alternative[:, [1, 2]] = v_vec_alternative[:, [2, 1]]
        v_vec_alternative[:, [3, 4]] = v_vec_alternative[:, [4, 3]]
        diff1 = v_vec - gdf_gt
        diff2 = v_vec_alternative - gdf_gt
        dist1 = torch.linalg.vector_norm(diff1, dim=(1, 2))
        dist2 = torch.linalg.vector_norm(diff2, dim=(1, 2))
        dist = torch.minimum(dist1, dist2)
        grasp_loss = torch.mean(dist)
        sdf_loss = F.l1_loss(sdf, sdf_gt, reduction="mean")
        codesize_loss = self.get_codesize_loss(embeddings)
        tot_loss = (
            self.specs["grasp_loss_w"] * grasp_loss
            + self.specs["sdf_loss_w"] * sdf_loss
            + self.specs["codesize_loss_w"] * codesize_loss
        )
        return grasp_loss, sdf_loss, codesize_loss, tot_loss

    def log_loss(self, grasp_loss, sdf_loss, codesize_loss, tot_loss, prefix):
        if self.logger is not None:
            llog = {}
            llog[f"{prefix}/loss/total"] = tot_loss
            llog[f"{prefix}/loss/grasp"] = grasp_loss
            llog[f"{prefix}/loss/sdf"] = sdf_loss
            llog[f"{prefix}/loss/codesize"] = codesize_loss
            self.logger.experiment.log(llog)  # type: ignore
        return

    def predict(
        self, points, embeddings_expanded, joint_code_expanded, safe: bool = True
    ):
        B, num_points, _ = points.size()
        max_points = self.specs["max_points"]

        safe = safe and num_points * B > max_points
        if safe:
            points = points.cpu()
            embeddings_expanded = embeddings_expanded.cpu()

        # # Concatenate shape embeddings, joint code, and points
        # x = torch.cat([embeddings_expanded, joint_code_expanded, points], dim=2)
        # x = torch.reshape(x, (-1, x.shape[-1]))
        # points = torch.reshape(points, (-1, 3))

        # Feed the network inputs without concatenation
        embeddings_expanded = torch.reshape(
            embeddings_expanded, (-1, embeddings_expanded.shape[-1])
        )
        joint_code_expanded = torch.reshape(
            joint_code_expanded, (-1, joint_code_expanded.shape[-1])
        )
        points = torch.reshape(points, (-1, 3))

        if safe:
            # x_chunks = torch.split(x, max_points)
            embeddings_expanded_chunks = torch.split(embeddings_expanded, max_points)
            joint_code_expanded_chunks = torch.split(joint_code_expanded, max_points)
            points_chunks = torch.split(points, max_points)

            gdf_out_raw = torch.zeros((embeddings_expanded.shape[0], 10)).cpu()
            for i, embeddings_chunk in enumerate(embeddings_expanded_chunks):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                gdf_out_chunk = self.decoder(
                    embeddings_chunk.to(DEVICE),
                    joint_code_expanded_chunks[i].to(DEVICE),
                    points_chunks[i].to(DEVICE),
                )
                gdf_out_raw[
                    i * max_points : (i + 1) * max_points, ...
                ] = gdf_out_chunk.cpu()
                del gdf_out_chunk
        else:
            gdf_out_raw = self.decoder(
                embeddings_expanded.to(DEVICE),
                joint_code_expanded.to(DEVICE),
                points.to(DEVICE),
            )
        sdf, grasp_poses = self.postprocess_out(gdf_out_raw, points)
        return sdf.to(DEVICE), grasp_poses.to(DEVICE)

    def do_step(
        self,
        embeddings,
        embeddings_expanded,
        joint_code_expanded,
        points,
        sdf_gt,
        gdf_gt,
        mode,
    ):
        # Forward pass
        sdf_out, grasp_poses_out = self.predict(
            points, embeddings_expanded, joint_code_expanded, safe=False
        )
        # Do clamping
        sdf_out = self.clamp_sdf(sdf_out)
        sdf_gt = self.clamp_sdf(sdf_gt)
        # Calculate loss
        grasp_loss, sdf_loss, codesize_loss, tot_loss = self.get_sgdf_loss(
            embeddings, sdf_out, grasp_poses_out, sdf_gt, gdf_gt
        )
        # Logging
        self.log(f"{mode}/grasp_loss", grasp_loss)
        self.log(f"{mode}/sdf_loss", sdf_loss)
        self.log(f"{mode}/codesize_loss", codesize_loss)
        self.log(f"{mode}/total_loss", tot_loss)

        # self.log_loss(grasp_loss, sdf_loss, codesize_loss, tot_loss, mode)
        return tot_loss, sdf_out

    def training_step(self, batch, batch_idx):
        obj_idx, points, sdf_gt, gdf_gt, joint_code = batch
        # Get embeddings
        embeddings = self.embeddings(obj_idx)
        embeddings_expanded = embeddings.unsqueeze(1).expand(-1, points.shape[1], -1)
        joint_code_expanded = joint_code.unsqueeze(1).expand(-1, points.shape[1], -1)

        # Augmentation
        if self.specs["points_random_scale"] > 0:
            points = (
                torch.randn_like(points) * self.specs["points_random_scale"] + points
            )
        if self.specs["embeddings_random_scale"] > 0:
            embeddings_expanded = (
                torch.randn_like(embeddings_expanded)
                * self.specs["embeddings_random_scale"]
                + embeddings_expanded
            )
        loss, _ = self.do_step(
            embeddings,
            embeddings_expanded,
            joint_code_expanded,
            points,
            sdf_gt,
            gdf_gt,
            "train",
        )
        return loss

    def validation_step(self, batch, batch_idx):
        obj_idx, points, sdf_gt, gdf_gt, joint_code = batch
        # Get embeddings
        embeddings = self.embeddings(obj_idx)
        embeddings_expanded = embeddings.unsqueeze(1).expand(-1, points.shape[1], -1)
        joint_code_expanded = joint_code.unsqueeze(1).expand(-1, points.shape[1], -1)

        loss, sdf_out = self.do_step(
            embeddings,
            embeddings_expanded,
            joint_code_expanded,
            points,
            sdf_gt,
            gdf_gt,
            "valid",
        )
        if self.logger is not None:
            self.log("valid/mean_abs_sdf", torch.mean(torch.abs(sdf_out)))
        return loss

    def configure_optimizers(self):
        lr_schedules = get_learning_rate_schedules(self.specs)
        optimizer = torch.optim.Adam(
            [
                {
                    "target": "network",
                    "params": self.decoder.parameters(),
                    "scheduler": lr_schedules[0],
                },
                {
                    "target": "embedding",
                    "params": self.embeddings.parameters(),
                    "scheduler": lr_schedules[1],
                },
            ]
        )
        return optimizer


class EmbeddingLogger(pl.Callback):
    def __init__(self, logger: pl.loggers.WandbLogger):
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module: LitSGDFModel):
        print(f"Logging Embeddings")
        raw_embeddings = pl_module.embeddings.weight.data.cpu().detach().numpy()
        self.logger.log_table(
            # Columns = dimension of the embedding vector
            "embedding",
            columns=[f"D{i}" for i in range(raw_embeddings.shape[1])],
            data=raw_embeddings,
        )
