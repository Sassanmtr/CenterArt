#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        joint_out=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        out_dim=1,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        self.joint_out = joint_out
        if len(self.joint_out) == 0:
            dims = [latent_size + 4] + dims + [out_dim]  # 3 for object position
        else:
            dims = (
                [latent_size + 3] + dims + [out_dim]
            )  # 4 = 3 (for object position) + 1 (for joint info)

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                if len(self.joint_out) == 0:
                    out_dim = dims[layer + 1] - dims[0]
                else:
                    out_dim = dims[layer + 1] - dims[0] - 1  # -1 for joint info
            elif layer + 1 in joint_out:
                out_dim = dims[layer + 1] - 1
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),  # type: ignore
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, shape_code, joint_code, xyz):
        if shape_code.shape[1] > 3 and self.latent_dropout:
            latent_vecs = shape_code
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            if len(self.joint_out) == 0:
                x = torch.cat((shape_code, joint_code, xyz), 1)
            else:
                x = torch.cat((shape_code, xyz), 1)

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, shape_code, joint_code, xyz], 1)
            elif layer in self.joint_out:
                x = torch.cat([x, joint_code], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
