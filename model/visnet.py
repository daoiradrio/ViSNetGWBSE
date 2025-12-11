# Code taken from https://github.com/microsoft/AI2BMD/tree/main/src/ViSNet/model

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import grad
from torch_scatter import scatter

from ViSNetGW.model.output_modules import EquivariantScalar
from ViSNetGW.model.visnet_block import ViSNetBlock



def create_model(cfg, prior_model=None, mean=None, std=None):
    visnet_args = dict(
        lmax=cfg.model.lmax,
        vecnorm_type=cfg.model.vecnorm_type,
        trainable_vecnorm=cfg.model.trainable_vecnorm,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        hidden_channels=cfg.model.hidden_channels,
        num_rbf=cfg.model.num_rbf,
        rbf_type=cfg.model.rbf_type,
        trainable_rbf=cfg.model.trainable_rbf,
        activation=cfg.model.activation,
        attn_activation=cfg.model.attn_activation,
        max_z=cfg.model.max_z,
        cutoff=cfg.model.cutoff,
        max_num_neighbors=cfg.model.max_num_neighbors,
    )

    representation_model = ViSNetBlock(**visnet_args)

    output_model = EquivariantScalar(cfg.model.hidden_channels, cfg.model.activation)

    model = ViSNet(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op="mean",
        mean=mean,
        std=std,
    )

    return model



class ViSNet(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
    ):
        super(ViSNet, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.prior_model = prior_model
        self.reduce_op = reduce_op
        self.derivative = derivative
        mean = torch.scalar_tensor(0) if mean is None else mean
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        self.register_buffer("std", std)
        self.reset_parameters()


    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()


    def forward(self, data: dict[str, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        x, v = self.representation_model(data)
        x = self.output_model.pre_reduce(x, v, data['z'], data['pos'], data['batch'])
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, data['z'])

        out = scatter(x, data['batch'], dim=0, reduce=self.reduce_op)
        out = self.output_model.post_reduce(out)

        out = out + self.mean

        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [data['pos']],
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )[0]
            if dy is None:
                raise RuntimeError(
                    "Autograd returned None for the force prediction."
                )
            return out, -dy
        return out, None
