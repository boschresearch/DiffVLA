from typing import Any, List, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import copy
import torch.nn.functional as F

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.diffvla.diffvla_config import DiffvlaConfigV2
from navsim.agents.diffvla.bevfusion.bevfusion import BEVDetEncoder
from navsim.agents.diffvla.diffvla_features import BoundingBox2DIndex
from navsim.agents.diffvla.trajectory_head_reward import TrajectoryHead

def ensure_batch_dim(x, dim=3):
    return x if x.ndim >= dim else x.unsqueeze(0)

class PartialStatusGater(nn.Module):
    def __init__(self, semantic_dim, gate_dim=20, total_dim=52):
        super().__init__()
        assert gate_dim <= total_dim
        self.gate_dim = gate_dim
        self.total_dim = total_dim
        self.attn_fc = nn.Linear(semantic_dim, 1)
        self.gate_fc = nn.Linear(semantic_dim, gate_dim)

        nn.init.constant_(self.gate_fc.bias, 1.0)

    def forward(self, semantic, status):
        """
        semantic: [B, semantic_dim]
        status:   [B, total_dim]
        """
        attn_scores = self.attn_fc(semantic)  # [B, N, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N, 1]
        semantic_global = (semantic * attn_weights).sum(dim=1)
        gate = torch.sigmoid(self.gate_fc(semantic_global))

        status_front = status[:, :self.gate_dim] * gate
        status_back = status[:, self.gate_dim:]

        return torch.cat([status_front, status_back], dim=-1)


class DiffvlaModelV2(nn.Module):
    """Torch module for Transfuser."""

    def __init__(
        self, trajectory_sampling: TrajectorySampling, config: DiffvlaConfigV2
    ):
        """
        Initializes TransFuser torch module.
        :param trajectory_sampling: trajectory sampling specification.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]
        self._config = config

        self.vlm_flag = config.with_vlm

        tf_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
            )


        self._backbone = BEVDetEncoder(config)

        self._keyval_embedding = nn.Embedding(
            config.bev_h * config.bev_w + 1, config.tf_d_model
        )  # bev_h*bev_w feature grid + trajectory
        self._query_embedding = nn.Embedding(
            sum(self._query_splits), config.tf_d_model
        )  # (30+1, 512)

        if self.vlm_flag:
            self._status_encoding = nn.Linear(32 + 11 + 9, config.tf_d_model)
        else:
            self._status_encoding = nn.Linear(32, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHeadUncer(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )

        # uncertainty is part of the paper named UniUncer which is uncer review for the submission of ICRA2026
        self.uncer_states_encoder = nn.Linear(20, config.tf_d_model)
        self.uncer_encoder = nn.MultiheadAttention(embed_dim=config.tf_d_model, num_heads=8, batch_first=True)
        if self.vlm_flag:
            self.uncer_gater = PartialStatusGater(config.tf_d_model, gate_dim=52, total_dim=52)
        else:
            self.uncer_gater = PartialStatusGater(config.tf_d_model, gate_dim=32, total_dim=32)

    def forward(
        self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature = self._backbone(features)  # [(B, 512, 128, 128)]

        if isinstance(bev_feature, list):
            bev_feature = bev_feature[0]  # (B, 512, 128, 128)

        bev_spatial_shape = bev_feature.shape[2:]  # (128, 128)

        bev_semantic_map = self._bev_semantic_head(bev_feature)


        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        # ego status encoding
        status_encoding = self._status_encoding(status_feature)  # (B, 512)

        keyval = torch.concatenate(
            [bev_feature.flatten(-2, -1).permute(0, 2, 1), status_encoding[:, None]],
            dim=1,
        )  # (B, 128*128+1, 512)
        keyval += self._keyval_embedding.weight[None, ...]  # (B, 128*128+1, 512)

        query = self._query_embedding.weight[None, ...].repeat(
            batch_size, 1, 1
        )  # (B, 31, 512)
        query_out = self._tf_decoder(query, keyval)  # (B, 31, 512)
        _, agents_query = query_out.split(self._query_splits, dim=1)

        # agent 1st
        agents = self._agent_head(agents_query)
        output.update(agents)
        agent_uncer = agents['agent_corners']
        uncer_encode = self.uncer_states_encoder(agent_uncer)
        uncer_agents_query, _ = self.uncer_encoder(query=agents_query, key=uncer_encode, value=uncer_encode)

        # ego status encoding 2nd
        masked_status_feature = self.uncer_gater(uncer_agents_query, status_feature)
        status_encoding = self._status_encoding(masked_status_feature)  # (B, 512)

        # traj 3rd
        trajectory = self._trajectory_head(uncer_agents_query, bev_feature, status_encoding, targets=targets)

        output.update(trajectory)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = (
            agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        )
        agent_states[..., BoundingBox2DIndex.HEADING] = (
            agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi
        )

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class AgentHeadUncer(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHeadUncer, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self.mapping_layers = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, self._d_model),
            nn.ReLU(),
        )

        self._mlp_corners_mean = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, 10),
        )

        self._mlp_corners_b = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, 10),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        uncer_agnet_queries = self.mapping_layers(agent_queries)
        agent_corners_mean = self._mlp_corners_mean(uncer_agnet_queries)
        agent_corners_b = self._mlp_corners_b(uncer_agnet_queries)
        agent_corners_b = torch.clamp(F.softplus(agent_corners_b), min=1e-5, max=10)

        agent_corners = torch.cat([agent_corners_mean, agent_corners_b], dim=-1)
        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels, "agent_corners":agent_corners}