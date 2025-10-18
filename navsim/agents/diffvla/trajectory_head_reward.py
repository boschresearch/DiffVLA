import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from navsim.agents.diffvla.modules.blocks import linear_relu_ln, GridSampleCrossBEVAttention
from navsim.agents.diffvla.modules.multimodal_loss import LossComputer
from navsim.common.enums import StateSE2Index
from navsim.agents.diffvla.diffvla_config import DiffvlaConfigV2
from typing import Dict

def compute_reward_loss(
    targets: torch.Tensor,
    predicted_rewards: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    epsilon = 1e-6
    # Compute loss using binary cross-entropy # 5 is the number of metrics
    reward_loss = -torch.mean(
        targets * (predicted_rewards + epsilon).log() + (1 - targets) * (1 - predicted_rewards + epsilon).log()
    ) * weight

    return reward_loss


class RewardHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.heads = nn.ModuleDict({
            "nc": nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 3),
            ),
            "dac": nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            ),
            "ddc": nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 3),
            ),
            "tlc": nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            ),
            "ep": nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            ),
            "tc": nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            ),
            "lk": nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            ),
            "hc": nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            ),
        })
    
    def forward(self, x):
        """
        x: Tensor of shape (B, d_model)
        returns: Dict of outputs for each head
        """
        return {name: head(x) for name, head in self.heads.items()}

class TrajectoryHead(nn.Module):
    def __init__(self, num_poses, d_ffn, d_model, plan_anchor_path, config):
        super().__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._num_h = config.tf_num_head
        self._dropout_rate = config.tf_dropout

        # trajectory anchors
        plan_anchor = np.load(plan_anchor_path)
        self.trajectory_anchors = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32), requires_grad=False
        )  # (256, 8, 3)

        # shared MLPs
        self.mlp_planning_vb = nn.Sequential(
            nn.Linear(num_poses * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.encode_ego_feat_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # bev sampling
        self.bev_sampling = nn.ModuleList([
            GridSampleCrossBEVAttention(d_model, self._num_h, num_points=num_poses, config=config, in_bev_dims=d_model)
            for _ in range(2)
        ])

        # attention and FFNs for 2 stages
        self.cross_bev_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, self._num_h, dropout=self._dropout_rate, batch_first=True)
            for _ in range(2)
        ])
        self.cross_agent_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, self._num_h, dropout=self._dropout_rate, batch_first=True)
            for _ in range(2)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model))
            for _ in range(2)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(6)  # 3 per stage
        ])

        # offset regressors
        self.offset_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_poses * 3)
            ) for _ in range(2)
        ])

        # classification branch
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(d_model, 1, 2), nn.Linear(d_model, 1)
        )

        self.loss_computer = LossComputer(config)
        self.dropout = nn.Dropout(self._dropout_rate)

        # for pdm scores reward
        self.reward_head = RewardHead(dim=d_model)


    def forward(self, agents_query, bev_feature, status_encoding, targets=None):
        """Unified forward interface."""
        if self.training:
            return self.forward_train(agents_query, bev_feature, status_encoding, targets)
        else:
            return self.forward_test(agents_query, bev_feature, status_encoding)

    def _refine_traj(self, anchors, agents_query, bev_feature, status_encoding, stage):
        bs, bev_d, bew_h, bev_w = bev_feature.shape

        t_num, n_pose, pose_dim = self.trajectory_anchors.shape
        feat = self.mlp_planning_vb(anchors)

        status = status_encoding.unsqueeze(1).expand(-1, t_num, -1)
        feat = self.encode_ego_feat_mlp(torch.cat([feat, status], dim=-1))

        # Sampling Traj Points from BEV feature
        bev_feature, feat = self.bev_sampling[stage](feat, self.trajectory_anchors.unsqueeze(0).expand(bs, -1, -1, -1), bev_feature)

        # Cross BEV
        bev_feature = bev_feature.flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)

        feat = self.norms[stage * 3 + 0](self.dropout(
            self.cross_bev_attention[stage](feat, bev_feature, bev_feature)[0]
        ))
        # Cross Agent
        feat = self.norms[stage * 3 + 1](feat + self.dropout(
            self.cross_agent_attention[stage](feat, agents_query, agents_query)[0]
        ))
        # FFN
        feat = self.norms[stage * 3 + 2](self.ffns[stage](feat))

        offset = self.offset_branches[stage](feat).view(bs, t_num, n_pose, pose_dim)
        offset[..., StateSE2Index.HEADING] = offset[..., StateSE2Index.HEADING].tanh() * np.pi

        out = self.trajectory_anchors + offset
        return out, feat

    def forward_train(self, agents_query, bev_feature, status_encoding, targets):
        bs = agents_query.shape[0]
        t_num, n_pose, pose_dim = self.trajectory_anchors.shape

        # print(f'trajectory heading value is {self.trajectory_anchors[:,:,2]}')
        anchor = self.trajectory_anchors.unsqueeze(0).expand(bs, -1, -1, -1).reshape(bs, t_num, -1)

        out1, feat1 = self._refine_traj(anchor, agents_query, bev_feature, status_encoding, stage=0)
        out2, feat2 = self._refine_traj(out1.view(bs, t_num, -1), agents_query, bev_feature, status_encoding, stage=1)
        

        cls_scores = self.plan_cls_branch(feat2).squeeze(-1)
        loss = self.loss_computer(out2, cls_scores, targets, self.trajectory_anchors)

        # pdm score
        # No at-fault Collisions (NC)
        reward = self.reward_head(feat2)
        nc_scores = reward['nc']
        # Drivable Area Compliance (DAC)
        dac_scores = reward['dac'].squeeze(-1)
        dac_scores = torch.sigmoid(dac_scores)
        # Driving Direction Compliance (DDC)
        ddc_scores = reward['ddc'].squeeze(-1)
        # Traffic Light Compliance (TLC)
        tlc_scores = reward['tlc'].squeeze(-1)
        tlc_scores = torch.sigmoid(tlc_scores)
        # ext pdm score
        # Ego Progress (EP)
        ep_scores = reward['ep'].squeeze(-1)
        # Time to Collision (TTC) within bound
        tc_scores = reward['tc'].squeeze(-1)
        tc_scores = torch.sigmoid(tc_scores)
        # Lane Keeping (LK)
        lk_scores = reward['lk'].squeeze(-1)
        lk_scores = torch.sigmoid(lk_scores)
        # History Comfort (HC)
        hc_scores = reward['hc'].squeeze(-1)
        hc_scores = torch.sigmoid(hc_scores)

        # combined_score = self.cal_pdm_scorer_soft(cls_scores, nc_scores,dac_scores,ddc_scores,tlc_scores,ep_scores,tc_scores,lk_scores,hc_scores)
        mode = cls_scores.argmax(dim=-1)[..., None, None, None].expand(-1, -1, n_pose, pose_dim)
        best_traj = torch.gather(out2, 1, mode).squeeze(1)

        return {
            "trajectory": best_traj,
            "trajectory_loss": loss,
            "trajectory_loss_dict": {"trajectory_loss": loss},
            "nc": nc_scores,
            "dac": dac_scores,
            "ddc": ddc_scores,
            "tlc": tlc_scores,
            "ep": ep_scores,
            "tc": tc_scores,
            "lk": lk_scores,
            "hc": hc_scores,
        }

    def forward_test(self, agents_query, bev_feature, status_encoding):
        bs = agents_query.shape[0]
        t_num, n_pose, pose_dim = self.trajectory_anchors.shape

        anchor = self.trajectory_anchors.unsqueeze(0).expand(bs, -1, -1, -1).reshape(bs, t_num, -1)

        out1, feat1 = self._refine_traj(anchor, agents_query, bev_feature, status_encoding, stage=0)
        out2, feat2 = self._refine_traj(out1.view(bs, t_num, -1), agents_query, bev_feature, status_encoding, stage=1)

        # cls scores
        cls_scores = self.plan_cls_branch(feat2).squeeze(-1)

        # pdm score
        # No at-fault Collisions (NC)
        reward = self.reward_head(feat2)
        nc_scores = reward['nc']
        # Drivable Area Compliance (DAC)
        dac_scores = reward['dac'].squeeze(-1)
        dac_scores = torch.sigmoid(dac_scores)
        # Driving Direction Compliance (DDC)
        ddc_scores = reward['ddc'].squeeze(-1)
        # Traffic Light Compliance (TLC)
        tlc_scores = reward['tlc'].squeeze(-1)
        tlc_scores = torch.sigmoid(tlc_scores)
        # ext pdm score
        # Ego Progress (EP)
        ep_scores = reward['ep'].squeeze(-1)
        # Time to Collision (TTC) within bound
        tc_scores = reward['tc'].squeeze(-1)
        tc_scores = torch.sigmoid(tc_scores)
        # Lane Keeping (LK)
        lk_scores = reward['lk'].squeeze(-1)
        lk_scores = torch.sigmoid(lk_scores)
        # History Comfort (HC)
        hc_scores = reward['hc'].squeeze(-1)
        hc_scores = torch.sigmoid(hc_scores)

        # combine pdm and cls for inference
        combined_score = self.cal_pdm_scorer_soft(cls_scores, nc_scores,dac_scores,ddc_scores,tlc_scores,ep_scores,tc_scores,lk_scores,hc_scores)
        mode = combined_score.argmax(dim=-1)[..., None, None, None].expand(-1, -1, n_pose, pose_dim)
        best_traj = torch.gather(out2, 1, mode).squeeze(1)

        return {
            "trajectory": best_traj,
            "nc": nc_scores,
            "dac": dac_scores,
            "ddc": ddc_scores,
            "tlc": tlc_scores,
            "ep": ep_scores,
            "tc": tc_scores,
            "lk": lk_scores,
            "hc": hc_scores,
        }

    
    def cal_pdm_scorer_soft(self, cls_s, nc, dac, ddc, tlc, ep, tc, lk, hc):
        cls_s = cls_s.sigmoid()
        nc = torch.softmax(nc, dim=-1)
        nc = map_three_class_to_confidence(nc)
        ddc = torch.softmax(ddc, dim=-1)
        ddc = map_three_class_to_confidence(ddc)

        w1 = 1.0
        w2 = 4.0
        w3 = 1.2
        w4 = 0.02
        w5 = 8.0

        final_score = w1*cls_s + w2*nc + w3*dac + w4*(5.0*tc + 2*lk) + w5*ddc

        return final_score


def map_three_class_to_confidence(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=-1)
    class_idx = probs.argmax(dim=-1)
    confidence = probs.gather(-1, class_idx.unsqueeze(-1)).squeeze(-1)
    return confidence
