from typing import Dict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from navsim.agents.diffvla.diffvla_config import DiffvlaConfigV2
from navsim.agents.diffvla.diffvla_features import BoundingBox2DIndex

from torch.distributions.laplace import Laplace
import torch.nn as nn

def batch_box_to_corners_with_zeros(gt_states: torch.Tensor) -> torch.Tensor:
    """
    Convert each [x, y, heading, length, width] row into [x, y, corner1_x, corner1_y, ..., corner4_x, corner4_y]
    Keep all-zero rows as all-zero in output. Output shape is (N, 10)
    """
    N = gt_states.shape[0]
    output = torch.zeros((N, 10), dtype=gt_states.dtype, device=gt_states.device)

    valid_mask = gt_states.abs().sum(dim=1) > 0
    if valid_mask.sum() == 0:
        return output
    
    boxes = gt_states[valid_mask]  # shape: (M, 5)
    x, y, heading, length, width = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    dx = length / 2
    dy = width / 2

    corners_local = torch.tensor([
        [1, 1],
        [1, -1],
        [-1, -1],
        [-1, 1],
    ], dtype=boxes.dtype, device=boxes.device)  # shape: (4, 2)

    corners_scaled = torch.stack([dx, dy], dim=1).unsqueeze(1) * corners_local.unsqueeze(0)  # (M, 4, 2)

    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    R = torch.stack([
        torch.stack([cos_h, -sin_h], dim=1),
        torch.stack([sin_h,  cos_h], dim=1)
    ], dim=1)  # (M, 2, 2)

    rotated_corners = torch.bmm(corners_scaled, R)  # (M, 4, 2)
    translated_corners = rotated_corners + boxes[:, None, 0:2]  # (M, 4, 2)

    centers = boxes[:, 0:2].unsqueeze(1)  # (M, 1, 2)
    all_points = torch.cat([centers, translated_corners], dim=1)  # (M, 5, 2)
    result = all_points.reshape(-1, 10)  # (M, 10)

    output[valid_mask] = result
    return output

# for pdm rewards items using BCE loss
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


def diffvla_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: DiffvlaConfigV2
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """

    if "trajectory_loss" in predictions:
        trajectory_loss = predictions["trajectory_loss"]
    else:
        trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    agent_class_loss, agent_box_loss, agent_uncer_loss = _agent_loss_uncer(targets, predictions, config)
    bev_semantic_loss = F.cross_entropy(
        predictions["bev_semantic_map"], targets["bev_semantic_map"].long()
    )

    # for pdm reward
    reward_loss = 0.0
    if "dac" in predictions:
        reward_loss += compute_reward_loss(predictions["dac"], targets["dac"], weight=config.dac_weight)
    if "tlc" in predictions:
        reward_loss += compute_reward_loss(predictions["tlc"], targets["tlc"], weight=config.tlc_weight)
    if "nc" in predictions:
        bs = predictions["nc"].size(0)
        n_traj = predictions["nc"].size(1)
        nc_targets = (targets["nc"] * 2).long()
        all_nc_loss= F.cross_entropy(predictions["nc"].view(-1, 3), nc_targets.view(-1),reduction='none')
        all_nc_loss = all_nc_loss.view(bs, n_traj)
        per_sample_nc_loss = all_nc_loss.mean(dim=1)
        nc_loss = per_sample_nc_loss.mean()
        reward_loss += nc_loss * config.nc_weight
    if "ddc" in predictions:
        bs = predictions["ddc"].size(0)
        n_traj = predictions["ddc"].size(1)
        ddc_targets = (targets["ddc"] * 2).long()
        all_ddc_loss= F.cross_entropy(predictions["ddc"].view(-1, 3), ddc_targets.view(-1),reduction='none')
        all_ddc_loss = all_ddc_loss.view(bs, n_traj)
        per_sample_ddc_loss = all_ddc_loss.mean(dim=1)
        ddc_loss = per_sample_ddc_loss.mean()
        reward_loss += ddc_loss * config.ddc_weight
    if "ep" in predictions:
        reward_loss += config.ep_weight * F.mse_loss(predictions["ep"], targets["ep"])
    if "tc" in predictions:
        reward_loss += compute_reward_loss(predictions["tc"], targets["tc"], weight=config.tc_weight)
    if "lk" in predictions:
        reward_loss += compute_reward_loss(predictions["lk"], targets["lk"], weight=config.lk_weight)
    if "hc" in predictions:
        reward_loss += compute_reward_loss(predictions["hc"], targets["hc"], weight=config.hc_weight)

    loss = (
        config.trajectory_weight * trajectory_loss
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
        # add reward loss
        + config.reward_weight * reward_loss
        # add uncer loss
        + config.uncer_weight * agent_uncer_loss
    )
    loss_dict = {
        'loss': loss,
        'trajectory_loss': config.trajectory_weight*trajectory_loss,
        'agent_class_loss': config.agent_class_weight*agent_class_loss,
        'agent_box_loss': config.agent_box_weight*agent_box_loss,
        'bev_semantic_loss': config.bev_semantic_weight*bev_semantic_loss,
        # add reward loss
        'reward_loss': config.reward_weight*reward_loss,
        # add uncer loss
        'agent_uncer_loss': config.uncer_weight*agent_uncer_loss,
    }
    if "trajectory_loss_dict" in predictions:
        trajectory_loss_dict = predictions["trajectory_loss_dict"]
        loss_dict.update(trajectory_loss_dict)

    return loss_dict


# def _agent_loss(
#     targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfigV2
# ):
#     """
#     Hungarian matching loss for agent detection
#     :param targets: dictionary of name tensor pairings
#     :param predictions: dictionary of name tensor pairings
#     :param config: global Transfuser config
#     :return: detection loss
#     """

#     gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
#     pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

#     if config.latent:
#         rad_to_ego = torch.arctan2(
#             gt_states[..., BoundingBox2DIndex.Y],
#             gt_states[..., BoundingBox2DIndex.X],
#         )

#         in_latent_rad_thresh = torch.logical_and(
#             -config.latent_rad_thresh <= rad_to_ego,
#             rad_to_ego <= config.latent_rad_thresh,
#         )
#         gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

#     # save constants
#     batch_dim, num_instances = pred_states.shape[:2]
#     num_gt_instances = gt_valid.sum()
#     num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

#     ce_cost = _get_ce_cost(gt_valid, pred_logits)
#     l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

#     cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
#     cost = cost.cpu()

#     indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
#     matching = [
#         (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#         for i, j in indices
#     ]
#     idx = _get_src_permutation_idx(matching)

#     pred_states_idx = pred_states[idx]
#     gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

#     pred_valid_idx = pred_logits[idx]
#     gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

#     l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
#     l1_loss = l1_loss.sum(-1) * gt_valid_idx
#     l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

#     ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
#     ce_loss = ce_loss.view(batch_dim, -1).mean()

#     return ce_loss, l1_loss

def _agent_loss_uncer(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: DiffvlaConfigV2
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_corners, pred_logits = predictions["agent_states"], predictions["agent_corners"], predictions["agent_labels"]

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    pred_corners_idx = pred_corners[idx]
    xy_means = pred_corners_idx[...,:10]
    xy_bs = pred_corners_idx[...,10:]

    xy_target = batch_box_to_corners_with_zeros(gt_states_idx)

    eps = 1e-3
    m_xy = Laplace(xy_means, xy_bs.clamp(min=eps))            
    log_prob_xy = m_xy.log_prob(xy_target)
    nll_loss_xy = -log_prob_xy

    nll_loss = nll_loss_xy.mean(-1).mean()
    # nll_loss = (nll_loss_xy.mean(-1) * gt_valid_idx).sum() / gt_valid_idx.sum()
    nll_loss = torch.clamp(nll_loss, min=1e-3, max=5)

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss, nll_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
