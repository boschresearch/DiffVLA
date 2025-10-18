from dataclasses import dataclass, field
from typing import Tuple, List

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

import os

navsim_exp_root = os.environ.get("NAVSIM_EXP_ROOT")

@dataclass
class DiffvlaConfigV2:
    """Global TransFuser config."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    with_vlm = False
    
    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    bkb_path: str = "%s/bkb_path/pytorch_model.bin" % (navsim_exp_root)

    v99_pretrained_path: str = "%s/ckpt/vov99.pth" % (
        navsim_exp_root
    )

    #for vlm cmd
    vlm_json_path:str = "%s/vlm/vlm_cmd.json" % (
        navsim_exp_root
    )

    # for pdm score
    num_voc = 8192
    plan_anchor_path = f"{navsim_exp_root}/planning_vb/trajectory_anchors_{num_voc}.npy"
    pdm_pkl_path = f"{navsim_exp_root}/pdm_scores_{num_voc}"

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    lidar_min_x: float = -32.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0
    lidar_min_z: float = -5.0
    lidar_max_z: float = 3.0

    cam_names: list = field(
        default_factory=lambda: ["f0", "l0", "l1", "r0", "r1", "b0"]
    )

    # img_neck
    img_neck = dict(
        in_channels=[768,1024],
        out_channels=256,
        start_level=0,
        num_outs=1,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        act_cfg=dict(type="ReLU", inplace=True),
        upsample_cfg=dict(mode="bilinear", align_corners=False),
    )

    # view_transform
    view_transform = dict(
        in_channels=256,
        out_channels=64,
        image_size=[256, 704],
        feature_size=[16, 44],
        xbound=[-32.0, 32.0, 0.5],
        ybound=[-32.0, 32.0, 0.5],
        zbound=[-3, 5, 8],
        dbound=[1.0, 60.0, 0.5],
        downsample=1,
    )
    
    # img_bev_encoder_backbone
    img_bev_encoder_backbone = dict(
        numC_input=64,
        num_channels=[128, 256, 512]
    )
    
    # img_bev_encoder_neck
    img_bev_encoder_neck = dict(
        in_channels=512+128,
        out_channels=256,
    )
    
    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True
    
    # bev grid
    bev_w = 128
    bev_h = 128
    
    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 512
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    diff_loss_weight: float = 20.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    # for reward loss
    reward_weight: float = 14.0

    dac_weight: float = 0.4
    tlc_weight: float = 0.1
    nc_weight: float = 0.1
    ddc_weight: float = 0.1
    ep_weight:float = 0.1
    tc_weight:float = 0.2
    lk_weight:float = 0.2
    hc_weight:float = 0.2

    use_ema: bool = False
    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: (
            "linestring",
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        ),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    lidar_resolution_width = 256
    lidar_resolution_height = 256

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = tf_d_model
    bev_upsample_factor: int = 2

    # optmizer
    weight_decay: float = 1e-4
    lr_steps = [70]
    optimizer_type = "AdamW"
    scheduler_type = "MultiStepLR"
    cfg_lr_mult = 0.5
    opt_paramwise_cfg = {"name": {"image_encoder": {"lr_mult": cfg_lr_mult}}}

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [
            self.lidar_min_x,
            self.lidar_max_x,
            self.lidar_min_y,
            self.lidar_max_y,
        ]
        return max([abs(value) for value in values])
