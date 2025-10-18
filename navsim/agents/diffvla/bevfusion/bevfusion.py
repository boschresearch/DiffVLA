"""
Implements the TransFuser vision backbone.
"""

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .ops import Voxelization
from .sparse_encoder import BEVFusionSparseEncoder
from .depth_lss import DepthLSSTransform, LSSTransform, LSSTransformBEVDepth
from mmdet.models.backbones.resnet import ResNet
from .bevfusion_necks import GeneralizedLSSFPN
from .fusion import ConvFuser
from mmdet3d.models.backbones import SECOND
from mmdet3d.models.necks import SECONDFPN
from .bevdet import CustomResNet, FPN_LSS

from navsim.agents.diffvla.diffvla_config import DiffvlaConfigV2

#add vov99 backbone
from vovnet.vovnet import build_vovnet_backbone
from vovnet.layers.shape_spec import ShapeSpec

class BEVDetEncoder(nn.Module):
    def __init__(self, config: DiffvlaConfigV2):
        super().__init__()

        self._config = config

        v299_cfg={'NORM':'BN','CONV_BODY':'V-99-eSE','FREEZE_AT':0, 'VOVNET_OUT_FEATURES':['stage4', 'stage5']}
        input_shape = ShapeSpec(3)
        self.img_backbone = build_vovnet_backbone(v299_cfg,input_shape)
        # "/data/gyu3nj/navsim_6/exp/backbone_ckpt/vov99.pth"
        pretrained_weights=torch.load(config.v99_pretrained_path)
        backbone_state_dict = {k.replace("backbone.", ""): v for k, v in pretrained_weights['model'].items() if k.startswith("backbone")}
        self.img_backbone.load_state_dict(backbone_state_dict, strict=False)

        print('########################## vov99 ######################### loaded ############################')
        # self.img_backbone = ResNet(**self._config.img_backbone)
        self.img_neck = GeneralizedLSSFPN(**self._config.img_neck)
        self.view_transform = LSSTransform(**self._config.view_transform)
        self.img_bev_encoder_backbone = CustomResNet(
            **self._config.img_bev_encoder_backbone
        )
        self.img_bev_encoder_neck = FPN_LSS(**self._config.img_bev_encoder_neck)

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()
        x = self.img_backbone(x)
        p4,p5 = x['stage4'],x['stage5']
        x = [p4,p5]

        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        # print(f'x before view transform shape is {x.shape}')
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )

        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)

        return x

    def extract_feat(self, batch_inputs_dict, batch_input_metas):
        imgs = batch_inputs_dict.get("imgs", None)  # (B, num_cams, 3, h, w)
        num_cams = imgs.shape[1]  # num_cams
        points = batch_inputs_dict.get("points", None)  # [(num_points, 3)] * B

        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta["lidar2img"])  # (num_cams, 4, 4)
                camera_intrinsics.append(meta["cam2img"])  # (num_cams, 4, 4)
                camera2lidar.append(meta["cam2lidar"])  # (num_cams, 4, 4)
                img_aug_matrix.append(
                    meta.get("img_aug_matrix", [np.eye(4) for _ in range(num_cams)])
                )  # (num_cams, 4, 4)
                lidar_aug_matrix.append(
                    meta.get("lidar_aug_matrix", np.eye(4))
                )  # (4, 4)

            lidar2image = imgs.new_tensor(
                np.asarray(lidar2image)
            )  # (B, num_cams, 4, 4)
            camera_intrinsics = imgs.new_tensor(
                np.array(camera_intrinsics)
            )  # (B, num_cams, 4, 4)
            camera2lidar = imgs.new_tensor(
                np.asarray(camera2lidar)
            )  # (B, num_cams, 4, 4)
            img_aug_matrix = imgs.new_tensor(
                np.asarray(img_aug_matrix)
            )  # (B, num_cams, 4, 4)
            lidar_aug_matrix = imgs.new_tensor(
                np.asarray(lidar_aug_matrix)
            )  # (B, 4, 4)

            img_feature = self.extract_img_feat(
                imgs,
                deepcopy(points),
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
            )
            features.append(img_feature)

        x = features[0] # (B, 512, 128, 128)
        
        return x

    def forward(self, features):
        batch_input_metas = features.pop("metas")
        batch_inputs_dict = features
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        return feats
    
    
