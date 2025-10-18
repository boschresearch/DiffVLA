import torch.utils.checkpoint as checkpoint
from torch import nn
import torch

from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmcv.cnn import build_norm_layer


class CustomResNet(nn.Module):

    def __init__(
        self,
        numC_input,
        num_layer=[2, 2, 2],
        num_channels=None,
        stride=[2, 2, 2],
        backbone_output_ids=None,
        norm_cfg=dict(type="BN"),
        with_cp=False,
        block_type="Basic",
    ):
        super(CustomResNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = (
            [numC_input * 2 ** (i + 1) for i in range(len(num_layer))]
            if num_channels is None
            else num_channels
        )
        self.backbone_output_ids = (
            range(len(num_layer))
            if backbone_output_ids is None
            else backbone_output_ids
        )
        layers = []
        if block_type == "BottleNeck":
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(
                            curr_numC, num_channels[i], 3, stride[i], 1
                        ),
                        norm_cfg=norm_cfg,
                    )
                ]
                curr_numC = num_channels[i]
                layer.extend(
                    [
                        Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                        for _ in range(num_layer[i] - 1)
                    ]
                )
                layers.append(nn.Sequential(*layer))
        elif block_type == "Basic":
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(
                            curr_numC, num_channels[i], 3, stride[i], 1
                        ),
                        norm_cfg=norm_cfg,
                    )
                ]
                curr_numC = num_channels[i]
                layer.extend(
                    [
                        BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                        for _ in range(num_layer[i] - 1)
                    ]
                )
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


class FPN_LSS(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor=4,
        input_feature_index=(0, 2),
        norm_cfg=dict(type="BN"),
        extra_upsample=2,
        lateral=None,
        use_input_conv=False,
    ):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.input_conv = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * channels_factor,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                ),
                build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[
                    1
                ],
                nn.ReLU(inplace=True),
            )
            if use_input_conv
            else None
        )
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample, mode="bilinear", align_corners=True
                ),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        if self.input_conv is not None:
            x = self.input_conv(x)
        x = self.conv(x)
        if self.extra_upsample:
            x = self.up2(x)
        return x
