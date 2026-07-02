from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_three_channels(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] > 3:
        return x[:, :3, :, :]
    pad_channels = 3 - x.shape[1]
    pad = x[:, -1:, :, :].repeat(1, pad_channels, 1, 1)
    return torch.cat([x, pad], dim=1)


class _EncoderResidualBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super().__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class _PrimaryEncoderBackbone(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super().__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _build_primary_encoder(**kwargs):
    kwargs.pop("pretrained", None)
    model = _PrimaryEncoderBackbone(_EncoderResidualBlock, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    return model


class _ConvBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class _ContextBridge(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            _ConvBN(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            _ConvBN(in_channel, out_channel, 1),
            _ConvBN(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            _ConvBN(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            _ConvBN(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            _ConvBN(in_channel, out_channel, 1),
            _ConvBN(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            _ConvBN(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            _ConvBN(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            _ConvBN(in_channel, out_channel, 1),
            _ConvBN(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            _ConvBN(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            _ConvBN(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = _ConvBN(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = _ConvBN(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class _DenseMerge(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = _ConvBN(channel, channel, 3, padding=1)
        self.conv_upsample2 = _ConvBN(channel, channel, 3, padding=1)
        self.conv_upsample3 = _ConvBN(channel, channel, 3, padding=1)
        self.conv_upsample4 = _ConvBN(channel, channel, 3, padding=1)
        self.conv_upsample5 = _ConvBN(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = _ConvBN(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = _ConvBN(3*channel, 3*channel, 3, padding=1)
        self.conv4 = _ConvBN(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class _PrimaryAnchorCore(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        self.encoder = _build_primary_encoder(pretrained=False)
        self.rfb2_1 = _ContextBridge(512, channel)
        self.rfb3_1 = _ContextBridge(1024, channel)
        self.rfb4_1 = _ContextBridge(2048, channel)
        self.agg1 = _DenseMerge(channel)
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = _ConvBN(2048, 256, kernel_size=1)
        self.ra4_conv2 = _ConvBN(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = _ConvBN(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = _ConvBN(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = _ConvBN(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = _ConvBN(1024, 64, kernel_size=1)
        self.ra3_conv2 = _ConvBN(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = _ConvBN(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = _ConvBN(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = _ConvBN(512, 64, kernel_size=1)
        self.ra2_conv2 = _ConvBN(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = _ConvBN(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = _ConvBN(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)

        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2


def _binary_logits_from_single_logit(logit: torch.Tensor) -> torch.Tensor:
    if logit.ndim != 4 or logit.shape[1] != 1:
        raise ValueError(f"Expected shape (B,1,H,W), got {tuple(logit.shape)}")
    return torch.cat([-0.5 * logit, 0.5 * logit], dim=1)


class InternalAnchorA(nn.Module):
    adapter_name = 'anchor_primary'

    def __init__(
        self,
        num_classes: int = 2,
        channel: int = 32,
        return_multi_outputs: bool = True,
    ):
        super().__init__()
        if int(num_classes) != 2:
            raise ValueError('InternalAnchorA expects num_classes=2')
        self.model = _PrimaryAnchorCore(channel=int(channel))
        self.return_multi_outputs = bool(return_multi_outputs)
        self.feature_channels = [256, 512, 1024, 2048]
        self.output_kind = 'logits'

    def _forward_backbone_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = _to_three_channels(x)
        m = self.model
        x = m.encoder.conv1(x)
        x = m.encoder.bn1(x)
        x = m.encoder.relu(x)
        x = m.encoder.maxpool(x)
        x1 = m.encoder.layer1(x)
        x2 = m.encoder.layer2(x1)
        x3 = m.encoder.layer3(x2)
        x4 = m.encoder.layer4(x3)
        return [x1, x2, x3, x4]

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self._forward_backbone_features(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_backbone_features(x)[1]

    def forward(self, x: torch.Tensor):
        x = _to_three_channels(x)
        outputs = self.model(x)
        if not isinstance(outputs, (tuple, list)) or len(outputs) == 0:
            raise RuntimeError('InternalAnchorA expects multi-scale outputs')
        logits_2c = []
        for out in outputs:
            if out.shape[-2:] != x.shape[-2:]:
                out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
            logits_2c.append(_binary_logits_from_single_logit(out))
        if self.return_multi_outputs:
            return tuple(logits_2c)
        return logits_2c[-1]


logger = logging.getLogger(__name__)


class InternalAnchorB(nn.Module):
    adapter_name = 'anchor_secondary'

    def __init__(self, in_channels=3, num_classes=2, encoder_name='mit_b0', encoder_weights='imagenet', use_se=False):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except Exception as exc:
            raise ImportError(
                "InternalAnchorB requires `segmentation_models_pytorch` "
                "with MiT encoders available."
            ) from exc

        if encoder_name not in ('mit_b0', 'MiT-B0', 'mit-b0'):
            logger.warning(
                "InternalAnchorB only supports MiT-B0; got %s, fallback to mit_b0.",
                encoder_name,
            )

        if not encoder_weights or str(encoder_weights).lower() in {"none", "null", "false"}:
            encoder_weights = None

        self.output_kind = "logits"
        self.encoder = smp.encoders.get_encoder(
            "mit_b0",
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        encoder_channels = [c for c in getattr(self.encoder, "out_channels", ()) if int(c) > 0]
        if len(encoder_channels) < 4:
            raise RuntimeError(
                f"Unexpected MiT-B0 encoder channels: {getattr(self.encoder, 'out_channels', None)}"
            )
        c1_ch, c2_ch, c3_ch, c4_ch = encoder_channels[-4:]
        self.feature_channels = [int(c1_ch), int(c2_ch), int(c3_ch), int(c4_ch)]

        self.linear_c1 = nn.Conv2d(c1_ch, 256, kernel_size=1)
        self.linear_c2 = nn.Conv2d(c2_ch, 256, kernel_size=1)
        self.linear_c3 = nn.Conv2d(c3_ch, 256, kernel_size=1)
        self.linear_c4 = nn.Conv2d(c4_ch, 256, kernel_size=1)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(256, num_classes, kernel_size=1)

    def _encode_features(self, x):
        features = self.encoder(x)
        features = [feat for feat in features if feat.ndim == 4 and feat.shape[1] > 0]
        if len(features) >= 4:
            features = features[-4:]
        return features

    def forward_features(self, x):
        return self._encode_features(x)

    def extract_features(self, x):
        return self._encode_features(x)[-1]

    def forward(self, x):
        if hasattr(x, 'ndim') and x.ndim > 4:
            x = x.view((-1, x.shape[-3], x.shape[-2], x.shape[-1]))
        features = self._encode_features(x)
        if len(features) < 4:
            raise RuntimeError(f"MiT-B0 encoder returned insufficient features: {len(features)}")
        c1, c2, c3, c4 = features[-4:]
        h, w = c1.shape[-2:]

        p1 = self.linear_c1(c1)
        p2 = F.interpolate(self.linear_c2(c2), size=(h, w), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.linear_c3(c3), size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.linear_c4(c4), size=(h, w), mode='bilinear', align_corners=False)

        logits = self.linear_fuse(torch.cat([p1, p2, p3, p4], dim=1))
        logits = self.dropout(logits)
        logits = self.linear_pred(logits)
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
