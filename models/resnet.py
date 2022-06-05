import torch
import torch.nn as nn
from torchvision import models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.strides = [8, 16, 32]
        self.num_channels = [512, 1024, 2048]

    def forward(self, inputs):
        x_original = self.conv_original_size0(inputs)
        x_original = self.conv_original_size1(x_original)
        layer0 = self.layer0(inputs)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        xs = {"0": layer2, "1": layer3, "2": layer4}
        all_feats = {'layer0': layer0, 'layer1': layer1, 'layer2': layer2,
                     'layer3': layer3, 'layer4': layer4, 'x_original': x_original}

        mask = torch.zeros(inputs.shape)[:, 0, :, :].to(layer4.device)
        return xs, mask, all_feats

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # fix all bn layers
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)


class ResNetUNet(nn.Module):
    def __init__(self, n_class, out_dim=None, ms_feat=False):
        super().__init__()

        self.return_ms_feat = ms_feat
        self.out_dim = out_dim

        self.base_model = models.resnet50(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        # self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        # self.layer1_1x1 = convrelu(256, 256, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        # self.layer2_1x1 = convrelu(512, 512, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        # self.layer3_1x1 = convrelu(1024, 1024, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        # self.layer4_1x1 = convrelu(2048, 2048, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(1024 + 2048, 1024, 3, 1)
        self.conv_up2 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        # self.conv_up1 = convrelu(512, 256, 3, 1)
        # self.conv_up0 = convrelu(256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        # self.conv_last = nn.Conv2d(128, n_class, 1)
        self.conv_last = nn.Conv2d(64, n_class, 1)
        if out_dim:
            self.conv_out = nn.Conv2d(64, out_dim, 1)
            # self.conv_out = nn.Conv2d(128, out_dim, 1)

        # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.strides = [8, 16, 32]
        self.num_channels = [512, 1024, 2048]

    def forward(self, inputs):
        x_original = self.conv_original_size0(inputs)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(inputs)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        # layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        layer3_up = x

        x = self.upsample(x)
        # layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        layer2_up = x

        x = self.upsample(x)
        # layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        # layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        out = out.sigmoid().squeeze(1)

        # xs = {"0": layer2, "1": layer3, "2": layer4}
        xs = {"0": layer2_up, "1": layer3_up, "2": layer4}
        mask = torch.zeros(inputs.shape)[:, 0, :, :].to(layer4.device)
        # ms_feats = self.ms_feat(xs, mask)

        if self.return_ms_feat:
            if self.out_dim:
                out_feat = self.conv_out(x)
                out_feat = out_feat.permute(0, 2, 3, 1)
                return xs, mask, out, out_feat
            else:
                return xs, mask, out
        else:
            return out

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # fix all bn layers
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)
