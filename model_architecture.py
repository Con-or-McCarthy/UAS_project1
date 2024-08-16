import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import ModuleList

def sensor_based_IMU_fusion(
        num_IMU=2,
        num_sensors=3,
        n_channels=3,
        num_classes=19,
        has_pressure=False,
):
    # Calculate number of individual sensors (excluding air pressure)
    num_sensors = num_IMU*(num_sensors - 1*has_pressure)

    models = []
    for _ in range(num_sensors):
        model_i = Resnet_ensemble_block(
            n_channels=n_channels 
        )
        models.append(model_i)

    # if there is pressure, append model which accepts 1 channel (not pretrained)
    if has_pressure:
        model_atm = Resnet_ensemble_block(
                n_channels=1 
            )
        models.append(model_atm)
        num_sensors += 1


    model = Resnet_fusion_n_IMU(sensor_models = models,
                                num_sensors=num_sensors,
                                sensor_output_size=1024, 
                                num_classes=num_classes)    
    return model

class Resnet_fusion_n_IMU(nn.Module):
    def __init__(self, sensor_models,num_sensors,sensor_output_size, num_classes):

        super(Resnet_fusion_n_IMU, self).__init__()
        self.sensor_models = ModuleList(sensor_models)
        self.num_sensors = num_sensors

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

        # Output layer 
        self.output_layer = nn.Linear(sensor_output_size*num_sensors, num_classes)

    def forward(self, inputs):
        # sensor_outputs should be a list of tensors with shape [batch_size, sensor_output_size]
        sensor_outputs = [model(x).squeeze(2) for model, x in zip(self.sensor_models, inputs)]

        # Concatenate outputs
        fusion_output = torch.cat(sensor_outputs, dim=1) # change to sum?
        fusion_output = self.dropout(fusion_output)

        # FC layer to connect to output classes
        output = self.output_layer(fusion_output)
        # Using softmax to transform output to probabilities (remove if doing training or anything because this messes up the loss function)
        out_probs = F.softmax(output,dim=1)

        return out_probs, fusion_output # return final output (output) and last layer embeddings (fusion_output)

class Resnet_ensemble_block(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        n_channels=3,
    ):
        super(Resnet_ensemble_block, self).__init__()

        cgf = [
            (64, 5, 2, 5, 2, 2),
            (128, 5, 2, 5, 2, 2),
            (256, 5, 2, 5, 5, 1),
            (512, 5, 2, 5, 5, 1),
            (1024, 5, 0, 5, 3, 1),
        ]

        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet_ensemble_block.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)
        # print("DEBUG feats:", feats.size())

        return feats


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        # print("DEBUG post conv1:", x.size())
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x

def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )
