import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation from DenseNet3D - see paper

# Layer
class _DenseLayer(nn.Sequential):
    """
    Dense layer object, made of 2 3D-Convolutions
    """
    def __init__(self, number_features, growth, batch_norm_size, dropout):
        super(_DenseLayer, self).__init__()
        # 3D convolution cluster
        self.norm1 = nn.BatchNorm3d(number_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(number_features, batch_norm_size * growth, kernel_size=1, stride=1, bias=False)


        # 3D convolution cluster
        self.norm2 = nn.BatchNorm3d(batch_norm_size * growth)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(batch_norm_size * growth, growth, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = dropout

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.dropout > 0:
            new_features = F.dropout(new_features, p=self.dropout, training=self.training)
        return torch.cat([x, new_features], 1)

# Block
class _DenseBlock(nn.Sequential):
    """
    Dense block that contains a certain amount of Dense layers
    """
    def __init__(self, number_layers, number_features, batch_norm_size, growth, dropout):
        super(_DenseBlock, self).__init__()
        for k in range(number_layers):
            added_layer = _DenseLayer(number_features + k * growth, growth, batch_norm_size, dropout)
            name_layer = 'denselayer%d' % (k + 1)
            self.add_module(name_layer, added_layer)

# Transition
class _Transition(nn.Sequential):
    """
    Transition between 2 Dense blocks
    """
    def __init__(self, number_features_in, number_features_out):
        super(_Transition, self).__init__()
        
        # Pooling kernel and stride
        kernel = (1, 2, 2)
        stride = (1, 2, 2)

        self.norm = nn.BatchNorm3d(number_features_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(number_features_in, number_features_out, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=kernel, stride=stride)

# Transition
class _Transition_V2(nn.Sequential):
    """
    Transition between 2 Dense blocks version 2
    """
    def __init__(self, number_features_in, number_features_out):
        super(_Transition, self).__init__()
        
        # Pooling kernel and stride
        kernel = (2, 4, 4)
        stride = (2, 4, 4)

        self.norm = nn.BatchNorm3d(number_features_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(number_features_in, number_features_out, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=kernel, stride=stride)

