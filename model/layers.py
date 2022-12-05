import torch.nn as nn

class UpConv3x3(nn.Module):
    """
    Use bilinear followed by conv
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(UpConv3x3, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels, bias=bias)
        self.non_linear = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2.0, mode='nearest')
        out = self.conv(out)
        out = self.non_linear(out)
        return out


class Conv3x3(nn.Module):
    """
    Convolution layer with 3 kernel size, followed by non_linear layer
    """
    def __init__(self, in_channels, out_channels, padding_mode='reflect', bias=True):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        out = self.conv(x)
        return out