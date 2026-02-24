# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

Adapted for inference from training model architecture.
'''
from utilities import ted
from utilities.util import ImageProcessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


class CURLLayer(nn.Module):
    """
    CURL Layer that matches the NEW_CURLLayer architecture from training.
    Processes images through LAB, RGB, and HSV curve adjustments.
    """

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation of class

        :param num_in_channels: number of input channels
        :param num_out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLayer, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.make_init_network()

    def make_init_network(self):
        """Initialise the CURL block layers

        :returns: N/A
        :rtype: N/A

        """
        self.lab_layer1 = ConvBlock(64, 64)
        self.lab_layer2 = MaxPoolBlock()
        self.lab_layer3 = ConvBlock(64, 64)
        self.lab_layer4 = MaxPoolBlock()
        self.lab_layer5 = ConvBlock(64, 64)
        self.lab_layer6 = MaxPoolBlock()
        self.lab_layer7 = ConvBlock(64, 64)
        self.lab_layer8 = GlobalPoolingBlock(2)

        self.fc_lab = torch.nn.Linear(64, 48)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)

        self.rgb_layer1 = ConvBlock(64, 64)
        self.rgb_layer2 = MaxPoolBlock()
        self.rgb_layer3 = ConvBlock(64, 64)
        self.rgb_layer4 = MaxPoolBlock()
        self.rgb_layer5 = ConvBlock(64, 64)
        self.rgb_layer6 = MaxPoolBlock()
        self.rgb_layer7 = ConvBlock(64, 64)
        self.rgb_layer8 = GlobalPoolingBlock(2)

        self.fc_rgb = torch.nn.Linear(64, 48)

        self.hsv_layer1 = ConvBlock(64, 64)
        self.hsv_layer2 = MaxPoolBlock()
        self.hsv_layer3 = ConvBlock(64, 64)
        self.hsv_layer4 = MaxPoolBlock()
        self.hsv_layer5 = ConvBlock(64, 64)
        self.hsv_layer6 = MaxPoolBlock()
        self.hsv_layer7 = ConvBlock(64, 64)
        self.hsv_layer8 = GlobalPoolingBlock(2)

        self.fc_hsv = torch.nn.Linear(64, 64)

    def forward(self, x, original_img, device):
        """Forward function for the CURL layer

        :param x: TED feature tensor (B x 64 x H x W)
        :param original_img: original input image (B x 3 x H x W) for curve application
        :param device: torch device (cuda/cpu)
        :returns: Tensor representing the predicted image
        :rtype: Tensor

        """
        x.contiguous()  # remove memory holes

        feat = x[:, 3:64, :, :]
        img = original_img  # Use original image for curve application, not TED features

        img_clamped = torch.clamp(img, 0, 1)

        img_lab = torch.clamp(ImageProcessing.rgb_to_lab(img_clamped, device=device), 0, 1)

        feat_lab = torch.cat((feat, img_lab), 1)

        x = self.lab_layer1(feat_lab)
        del feat_lab
        x = self.lab_layer2(x)
        x = self.lab_layer3(x)
        x = self.lab_layer4(x)
        x = self.lab_layer5(x)
        x = self.lab_layer6(x)
        x = self.lab_layer7(x)
        x = self.lab_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout1(x)
        L = self.fc_lab(x)

        img_lab, gradient_regulariser_lab = ImageProcessing.adjust_lab(img_lab, L[:, 0:48], device=device)

        img_rgb = ImageProcessing.lab_to_rgb(img_lab, device=device)

        img_rgb = torch.clamp(img_rgb, 0, 1)

        feat_rgb = torch.cat((feat, img_rgb), 1)

        x = self.rgb_layer1(feat_rgb)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        x = self.rgb_layer4(x)
        x = self.rgb_layer5(x)
        x = self.rgb_layer6(x)
        x = self.rgb_layer7(x)
        x = self.rgb_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout2(x)
        R = self.fc_rgb(x)

        img_rgb, gradient_regulariser_rgb = ImageProcessing.adjust_rgb(img_rgb, R[:, 0:48], device=device)

        img_rgb = torch.clamp(img_rgb, 0, 1)

        # Use rgb_to_hsv_new matching the training code
        img_hsv = ImageProcessing.rgb_to_hsv_new(img_rgb, device=device)
        img_hsv = torch.clamp(img_hsv, 0, 1)
        feat_hsv = torch.cat((feat, img_hsv), 1)

        x = self.hsv_layer1(feat_hsv)
        del feat_hsv
        x = self.hsv_layer2(x)
        x = self.hsv_layer3(x)
        x = self.hsv_layer4(x)
        x = self.hsv_layer5(x)
        x = self.hsv_layer6(x)
        x = self.hsv_layer7(x)
        x = self.hsv_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout3(x)
        H = self.fc_hsv(x)

        img_hsv, gradient_regulariser_hsv = ImageProcessing.adjust_hsv(img_hsv, H[:, 0:64], device=device)

        img_hsv = torch.clamp(img_hsv, 0, 1)

        # Convert back from HSV to RGB - this IS the enhanced image, not a residual
        img_enhanced = torch.clamp(ImageProcessing.hsv_to_rgb(img_hsv), 0, 1)

        gradient_regulariser = gradient_regulariser_rgb + \
                               gradient_regulariser_lab + gradient_regulariser_hsv

        return img_enhanced, gradient_regulariser


class Block(nn.Module):

    def __init__(self):
        """Initialisation for a lower-level DeepLPF conv block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

    def conv3x3(self, in_channels, out_channels, stride=1):
        """Represents a convolution of shape 3x3

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: the convolution stride
        :returns: convolution function with the specified parameterisation
        :rtype: function

        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=True)


class ConvBlock(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block

        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:

        """
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        """Forward function for the higher level convolution block

        :param x: Tensor representing the input BxCxWxH
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.lrelu(self.conv(x))
        return img_out


class MaxPoolBlock(Block, nn.Module):

    def __init__(self):
        """Initialise function for the max pooling block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Forward function for the max pooling block

        :param x: Tensor representing the input BxCxWxH
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.max_pool(x)
        return img_out


class GlobalPoolingBlock(Block, nn.Module):

    def __init__(self, receptive_field):
        """Implementation of the global pooling block.

        :param receptive_field: (unused, kept for compatibility)
        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Forward function for the high-level global pooling block

        :param x: Tensor of shape BxCxAxA
        :returns: Tensor of shape BxCx1x1
        :rtype: Tensor

        """
        out = self.avg_pool(x)
        return out


class CURLNet(nn.Module):
    """
    Complete CURL Network for inference.
    This matches the architecture from the training code.
    """

    def __init__(self, device):
        """Initialisation function

        :param device: torch device (cuda/cpu)
        :returns: initialises parameters of the neural network
        :rtype: N/A

        """
        super(CURLNet, self).__init__()
        self.tednet = ted.TEDModel()
        self.curllayer = CURLLayer()
        self.device = device

    def forward(self, img):
        """Neural network forward function

        :param img: forward the data img through the network
        :returns: processed image and gradient regulariser
        :rtype: Tensor, Tensor

        """
        feat = self.tednet(img)
        enhanced_img, gradient_regulariser = self.curllayer(feat, img, self.device)
        return enhanced_img, gradient_regulariser

    def get_transformation_curves(self, img):
        """Extract transformation curves from a preview image.
        
        This runs the neural network to compute the L, R, H curve parameters
        without applying them. Use this on a downscaled image for efficiency.
        
        :param img: Input tensor (B, 3, H, W) - can be a small preview
        :returns: Dictionary containing 'L', 'R', 'H' curve parameters
        :rtype: dict
        """
        img = img.contiguous()
        
        # Run through TED network to get features
        feat = self.tednet(img)
        
        feat_only = feat[:, 3:64, :, :]
        
        img_clamped = torch.clamp(img, 0, 1)
        img_lab = torch.clamp(ImageProcessing.rgb_to_lab(img_clamped, device=self.device), 0, 1)
        
        # Extract L curves (LAB adjustment)
        feat_lab = torch.cat((feat_only, img_lab), 1)
        x = self.curllayer.lab_layer1(feat_lab)
        x = self.curllayer.lab_layer2(x)
        x = self.curllayer.lab_layer3(x)
        x = self.curllayer.lab_layer4(x)
        x = self.curllayer.lab_layer5(x)
        x = self.curllayer.lab_layer6(x)
        x = self.curllayer.lab_layer7(x)
        x = self.curllayer.lab_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.curllayer.dropout1(x)
        L = self.curllayer.fc_lab(x)
        
        # Apply LAB curves to get intermediate RGB for next stage
        img_lab_adjusted, _ = ImageProcessing.adjust_lab(img_lab, L[:, 0:48], device=self.device)
        img_rgb = ImageProcessing.lab_to_rgb(img_lab_adjusted, device=self.device)
        img_rgb = torch.clamp(img_rgb, 0, 1)
        
        # Extract R curves (RGB adjustment)
        feat_rgb = torch.cat((feat_only, img_rgb), 1)
        x = self.curllayer.rgb_layer1(feat_rgb)
        x = self.curllayer.rgb_layer2(x)
        x = self.curllayer.rgb_layer3(x)
        x = self.curllayer.rgb_layer4(x)
        x = self.curllayer.rgb_layer5(x)
        x = self.curllayer.rgb_layer6(x)
        x = self.curllayer.rgb_layer7(x)
        x = self.curllayer.rgb_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.curllayer.dropout2(x)
        R = self.curllayer.fc_rgb(x)
        
        # Apply RGB curves to get intermediate HSV for next stage
        img_rgb_adjusted, _ = ImageProcessing.adjust_rgb(img_rgb, R[:, 0:48], device=self.device)
        img_rgb_adjusted = torch.clamp(img_rgb_adjusted, 0, 1)
        img_hsv = ImageProcessing.rgb_to_hsv_new(img_rgb_adjusted, device=self.device)
        img_hsv = torch.clamp(img_hsv, 0, 1)
        
        # Extract H curves (HSV adjustment)
        feat_hsv = torch.cat((feat_only, img_hsv), 1)
        x = self.curllayer.hsv_layer1(feat_hsv)
        x = self.curllayer.hsv_layer2(x)
        x = self.curllayer.hsv_layer3(x)
        x = self.curllayer.hsv_layer4(x)
        x = self.curllayer.hsv_layer5(x)
        x = self.curllayer.hsv_layer6(x)
        x = self.curllayer.hsv_layer7(x)
        x = self.curllayer.hsv_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.curllayer.dropout3(x)
        H = self.curllayer.fc_hsv(x)
        
        return {
            'L': L.detach(),
            'R': R.detach(),
            'H': H.detach()
        }

    def apply_curves(self, img, curves):
        """Apply pre-computed transformation curves to an image.
        
        This applies the L, R, H curves without running the neural network.
        Use this on the full-resolution image with curves computed from a preview.
        
        :param img: Input tensor (B, 3, H, W) - full resolution image
        :param curves: Dictionary containing 'L', 'R', 'H' curve parameters
        :returns: Transformed image tensor
        :rtype: Tensor
        """
        img = torch.clamp(img, 0, 1)
        
        L = curves['L']
        R = curves['R']
        H = curves['H']
        
        # Step 1: Apply LAB adjustment
        img_lab = torch.clamp(ImageProcessing.rgb_to_lab(img, device=self.device), 0, 1)
        img_lab, _ = ImageProcessing.adjust_lab(img_lab, L[:, 0:48], device=self.device)
        img_rgb = ImageProcessing.lab_to_rgb(img_lab, device=self.device)
        img_rgb = torch.clamp(img_rgb, 0, 1)
        
        # Step 2: Apply RGB adjustment
        img_rgb, _ = ImageProcessing.adjust_rgb(img_rgb, R[:, 0:48], device=self.device)
        img_rgb = torch.clamp(img_rgb, 0, 1)
        
        # Step 3: Apply HSV adjustment
        img_hsv = ImageProcessing.rgb_to_hsv_new(img_rgb, device=self.device)
        img_hsv = torch.clamp(img_hsv, 0, 1)
        img_hsv, _ = ImageProcessing.adjust_hsv(img_hsv, H[:, 0:64], device=self.device)
        img_hsv = torch.clamp(img_hsv, 0, 1)
        
        # Convert back to RGB - this IS the enhanced image, not a residual
        img_out = torch.clamp(ImageProcessing.hsv_to_rgb(img_hsv), 0, 1)
        
        return img_out


def load_model(checkpoint_path, device):
    """Load a trained CURL model from a checkpoint.
    
    :param checkpoint_path: Path to the checkpoint file (.pt)
    :param device: torch device (cuda/cpu)
    :returns: Loaded CURLNet model in eval mode
    :rtype: CURLNet
    """
    net = CURLNet(device=device)
    net = net.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    
    return net
