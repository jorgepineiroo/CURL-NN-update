# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

'''
import matplotlib
import numpy as np
import sys
import torch
import torch.nn as nn
import rgb_ted
from utils import ImageProcessing
import math
import torch.nn.functional as F

matplotlib.use('agg')
np.set_printoptions(threshold=sys.maxsize)


class NEW_CURLLoss(nn.Module):
    """
    Simplified CURL Loss function following the paper structure:
    Total Loss = L_lab + L_hsv + L_rgb + L_reg
    
    Where:
    - L_lab: L1 loss in CIELab color space
    - L_hsv: HSV loss in conical space (handles angular nature of Hue)
    - L_rgb: L1 loss + Cosine similarity loss in RGB space
    - L_reg: Curve regularization loss (gradient smoothness)
    """

    def __init__(self, device, w_lab=1.0, w_hsv=1.0, w_rgb=1.0, w_cosine=1.0, w_reg=1e-6):
        """
        Initialisation of the CURL loss function.
        
        :param device: torch device (cuda/cpu)
        :param w_lab: weight for LAB loss term
        :param w_hsv: weight for HSV loss term  
        :param w_rgb: weight for RGB L1 loss term
        :param w_cosine: weight for cosine similarity loss within RGB term
        :param w_reg: weight for curve regularization term
        """
        super(NEW_CURLLoss, self).__init__()
        self.device = device
        self.w_lab = w_lab
        self.w_hsv = w_hsv
        self.w_rgb = w_rgb
        self.w_cosine = w_cosine
        self.w_reg = w_reg

    def compute_hsv_loss(self, predicted_img_batch, target_img_batch):
        """
        Compute HSV loss in conical space to handle the angular nature of Hue.
        
        L_hsv = mean(|s_p * v_p * cos(h_p) - s_r * v_r * cos(h_r)| + 
                     |s_p * v_p * sin(h_p) - s_r * v_r * sin(h_r)|)
        
        :param predicted_img_batch: predicted RGB images (BxCxHxW)
        :param target_img_batch: target RGB images (BxCxHxW)
        :returns: HSV loss value
        """
        # Convert RGB to HSV
        predicted_hsv = ImageProcessing.rgb_to_hsv_new(predicted_img_batch, device=self.device)
        target_hsv = ImageProcessing.rgb_to_hsv_new(target_img_batch, device=self.device)
        
        # Extract H, S, V channels - H is normalized to [0,1], convert to radians [0, 2π]
        pred_hue = predicted_hsv[:, 0, :, :] * 2 * math.pi
        pred_sat = predicted_hsv[:, 1, :, :]
        pred_val = predicted_hsv[:, 2, :, :]
        
        target_hue = target_hsv[:, 0, :, :] * 2 * math.pi
        target_sat = target_hsv[:, 1, :, :]
        target_val = target_hsv[:, 2, :, :]
        
        # Conical space representation
        pred_x = pred_sat * pred_val * torch.cos(pred_hue)
        pred_y = pred_sat * pred_val * torch.sin(pred_hue)
        
        target_x = target_sat * target_val * torch.cos(target_hue)
        target_y = target_sat * target_val * torch.sin(target_hue)
        
        # L1 loss on conical coordinates
        term1 = torch.abs(pred_x - target_x)
        term2 = torch.abs(pred_y - target_y)
        
        return torch.mean(term1 + term2)

    def compute_lab_loss(self, predicted_img_batch, target_img_batch):
        """
        Compute L1 loss in CIELab color space.
        
        :param predicted_img_batch: predicted RGB images (BxCxHxW)
        :param target_img_batch: target RGB images (BxCxHxW)
        :returns: LAB L1 loss value
        """
        predicted_lab = torch.clamp(
            ImageProcessing.rgb_to_lab(predicted_img_batch, device=self.device), 0, 1
        )
        target_lab = torch.clamp(
            ImageProcessing.rgb_to_lab(target_img_batch, device=self.device), 0, 1
        )
        
        return F.l1_loss(predicted_lab, target_lab)

    def compute_rgb_loss(self, predicted_img_batch, target_img_batch):
        """
        Compute RGB loss: L1 pixel-wise loss + Cosine similarity loss.
        
        L_rgb = L1(pred, target) + w_cosine * (1 - cosine_similarity(pred, target))
        
        :param predicted_img_batch: predicted RGB images (BxCxHxW)
        :param target_img_batch: target RGB images (BxCxHxW)
        :returns: RGB loss value (L1 + cosine)
        """
        # L1 pixel-wise loss
        l1_loss = F.l1_loss(predicted_img_batch, target_img_batch)
        
        # Cosine similarity loss between RGB pixel vectors
        # cosine_similarity computes along dim=1 (channel dimension)
        cosine_sim = F.cosine_similarity(predicted_img_batch, target_img_batch, dim=1)
        cosine_loss = 1.0 - torch.mean(cosine_sim)
        
        return l1_loss + self.w_cosine * cosine_loss

    def compute_reg_loss(self, gradient_regulariser):
        """
        Compute curve regularization loss.
        Penalizes non-smooth curves by measuring gradient variations.
        
        :param gradient_regulariser: regularization tensor from curve layers
        :returns: regularization loss value
        """
        if not torch.is_tensor(gradient_regulariser):
            return torch.tensor(0.0, device=self.device)
        
        return torch.mean(gradient_regulariser)

    def forward(self, predicted_img_batch, target_img_batch, gradient_regulariser):
        """
        Forward function for the CURL loss.
        
        Total Loss = w_lab * L_lab + w_hsv * L_hsv + w_rgb * L_rgb + w_reg * L_reg
        
        :param predicted_img_batch: predicted images (BxCxHxW)
        :param target_img_batch: ground truth images (BxCxHxW)
        :param gradient_regulariser: curve regularization tensor
        :returns: total loss value
        """
        # Compute individual loss terms
        l_lab = self.compute_lab_loss(predicted_img_batch, target_img_batch)
        l_hsv = self.compute_hsv_loss(predicted_img_batch, target_img_batch)
        l_rgb = self.compute_rgb_loss(predicted_img_batch, target_img_batch)
        l_reg = self.compute_reg_loss(gradient_regulariser)
        
        # Weighted sum of all losses
        total_loss = (
            self.w_lab * l_lab + 
            self.w_hsv * l_hsv + 
            self.w_rgb * l_rgb + 
            self.w_reg * l_reg
        )
        
        return total_loss


class NEW_CURLLayer(nn.Module):
    import torch.nn.functional as F

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation of class

        :param num_in_channels: number of input channels
        :param num_out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(NEW_CURLLayer, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.make_init_network()

    def make_init_network(self):
        """ Initialise the CURL block layers

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
        :param device: torch device
        :returns: Tensor representing the predicted image
        :rtype: Tensor

        """

        '''
        This function is where the magic happens :)
        Curves are applied to the ORIGINAL image, while TED features
        are used for predicting the curve parameters.
        '''
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

        # img_hsv = ImageProcessing.rgb_to_hsv(img_rgb, device=device)
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
        """ Forward function for the higher level convolution block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
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
        """ Forward function for the max pooling block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.max_pool(x)
        return img_out


class GlobalPoolingBlock(Block, nn.Module):

    def __init__(self, receptive_field):
        """Implementation of the global pooling block. Takes the average over a 2D receptive field.
        :param receptive_field:
        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Forward function for the high-level global pooling block

        :param x: Tensor of shape BxCxAxA
        :returns: Tensor of shape BxCx1x1, where B is the batch size
        :rtype: Tensor

        """
        out = self.avg_pool(x)
        return out


class CURLNet(nn.Module):

    def __init__(self, device):
        """Initialisation function

        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CURLNet, self).__init__()
        self.tednet = rgb_ted.TEDModel()
        self.curllayer = NEW_CURLLayer()
        self.device = device

    def forward(self, img):
        """Neural network forward function

        :param img: forward the data img through the network
        :returns: enhanced image and gradient regulariser
        :rtype: Tensor

        """
        feat = self.tednet(img)
        enhanced_img, gradient_regulariser = self.curllayer(feat, img, self.device)
        return enhanced_img, gradient_regulariser
