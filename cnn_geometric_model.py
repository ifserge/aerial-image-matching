from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np


class FeatureExtraction(torch.nn.Module):
    def __init__(self, use_cuda=True, feature_extraction_cnn='vgg', last_layer=''):
        super(FeatureExtraction, self).__init__()
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                                  'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                  'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                                  'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children())[:last_layer_idx + 1])
        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]

            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
        # freeze parameters
        for param in self.model.parameters():
            # param.requires_grad = False
            param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model.cuda()
        self.model.eval();

    def forward(self, image_batch):
        return self.model(image_batch)


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class Feature2Pearson(torch.nn.Module):
    def __init__(self):
        super(Feature2Pearson, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        feature_mean = torch.mean(feature, 1, True)
        pearson = feature - feature_mean
        norm = torch.pow(torch.sum(torch.pow(pearson, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(pearson, norm)

class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        # Existed ver
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)

        # else:
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)

        return correlation_tensor


class FeatureMasking(torch.nn.Module):
    def __init__(self):
        super(FeatureMasking, self).__init__()
    #
    def forward(self, correlation_tensor):
        correlation_tensor = correlation_tensor.transpose(1, 2).transpose(2, 3)
        l = 11
        h = 15
        w = 15
        limit_region = np.zeros((w, h, w * h))
        for i in range(h):
            for j in range(w):
                for r_h in range(-1 * l, l + 1):
                    for r_w in range(-1 * l, l + 1):
                        temp_col = j + r_w
                        temp_raw = i + r_h
                        if temp_col in range(w) and temp_raw in range(h):
                            limit_region[i][j][w * (temp_col) + temp_raw] = 1
        cor_mask = torch.unsqueeze(Variable(torch.FloatTensor(limit_region), requires_grad=False), 0)
        cor_mask = cor_mask.cuda()
        correlation_tensor = correlation_tensor * cor_mask
        correlation_tensor = correlation_tensor.transpose(2, 3).transpose(1, 2)

        return correlation_tensor
