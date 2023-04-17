import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


models = ['convnet']


class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, self.shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        self.num_feat = self.shape_feat[0]*self.shape_feat[1]*self.shape_feat[2]
        #self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)) #new line
        self.classifier = nn.Linear(self.shape_feat[0], num_classes)#first arg first self.num_feat
        #self.out_features = shape_feat[0]


    #def get_out_features(self):
    #    return self.shape_feat[0]
    #out_features = property(get_out_features)
#The dimension of output features
    

    def forward(self, x):
        out = self.features(x)
       # out = self.avg_pool(out)
        #out = out.view(out.size(0), -1) #This line is present in original condensation repo
        #out = self.classifier(out)      #This line is present in original condensation repo
        return out

    def embed(self, x):
        out = self.features(x)
        #out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat
    
    
    @property
    def out_features(self) -> int:
       return self.shape_feat[0]

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.classifier)
    
#Convnet model from DatasetCondensation
def convnet(pretrained= False,**kwargs):
    return ConvNet(**kwargs)

