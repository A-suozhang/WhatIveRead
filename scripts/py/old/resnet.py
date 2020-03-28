import torch
import torch.nn as nn
import math
from .module import *

__all__ = ['resnet', 'ResNet']
        

class BasicBlock(nn.Module):
    ''' basicblock of resnet which supports quantize and asparse

        inplanes    : number of input channel
        planes      : number of output channel
        stride      : stride of the first conv
        downsample  : nn.Sequential, how to do downsample on residual path
        q_cfg       : dict, configs of quantization
    '''

    expansion = 1       # channel expansion

    def __init__(self, inplanes, planes, stride=1, downsample=None, q_cfg=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.downsample = downsample

        # quantize mode
        if q_cfg is not None:
            self.conv1 = QConv2d(inplanes, planes, kernel_size=3, stride=self.stride,
                                 padding=1, bias=False, q_cfg=q_cfg)
            self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=1,
                                 padding=1, bias=False, q_cfg=q_cfg)
        # float mode
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=self.stride,
                                   padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = residual + out
        out = self.relu2(out)

        return out



class Bottleneck(nn.Module):
    ''' bottleneck of resnet which support quantize and asparse

        inplanes    : number of input channel
        planes      : number of output channel
        stride      : stride of the first conv
        downsample  : nn.Sequential, how to do downsample on residual path
        q_cfg       : dict, configs of quantization
    '''

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, q_cfg=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.stride = stride

        # quantize mode
        if q_cfg is not None:
            self.conv1 = QConv2d(inplanes, planes, kernel_size=1, bias=False, q_cfg=q_cfg)
            self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=self.stride,
                                 padding=1, bias=False, q_cfg=q_cfg)
            self.conv3 = QConv2d(planes, planes * 4, kernel_size=1, bias=False, q_cfg=q_cfg)
        # float mode
        else:    
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=self.stride,
                                   padding=1, bias=False)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out+= residual
        out = self.relu3(out)

        return out

# dictionary of layer numbers and block types to define the structure of model
_depth = {
    'imagenet': {
        18: ([2, 2, 2, 2], BasicBlock),
        34: ([3, 4, 6, 3], BasicBlock),
        50: ([3, 4, 6, 3], Bottleneck),
        101: ([3, 4, 23, 3], Bottleneck),
        152: ([3, 8, 36, 3], Bottleneck),
    },
    'cifar': {
        20: ([3, 3, 3], BasicBlock),
        32: ([5, 5, 5], BasicBlock),
        44: ([7, 7, 7], BasicBlock),
        56: ([9, 9, 9], BasicBlock),
        110: ([18, 18, 18], BasicBlock),
        1202: ([200, 200, 200], BasicBlock)
    }
}


class ResNet(nn.Module):
    ''' ResNet class define which support quantize and asparse
        
        depth       : depth of the network
        num_classes : how many classed does the network used to classifying
        q_cfg       : dict, how to do quantization
    '''

    def __init__(self, depth, num_classes=10, q_cfg=None):

        assert depth in _depth['imagenet'] or depth in _depth['cifar']
        self.depth = depth
        super(ResNet, self).__init__()
        
        if self.depth in _depth['imagenet']:
            # the first layer of imagenet-resnet
            self.inplanes = 64
            if False:
                self.conv1 = QConv2d(3, self.inplanes, kernel_size=7,
                                     stride=2, padding=3, bias=False, q_cfg=q_cfg)
            else:
                self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7,
                                       stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # the rest layers of imagenet-resnet
            self.layers = []
            layers, block_type = _depth['imagenet'][self.depth]
            # the first block with stride=1
            self.layers.append(
                self._make_layer(block_type, self.inplanes, layers[0], q_cfg=q_cfg)
            )
            # the other blocks with stride=2
            for idx in range(1, len(layers)):
                self.layers.append(
                    self._make_layer(block_type,self.inplanes*(2),layers[idx],
                                     stride=2, q_cfg=q_cfg)
                )
            self.layers = nn.Sequential(*self.layers)

            # the pooling and linear layer at the end
            self.avgpool = nn.AvgPool2d(7)
            if q_cfg is not None and False:
                self.fc = QLinear(512 * block_type.expansion, num_classes, q_cfg=q_cfg)
            else:
                self.fc = nn.Linear(512 * block_type.expansion, num_classes)
            
        else:
            # the first layer of cifar-resnet
            self.inplanes = 16
            if False:
                self.conv1 = QConv2d(3, self.inplanes, kernel_size=3, stride=1,
                                     padding=1, bias=False, q_cfg=q_cfg)
            else:
                self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1,
                                       padding=1, bias=False)
            self.bn1     = nn.BatchNorm2d(self.inplanes)
            self.relu    = nn.ReLU(inplace=True)
            self.maxpool = None

            # the rest layers of cifar-resnet
            self.layers  = []
            layers, block_type = _depth['cifar'][self.depth]
            # the first block with stride=1
            self.layers.append(
                self._make_layer(block_type, self.inplanes, layers[0], q_cfg=q_cfg)
            )
            # the rest blocks with stride=2
            for idx in range(1, len(layers)):
                self.layers.append(
                    self._make_layer(block_type, self.inplanes*2, layers[idx], 
                                     stride=2, q_cfg=q_cfg)
                )
            self.layers = nn.Sequential(*self.layers)

            # the pooling and linear layer at the end
            self.avgpool = nn.AvgPool2d(8)
            if q_cfg is not None and False:
                self.fc = QLinear(64 * block_type.expansion, num_classes, q_cfg=q_cfg)
            else:
                self.fc = nn.Linear(64 * block_type.expansion, num_classes)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, q_cfg=None):
        r''' the inner function to make layers of resnet 
             whick consists of some blocks
            
            block   : class, block type of the layers set
            planes  : number of output channel
            blocks  : number of blocks in one layers set
            stride  : stride of the first block
            q_cfg   : dict, how to do quantization
        '''

        downsample = None       # no downsample for default
        # set downsample of cifar-layers
        if self.depth in _depth['cifar']:
            # set downsample if activation shape changes
            if self.inplanes != planes * block.expansion:
                if q_cfg is not None and (True):
                    downsample = nn.Sequential(
                        QConv2d(self.inplanes, planes*block.expansion, kernel_size=1,
                                stride=stride, bias=False, q_cfg=q_cfg),
                        nn.BatchNorm2d(planes*block.expansion),
                    )
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes * block.expansion),
                    )
        # set downsample of imagenet-layers
        else:
            # set downsample if activation shape changes
            if stride != 1 or self.inplanes != planes*block.expansion:
                if q_cfg is not None:
                    downsample = nn.Sequential(
                        QConv2d(self.inplanes, planes*block.expansion, kernel_size=1,
                                stride=stride, bias=False, q_cfg=q_cfg),
                        nn.BatchNorm2d(planes * block.expansion),
                    )
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes * block.expansion),
                    )

        layers = []
        # define a block with arguments and append to layers list
        layers.append(block(self.inplanes, planes, stride, downsample, q_cfg))
        self.inplanes = planes * block.expansion
        # define other blocks with differnet planes and append to layers list
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, q_cfg=q_cfg))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.layers(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


    def enable_quantize(self):
        ''' API to enable quantize
        '''
        self.apply(q_enable)

    def set_lr_scale(self, lr_p):
        ''' API to set learning rate scale bit
       '''
        for m in self.modules():
            if hasattr(m, 'lr_scale_p') and hasattr(m, "magnitude"):
                if m.magnitude == 'ceil' and lr_p != 0:
                    m.lr_scale_p = lr_p
                    print("    Set lr scale bit of",lr_p)


def resnet(depth, num_classes, q_cfg=None):
    ''' function to get a ResNet
    '''

    return ResNet(depth, num_classes, q_cfg)

