import torch
import torch.nn as nn
import math
# from variable_length_pooling import VariableLengthPooling

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                     padding=kernel_size//2, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, embedding_dim, inplanes, block, layers, sample_cnn, dropout=0.5):
        self.inplanes = inplanes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, self.inplanes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, embedding_dim*2, layers[0])
        self.layer1 = self._make_layer(block, embedding_dim*4, layers[1], kernel_size=3, stride=1)
        # self.layer2 = self._make_layer(block, embedding_dim*8, layers[2], kernel_size=3, stride=1)
        # self.layer3 = self._make_layer(block, embedding_dim*16, layers[3], kernel_size=3, stride=1)

        # self.layer0 = self._make_layer(block, 64, layers[0])
        # self.layer1 = self._make_layer(block, 128, layers[1], kernel_size=3, stride=1)
        # self.layer2 = self._make_layer(block, 256, layers[2], kernel_size=3, stride=1)
        # self.layer3 = self._make_layer(block, 512, layers[3], kernel_size=3, stride=1)
        # self.layer4 = self._make_layer(block, 512, layers[3], kernel_size=1, stride=1)
        # self.layer5 = self._make_layer(block, 512, layers[3], stride=1)

        # self.layer0 = self._make_layer(block, 64, layers[0])
        # self.layer1 = self._make_layer(block, 64, layers[0], kernel_size=1, stride=1)
        # self.layer2 = self._make_layer(block, 128, layers[1], kernel_size=5, stride=1)
        # self.layer3 = self._make_layer(block, 128, layers[2], kernel_size=5, stride=1)
        # self.layer4 = self._make_layer(block, 256, layers[3], kernel_size=1, stride=1)
        # self.layer5 = self._make_layer(block, 256, layers[3], stride=1)

        # self.layer0 = self._make_layer(block, 256, layers[0])
        # self.layer1 = self._make_layer(block, 256, layers[0], kernel_size=1, stride=1)
        # self.layer2 = self._make_layer(block, 256, layers[1], kernel_size=5, stride=1)
        # self.layer3 = self._make_layer(block, 256, layers[2], kernel_size=5, stride=1)
        # self.layer4 = self._make_layer(block, 512, layers[3], kernel_size=1, stride=1)
        # self.layer5 = self._make_layer(block, 512, layers[3], stride=1)

        self.dropout = nn.Dropout(dropout)
        self.conv_merge = nn.Conv1d(embedding_dim*4 * block.expansion, 1,
                                    kernel_size=3, stride=1, padding=1,
                                    bias=True)
        # self.vlp = VariableLengthPooling()
        # self.avgpool = nn.AvgPool2d((5, 1), stride=1)
        self.fc = nn.Linear(sample_cnn, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size,
                            stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x, bounds=None):
        # import pdb;
        # pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #
        x = self.relu(self.conv_merge(x))
        # import pdb; pdb.set_trace()
        x = torch.squeeze(x, dim=1)
        x = self.fc(self.dropout(x))
        # x = self.vlp(x, bounds=bounds)
        # import pdb; pdb.set_trace()
        return x

