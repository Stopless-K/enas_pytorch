
import torch
from torch import nn

class IdentityLayer(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        if self.in_channel != self.out_channel or stride != 1:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=stride, stride=stride)
            #need to separate into two convs?

            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(self.out_channel)
        else:
            self.conv = None

    def forward(self, x):
        if self.conv:
            return self.relu(self.bn(self.conv(x)))
        else:
            return x


class Conv2dBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, **kargs):

        super(Conv2dBN,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.kernel_size = kernel_size

        if self.in_channel != self.out_channel:
            self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
            self.relu_1 = nn.ReLU()
            self.bn_1 = nn.BatchNorm2d(out_channel)
        else:
            self.conv_1 = None

        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size, stride, **kargs)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        if self.conv_1:
            x = self.relu_1(self.bn_1(self.conv_1(x)))

        x = self.relu(self.bn(self.conv(x)))

        return x



class SepConv2dBN(Conv2dBN):
    def __init__(self, **kargs):
        kargs['groups'] = kargs['out_channel']
        super(SepConv2dBN,self).__init__( **kargs)

class ReshapePool2d(nn.Module):
    def __init__(self, in_channel, out_channel, mode = 'avg', kernel_size= 3, stride=1, **kargs):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.mode = mode
        self.kernel_size = kernel_size

        if self.in_channel != self.out_channel:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(self.out_channel)
        else:
            self.conv = None

        if mode == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, **kargs)
        elif mode == 'max':
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, **kargs)

        else:
            raise NotImplementedError

    def forward(self, x):

        x = self.pool(x)
        if self.conv:
            x = self.relu(self.bn(self.conv(x)))
        return x

class StackLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers= 1):

        super(StackLSTM, self).__init__()

        self.stack_lstm = nn.ModuleList()

        for i in range(num_layers):
            input_size = input_size if i==0 else hidden_size
            self.stack_lstm.append(nn.LSTMCell(input_size, hidden_size))


    def forward(self, *input):

        inputs, prev_h, prev_c = input
        next_h, next_c = [],[]

        for layer, (h, c) in enumerate(zip(prev_h, prev_c)):
            lstm_layer = self.stack_lstm[layer]
            if layer > 0: inputs = next_h[-1]

            # print(inputs)
            # print(h)
            # print(c)
            h_1, c_1 = lstm_layer(inputs, (h,c))
            next_h.append(h_1)
            next_c.append(c_1)

        return next_h, next_c

