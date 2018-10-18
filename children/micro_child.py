
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from collections import OrderedDict

from layers import IdentityLayer, SepConv2dBN, ReshapePool2d
from config import Config
config = Config()

class EnasBlock(nn.Module):

    def __init__(self,  in_channel, out_channel, num_nodes=4, num_operations=5, stride=1):
        super(EnasBlock,self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_nodes = num_nodes
        self.num_operations = num_operations
        self.stride = stride

        self.op_fns = OrderedDict([
            ("identity", IdentityLayer),
            ("sepconv3", partial(SepConv2dBN, kernel_size=3, padding=1)),
            ("sepconv5", partial(SepConv2dBN, kernel_size=5, padding=2)),
            ("avgpool_", partial(ReshapePool2d, mode='avg', kernel_size=3, padding=1)),
            ("maxpool_", partial(ReshapePool2d, mode='max', kernel_size=3, padding=1)),
        ]) # adapted from  https://github.com/bkj/ripenet

        self.op_lookup = dict(zip(range(num_operations), self.op_fns.keys()))

        self.construct_full_dag()

        # self.reset_parameters()


    def construct_full_dag(self):
        # register all edges
        self.dag = []

        for node_id in range(2, self.num_nodes+2):
            node_ops = []
            for node_input in range(node_id): # including two inputs
                input_ops_list = []
                for op_id in range(self.num_operations):
                    op_name = self.op_lookup[op_id]
                    if node_input == 0 or node_input == 1:
                        op_fn = self.op_fns[op_name](in_channel = self.in_channel[node_input],
                                                     out_channel = self.out_channel,
                                                     stride = self.stride
                                                     )
                    else:
                        op_fn = self.op_fns[op_name](in_channel=self.out_channel,
                                                     out_channel=self.out_channel,
                                                     stride=1
                                                     )
                    self.add_module('node_{0} to node_{1} with op_{2}'.format(node_input, node_id, op_name), op_fn)
                    input_ops_list.append(op_fn)

                node_ops.append(input_ops_list)
            out_aggregate_conv = nn.Sequential(nn.Conv2d(self.out_channel, self.out_channel, kernel_size=1),
                                               nn.BatchNorm2d(self.out_channel),
                                               nn.ReLU()
                                               )
            self.add_module('node_{} out aggregate conv'.format(node_id), out_aggregate_conv)
            node_ops.append(out_aggregate_conv)

            self.dag.append(node_ops)


    def activate_partial_dag(self):
        #may not need this
        pass

    def reset_parameters(self):
        #todo param initialization
        pass


    def forward(self, *input):
        pre_layers, arc = input
        layers = [pre_layers[0], pre_layers[1]]
        # inference with nodes in dag through edges specified by arc
        used = torch.zeros((self.num_nodes+2))
        for node_id in range( self.num_nodes):
            node_ops = self.dag[node_id]
            x_id, y_id, op_x, op_y = arc[4*node_id: 4*(node_id+1)]

            used[x_id] += 1
            used[y_id]+= 1
            # input x
            input_x = layers[x_id]
            x_op = node_ops[x_id][op_x]
            x = x_op(input_x)

            input_y = layers[y_id]
            y_op = node_ops[y_id][op_y]
            y = y_op(input_y)

            out = node_ops[-1](x+y)

            layers.append(out)

        layers = torch.stack(layers)
        out_nodes = torch.nonzero(used == 0).view(-1)
        out = layers[out_nodes].sum(0)
        return out

class Skeletion(nn.Module):

    def __init__(self, in_channel=3, num_nodes= 2, num_operations=5, channels =(8,8,16,16,32,32,64,64),
                 num_classes=10, ):
        super(Skeletion,self).__init__()
        self.in_channel = in_channel
        self.num_nodes = num_nodes
        self.channels = channels
        self.num_classes = num_classes
        self.num_operations = num_operations

        self.num_layers = len(channels)
        self.pool_layers = [0, 2, 4, 6] # pool to reduce spatial dimension at these layers

        self.conv_stem = nn.Conv2d(in_channel, channels[0]*3, kernel_size=3, padding=1)
        self.bn_stem = nn.BatchNorm2d(channels[0]*3)
        self.relu_stem = nn.ReLU()

        self.in_channel = channels[0]*3

        self.stem_op = nn.Sequential(self.conv_stem,
                                     self.bn_stem,
                                     self.relu_stem,
                                     )

        self.blocks = nn.ModuleList()

        in_channel = [self.in_channel, self.in_channel]
        for layer_id in range(self.num_layers):

            out_channel = channels[layer_id]

            if layer_id  in self.pool_layers:
                reduce_layer_0 = nn.Sequential(
                        nn.Conv2d(in_channel[0],out_channel, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU()
                    )
                reduce_layer_1 = nn.Sequential(
                        nn.Conv2d(in_channel[1], out_channel, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU()
                    )
                self.blocks.extend([reduce_layer_0,reduce_layer_1])

                in_channel = [out_channel, out_channel]

            block = EnasBlock(in_channel, out_channel, self.num_nodes, self.num_operations, stride=1)
            self.blocks.append(block)
        self.fc = nn.Linear(self.channels[-1],
                            self.num_classes
                            )

    def forward(self, x, arc):

        x = self.stem_op(x)
        layers = [x,x]
        p= 0
        for block_id in range(self.num_layers):

            if block_id in self.pool_layers:
                reduce_0, reduce_1 = self.blocks[p:p+2]
                p+=2
                layers = [reduce_0(layers[0]), reduce_1(layers[1])]
            block = self.blocks[p]
            p+=1
            x =block(layers, arc)

            # print(x.shape)
            layers = [layers[-1],x]

        # print(x)
        x = F.adaptive_avg_pool2d(x, (1,1))

        x = self.fc(x.view(x.shape[0],-1))

        return x

class EnasNet(nn.Module):

    def __init__(self, provider, test_loader, in_channel, img_shape, lr_scheduler,
                 num_nodes=2,
                 num_classes=10,
                 num_operations=5,
                 channels=(8,8,16,16),

                 grad_bound = None,
                 **kargs
                 ):
        super(EnasNet, self).__init__()

        self.provider = provider
        self.test_loader = test_loader

        self.img_shape = img_shape

        self.in_channel = in_channel

        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.num_operations = num_operations
        self.channels = channels
        self.lr_scheduler = lr_scheduler
        self.grad_bound = grad_bound
        self.build_model()

        self.set_optimizer(kargs['opt_params'])


    def train_step(self, step, arc):


        self.decay_learning_rate(step)

        x_batch, y_batch = next(self.provider)
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        logits = self.model(x_batch, arc)
        loss= self.loss(logits, y_batch)

        self.opt.zero_grad()

        loss.backward()

        if self.grad_bound:
            torch.nn.utils.clip_grad_norm(self.parameters(), self.grad_bound)

        self.opt.step()

        # print(torch.argmax(logits, dim=-1))

        # print(y_batch)

        # print(torch.argmax(logits, dim=-1) == y_batch)

        acc = (torch.argmax(logits, dim=-1) == y_batch).sum() / x_batch.shape[0]


        return loss, acc


    def loss(self, logits, label):

        loss = nn.functional.cross_entropy(logits, label)
        return loss

    def build_model(self):

        self.model = Skeletion(self.in_channel,
                               self.num_nodes,
                               self.num_operations,
                               self.channels,
                               self.num_classes,
                               )

    def set_optimizer(self, params):

        opt_name = params['opt']

        init_lr = params['lr']

        if opt_name == 'adam':
            self.opt = torch.optim.Adam(self.model.parameters(),
                                        lr = init_lr
                                        )

    def decay_learning_rate(self,step):

        if self.lr_scheduler.lr_cosine:
            step = self.provider.epoch

        for param_group in self.opt.param_groups:
            param_group['lr'] = self.lr_scheduler(param_group['lr'], step)


    def evaluate(self, arc):
        #todo evaluate on validation set

        acc = []
        for x_batch, y_batch in self.test_loader:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            logits = self.model(x_batch, arc)
            acc.append((torch.argmax(logits, dim=-1) == y_batch).sum() / x_batch.shape[0])


        return np.mean(acc)




















