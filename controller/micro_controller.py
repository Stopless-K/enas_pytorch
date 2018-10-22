"""

pytorch implementation of enas micro search controller
refer to https://github.com/melodyguan/enas
"""
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

from .controller import Controller
from layers import StackLSTM
from .utils import process_logits, sample

class MicroController(Controller):

    def __init__(self,
                 input_size=32,
                 hidden_size=32,
                 num_nodes= 6,
                 num_operations = 8,
                 cuda = False,
                 temperature = None,
                 tanh_constant= None,
                 op_tanh_reduce = 1,
                 lstm_num_layers= 2,
                 **kargs):

        super(MicroController, self).__init__(**kargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.lstm_num_layers = lstm_num_layers
        self.num_operations = num_operations
        self.is_cuda = cuda
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce
        self.stack_lstm = StackLSTM(self.input_size, self.hidden_size, self.lstm_num_layers)

        #used to compute logits for operations
        self.decoder = nn.Linear(self.hidden_size, self.num_operations)
        self.op_embedding = nn.Embedding(self.num_operations, self.input_size)
        self.tanh = nn.Tanh()
        #attention
        self.attend_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.attend_2 = nn.Linear(self.hidden_size, self.hidden_size)

        #used to compute logits for previous nodes
        self.attend_v = nn.Linear(self.hidden_size, 1)

        self.g_emb = Variable(torch.from_numpy(np.random.uniform(-0.1, 0.1, (1, self.input_size))))
        self.g_emb = self.g_emb.float()

        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr_init, weight_decay=self.l2_reg)
        self.reset_parameters()

        for module in self.children():
            print(module)

    def reset_parameters(self, init_bound= 0.1):
        for param in self.parameters():
            param.data.uniform_(-init_bound, init_bound)

        #init decoder b to bias first two ops, see https://github.com/melodyguan/enas/issues/46
        self.decoder.bias[:2].data.fill_(10)
        self.decoder.bias[2:].data.fill_(0)

    def __call__(self, prev_h=None, prev_c=None, *args, **kwargs):

        if self.is_cuda:
            zero_func = lambda *args, **kargs: torch.zeros(*args, **kargs).cuda()
            zero_like_func = lambda *args, **kargs: torch.zeros_like(*args, **kargs).cuda()
            g_emb = self.g_emb.cuda()

        else:
            zero_func = lambda *args, **kargs: torch.zeros(*args, **kargs)
            zero_like_func = lambda *args, **kargs: torch.zeros_like(*args, **kargs)
            g_emb = self.g_emb

        if not prev_c:
            prev_c =[zero_func(1, self.hidden_size, dtype=torch.float32)
                     for _ in range(self.lstm_num_layers)]
            prev_h = [zero_func(1, self.hidden_size, dtype= torch.float32)
                      for _ in range(self.lstm_num_layers)]


        anchors = zero_func(self.num_nodes+2, self.hidden_size)
        anchors_w = zero_func(self.num_nodes+2, self.hidden_size)

        #maybe convert to cuda
        inputs = g_emb

        #node1, 2
        for i in range(2):
            next_h, next_c = self.stack_lstm(inputs, prev_h, prev_c)
            prev_h, prev_c = next_h, next_c
            anchors[i] = zero_like_func(next_h[-1])
            anchors_w[i] = self.attend_1(next_h[-1])

        actions, log_probs, entropies = [],[],[]

        for node in range(2, self.num_nodes+2):

            idx = range(node)

            for _ in range(2):
                next_h, next_c = self.stack_lstm(inputs, prev_h, prev_c)
                prev_h, prev_c = next_h, next_c
                query = anchors[idx]
                query = query + self.attend_2(next_h[-1])
                logits  = self.attend_v(query).view(1, -1)
                logits = process_logits(logits,self.temperature, self.tanh_constant)
                index, log_prob, entropy = sample(logits)
                index = index.long()
                actions.append(index)
                log_probs.append(log_prob)
                entropies.append(entropy)
                inputs = anchors[index]

            for _ in range(2):
                next_h, next_c = self.stack_lstm(inputs, prev_h, prev_c)
                prev_h, prev_c = next_h, next_c

                logits = self.decoder(next_h[-1])
                logits = process_logits(logits, self.temperature, self.tanh_constant/self.op_tanh_reduce)
                index, log_prob, entropy = sample(logits)
                index = index.long()
                actions.append(index)
                log_probs.append(log_prob)
                entropies.append(entropy)

                inputs = self.op_embedding(index)

            next_h, next_c = self.stack_lstm(inputs, prev_h, prev_c)
            prev_h, prev_c = next_h, next_c
            anchors[node] = next_h[-1]
            anchors_w[node] = self.attend_1(next_h[-1])
            inputs = g_emb


        log_probs = torch.cat(log_probs, -1).sum(-1, True)
        entropies = torch.cat(entropies, -1).sum(-1, True)

        return actions, log_probs, entropies









