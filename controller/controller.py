
import torch
from torch import nn
from torch.autograd import Variable

class Controller(nn.Module):
    def __init__(self,
                 lr_init=0.03,
                 lr_dec_start=0,
                 lr_dec_every=100,
                 lr_dec_rate=0.9,
                 l2_reg=0,
                 entropy_weight=None,
                 grad_bound=None,
                 bl_dec=0.999,
                 lr_cosine_dec_args = None
                 ):

        super(Controller, self).__init__()
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_every = lr_dec_every
        self.lr_dec_rate = lr_dec_rate
        self.l2_reg = l2_reg
        self.entropy_weight = entropy_weight
        self.grad_bound = grad_bound
        self.bl_dec = bl_dec
        self.lr_cosine_dec_args = lr_cosine_dec_args

        self.baseline = None



    def advantage(self, reward, entropy=None):
        #what should be the initial value for baseline?

        reward = reward.view(-1, 1)
        if not self.baseline:
            self.baseline = Variable(torch.zeros(1, dtype= torch.float32), requires_grad = False)
        else:
            self.baseline = self.bl_dec* self.baseline + (1- self.bl_dec)* reward


        if self.entropy_weight:
            assert  entropy
            reward += self.entropy_weight* entropy

        return reward - self.baseline

    def reinforce_train(self, controller_step, reward, log_prob, entropy=None):

        advantage = self.advantage(reward, entropy)

        loss = -(log_prob * advantage).sum()
        #lr schedule
        self.decay_learning_rate(controller_step)

        self.opt.zero_grad()

        loss.backward()
        if self.grad_bound:
            torch.nn.utils.clip_grad_norm(self.parameters(), self.grad_bound)
        self.opt.step()

        return loss

    def decay_learning_rate(self, step):


        if not self.lr_cosine_dec_args:
            step = max(step - self.lr_dec_start, 0)
            if step % self.lr_dec_every ==0:
                for param_group in self.opt.param_groups:
                    param_group['lr']  = param_group['lr'] * self.lr_dec_rate

        else:
            #TODO perform cosine decay
            return



