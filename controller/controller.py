
import torch
from torch import nn
from torch.autograd import Variable

class Controller(nn.Module):
    def __init__(self,
                 lr_init,
                 lr_scheduler,
                 l2_reg=0,
                 entropy_weight=None,
                 grad_bound=None,
                 bl_dec=0.999,
                 ):

        super(Controller, self).__init__()

        self.lr_init = lr_init
        self.lr_scheduler = lr_scheduler
        self.l2_reg = l2_reg
        self.entropy_weight = entropy_weight
        self.grad_bound = grad_bound
        self.bl_dec = bl_dec

        self.baseline = None



    def advantage(self, reward, entropy=None):
        #what should be the initial value for baseline?

        if not self.baseline:
            self.baseline = Variable(torch.zeros([1,1], dtype= torch.float32), requires_grad = False)
            if self.is_cuda: self.baseline = self.baseline.cuda()
        
        self.baseline = self.bl_dec* self.baseline + (1- self.bl_dec)* reward


        if self.entropy_weight:
            assert  entropy
            print('entropy-----')
            print(entropy)
            reward += self.entropy_weight* entropy

        return reward - self.baseline

    def reinforce_train(self, controller_step, reward, log_prob, entropy=None):

        advantage = self.advantage(reward, entropy)
        print('advantage:')
        print(advantage)
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
        for param_group in self.opt.param_groups:
            param_group['lr'] = self.lr_scheduler(param_group['lr'], step)



