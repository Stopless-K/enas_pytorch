import numpy as np


class Lr_Scheduler():

    def __init__( self,
                 decay_type = 'exponential',
                 lr_dec_start= 0,
                 lr_dec_every=1000,
                 lr_dec_rate=0.9,
                 lr_dec_min = 0,
                 lr_cosine = False,
                 lr_t_0 = None,
                 lr_t_mul = None,
                 lr_max = None,
                 lr_min = None
                ):

        self.decay_type = decay_type
        self.lr_dec_start = lr_dec_start
        self.lr_dec_every = lr_dec_every
        self.lr_dec_rate = lr_dec_rate
        self.lr_dec_min = lr_dec_min
        self.lr_cosine = lr_cosine
        self.lr_t_0 = lr_t_0
        self.lr_t_mul = lr_t_mul
        self.lr_max = lr_max
        self.lr_min = lr_min


        if lr_cosine:
            assert lr_t_0 is not None
            assert lr_t_mul is not None
            assert lr_max is not None
            assert lr_min is not None

            self.last_reset = 0
            self.t_i = self.lr_t_0

        if self.decay_type == 'exponential':
            self._decay_func = self._expoential_decay
        elif self.decay_type == 'cosine':
            self._decay_func = self._cosine_decay


    def _expoential_decay(self, lr, step):
        if lr <= self.lr_dec_min: return lr

        if step > self.lr_dec_start and step % self.lr_dec_every == 0:

            new_lr = lr * self.lr_dec_rate

            return max(new_lr, self.lr_dec_min)

        return lr

    def _cosine_decay(self, lr,epoch):

        t_curr = epoch - self.last_reset
        if t_curr >= self.t_i:
            self.last_reset = epoch
            self.t_i = self.t_i* self.lr_t_mul

            t_curr = epoch - self.last_reset

        rate = float(t_curr) / float(self.t_i) * 3.1415926

        lr = self.lr_min + 0.5*(self.lr_max - self.lr_min)*(1.0+ np.cos(rate))

        return lr



    def __call__(self, *args, **kwargs):

        return self._decay_func(*args, **kwargs)



