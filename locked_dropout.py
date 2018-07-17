import torch
import torch.nn as nn
# from torch.autograd import Variable


class LockedDropout(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        # m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        # mask = Variable(m, requires_grad=False) / (1 - dropout)

        # my alternative since pytorch 0.4, which doesn't have new()
        # nor Variable
        # create tensor like x with all values set to 1 - dropout
        # generate bernoulli masks with 1 - dropout probability
        m = (torch.zeros(1, x.size(1), x.size(2)) + (1 - dropout)).bernoulli()
        mask = m / (1 - dropout)

        # x is output from RNN, x.shape == (seq_len, batch_size, x_dim)
        # therefore by expanding_as(x) we use the same mask for all time steps.
        mask = mask.expand_as(x)

        if x.is_cuda:
            mask = mask.cuda()

        return mask * x
