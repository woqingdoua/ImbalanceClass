import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

class RL(nn.Module):

    def __init__(self, hidden_dim):
        super(RL, self).__init__()

        self.fc_rl = nn.Linear(hidden_dim*2, hidden_dim*2)

    def forward(self, x):

        out = torch.sigmoid(self.fc_rl(x))
        m = Bernoulli(out)
        action = m.sample()
        log_softmax = torch.exp(m.log_prob(action))
        entrop = m.entropy()

        return x * action, log_softmax, entrop, out,action

    def cal_loss(self,reward,predictions,log_softmax, entropy):

        reward_baseline = reward - predictions.squeeze()
        loss = -torch.mean(torch.mean(reward_baseline.unsqueeze(1) * log_softmax + entropy,dim=1), dim=0)

        return loss

class Htanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        _, indices = torch.sort(x,dim=-1)
        indices = (indices > 450).float()
        ctx.save_for_backward(x,indices)
        return x * indices

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_variables
        dx = grad_output * x * weight
        return dx