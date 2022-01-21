import torch.nn as nn
import torch
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self,args,hidden_dim=256):
        super(Critic, self).__init__()

        self.fc = nn.Linear(hidden_dim*2, args.OUTPUT_DIM)
        self.pre = nn.Linear(args.OUTPUT_DIM, 1)

    def forward(self,x_m):

        x_m = self.fc(x_m)
        reward = self.pre(x_m)

        return reward.squeeze()


class CalReward:
    def __init__(self,dmodel,cmodel,args):
        self.d = dmodel
        self.c = cmodel
        self.args = args

    def reward(self,e,e_m,y,action):

        pro = self.d(e)
        pro_m = self.d(e_m)
        loss = F.binary_cross_entropy(pro,y,reduction='none')
        loss_m = F.binary_cross_entropy(pro_m,y,reduction='none')
        reward = torch.abs(loss_m-loss)

        a = torch.sum(action,dim=-1)

        pre1 = self.c(e)
        pre2 = self.c(e_m)
        if self.args.n_class > 2:
            reward2 = -torch.sum(torch.abs(pre1-pre2),dim=-1)
        else:
            reward2 = -torch.abs(pre1-pre2)
        reward = self.args.reward2*reward2 + self.args.reward1*reward + 0.1*a

        return reward

