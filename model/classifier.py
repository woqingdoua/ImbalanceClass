import torch.nn as nn
import torch
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, hidden_dim,args):
        super(Classifier,self).__init__()

        self.hidden_dim = hidden_dim
        if args.n_class == 2:
            class_num = 1
            self.fc = nn.Linear(hidden_dim, class_num)

        else:
            class_num = 11
            self.fc = nn.Linear(hidden_dim,class_num)
            self.fc2 = nn.Linear(class_num, class_num)

    def forward(self, x,mask=None):

        if mask == None:
            x = self.fc(x)
        else:
            x = self.fc(x*mask)

        return torch.sigmoid(x).squeeze()


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator,self).__init__()

        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        x = self.fc(x)
        return torch.sigmoid(x.squeeze(-1))