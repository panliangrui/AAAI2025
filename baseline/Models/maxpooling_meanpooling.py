import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Maxpooling(nn.Module):
    def __init__(self, in_size, n_classes=1):
        super(Maxpooling, self).__init__()

        self.classifier =  nn.Linear(in_size, n_classes)

    def forward(self, x):
        M = torch.max(x, dim=0)[0].view(1,-1) # KxL
        Y_prob = self.classifier(M)
        return Y_prob


class Meanpooling(nn.Module):
    def __init__(self, in_size, n_classes=1):
        super(Meanpooling, self).__init__()

        self.classifier =  nn.Linear(in_size, n_classes)

    def forward(self, x):
        M = torch.mean(x, dim=0).view(1,-1)  # KxL
        Y_prob = self.classifier(M)
        return Y_prob
