import torch
from torch import nn


class Policy(nn.Module):

    def __init__(self, inDim, outDim):
        super().__init__()

        self.baseNet = nn.Sequential([
            nn.Linear(inDim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, outDim)
        ])

        self.longIONet = nn.Sequential([
            nn.Conv1d(),

        ])

    
    def forward(self, c, refMotion, shortIO, longIO):
