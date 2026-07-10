import torch
from torch import nn

import numpy as np

class Policy(nn.Module):

    def __init__(self, obsDim, actionDim):
        super().__init__()


        self.obsDim = obsDim
        self.actionDim = actionDim

        self.obsActDim = obsDim + actionDim


        self.baseNet = nn.Sequential([
            nn.Linear(obsDim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, actionDim),
            nn.Tanh()
        ])

        self.longIONet = nn.Sequential([
            nn.Conv1d(1, 32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=4, stride=2),
            nn.ReLU()
        ])
        self.shortInput = np.zeros((4,self.actionDim,), dtype=np.float16)
        self.shortOutput = np.zeros((4,self.obsDim,), dtype=np.float16)

        self.longIO = np.zeros((66,self.obsActDim), dtype=np.float16)
        
        self.prevAction = np.zeros(self.actionDim)

    def awaken(self, obs):
        self.shortInput[:] = obs
        self.shortHistory[:] = obs

        self.longIO[:] = np.stack([obs, obs]) # We are assuming observation is just the motor

        self.prevAction = obs
    
    def forward(self, obs, c):

        self.shortInput = np.concatenate(([self.prevAction], self.shortInput[:-1]))
        self.shortOutput = np.concatenate(([obs], self.shortOutput[:-1]))
        self.longIO = np.concatenate((np.stack([obs, self.prevAction]), self.longIO[:-1]))
        
        longIOEmbedding = self.longIONet(self.longIO)

        input = np.vstack([c, np.vstack([self.shortOutput, self.shortInput]), longIOEmbedding])

        meansNormalized = self.baseNet(input)

        self.prevAction = meansNormalized[:]

        return meansNormalized