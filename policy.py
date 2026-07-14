import torch
from torch import nn
import torch.nn.functional as F

from torch.distributions import Beta

def _conv_out_len(length, kernel, stride):
    return (length - kernel) // stride + 1


class Policy(nn.Module):

    # initial Beta "concentration" (alpha+beta ~= this * a normalizing spread); higher means a
    # tighter initial spread around standAction instead of near-uniform over the full joint range
    INIT_CONCENTRATION = 50.0

    def __init__(self, obsDim, actionDim, shortHistLen, longHistLen, standAction):
        super().__init__()

        self.obsDim = obsDim
        self.actionDim = actionDim
        self.obsActDim = obsDim + actionDim

        self.shortHistLen = shortHistLen
        self.longHistLen = longHistLen

        self.longIONet = nn.Sequential(
            nn.Conv1d(self.obsActDim, 32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        longOutLen = _conv_out_len(_conv_out_len(longHistLen, 6, 3), 4, 2)
        longEmbedDim = 16 * longOutLen

        inputDim = shortHistLen * actionDim + shortHistLen * obsDim + longEmbedDim + 1

        self.baseNet = nn.Sequential(
            nn.Linear(inputDim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
        )

        # raw (alpha, beta) logits per action dim; softplus+1 in forward() keeps the Beta unimodal
        self.actionHead = nn.Linear(512, 2 * actionDim)

        # zero the weight and solve the bias so that, at init, the action distribution starts
        # centered on standAction with a tight-ish spread (concentration = INIT_CONCENTRATION)
        # instead of near-uniform over each joint's full range
        standAction_t = torch.as_tensor(standAction, dtype=torch.float32)
        alphaTarget = self.INIT_CONCENTRATION * standAction_t
        betaTarget = self.INIT_CONCENTRATION * (1.0 - standAction_t)
        alphaBias = torch.log(torch.expm1(alphaTarget - 1.0))
        betaBias = torch.log(torch.expm1(betaTarget - 1.0))

        with torch.no_grad():
            self.actionHead.weight.zero_()
            self.actionHead.bias.copy_(torch.cat([alphaBias, betaBias]))

        self.valueHead = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, shortInput, shortOutput, longIO, c):
        batch = shortInput.shape[0]

        longEmbedding = self.longIONet(longIO.transpose(1, 2)).flatten(start_dim=1)

        if torch.is_tensor(c):
            cTensor = c.reshape(batch, 1).to(dtype=shortInput.dtype, device=shortInput.device)
        else:
            cTensor = torch.full((batch, 1), float(c), dtype=shortInput.dtype, device=shortInput.device)

        x = torch.cat([
            shortInput.flatten(start_dim=1),
            shortOutput.flatten(start_dim=1),
            longEmbedding,
            cTensor,
        ], dim=1)

        bodyOutput = self.baseNet(x)

        alphaLogit, betaLogit = self.actionHead(bodyOutput).chunk(2, dim=-1)
        alpha = F.softplus(alphaLogit) + 1.0
        beta = F.softplus(betaLogit) + 1.0

        value = self.valueHead(bodyOutput)

        return alpha, beta, value

    def act(self, shortInput, shortOutput, longIO, c):
        with torch.no_grad():
            alpha, beta, value = self(shortInput, shortOutput, longIO, c)
            dist = Beta(alpha, beta)
            action = dist.sample()
            logProb = dist.log_prob(action).sum(-1)

        return action, logProb, value

    def evaluate(self, shortInput, shortOutput, longIO, c, action):
        alpha, beta, value = self(shortInput, shortOutput, longIO, c)
        dist = Beta(alpha, beta)

        logProb = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return logProb, value.squeeze(-1), entropy
