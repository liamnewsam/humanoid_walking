import torch
from record import record_video
from policy import Policy
from train import Config
from env import FlatWorld

cfg = Config()
world = FlatWorld()
policy = Policy(
    world.obsDim, world.actDim, cfg.SHORT_HIST_LENGTH, cfg.LONG_HIST_LENGTH, world.standAction
).to(cfg.device)

policy.load_state_dict(torch.load('./checkpoints/test1.pth'))
record_video(policy, 5, path="rollout.mp4")

s = world.model.opt.timestep
print(s*150)