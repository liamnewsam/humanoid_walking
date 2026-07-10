import torch
import numpy as np
from policy import Policy
from env import FlatWorld
from dataclasses import dataclass

import torch.nn.functional as F



world = FlatWorld()

policy = Policy()

@dataclass
class Config:

    lr = 5e-4
    gamma = 0.99
    lam: 0.95

    n_epochs = 10

    device = "cpu"



class RolloutBuffer:

    def __init__(self, capacity):
        self._ptr = 0
        self.capacity = capacity

        self.obs = np.zeros((capacity, world.obsDim), dtype=np.float16)
        self.actions = np.zeros((capacity, world.actDim), dtype=np.float16)
        self.rewards = np.zeros(capacity, dtype=np.float16)
        self.values = np.zeros(capacity, dtype=np.float16)
        self.log_probs = np.zeros(capacity, dtype=np.float16)
        self.dones = np.zeros(capacity, dtype=np.float16)

    def push(self, action, obs, reward, value, log_prob, done):

        i = self._ptr

        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = done

        self._ptr += 1

    @property
    def full(self):
        return self._ptr >= self.capacity
    

    def compute_gae(self, last_value, gamma, lam):

        n = self._ptr
        adv = np.zeros(n, dtype=np.float16)
        gae = 0.0

        for t in reversed(range(n)):
            next_val = last_value if t == n-1 else self.values[t+1]
            delta = self.rewards[t] + gamma * next_val - self.values[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        
        returns = adv + self.values[:n]

        return adv, returns
    
    def tensors(self, device):
        n = self._ptr
        return (
            torch.as_tensor(self.obs[:n], device=device),
            torch.as_tensor(self.actions[:n], device=device),
            torch.as_tensor(self.rewards[:n], device=device),
            torch.as_tensor(self.values[:n], device=device),
            torch.as_tensor(self.log_probs[:n], device=device),
            torch.as_tensor(self.dones[:n], device=device),
        )
    

    def clear(self):
        self._ptr = 0



def ppo_update(policy, optimizer, buffer, last_value, cfg):
    adv, returns = buffer.compute_gae(last_value, cfg.gamma, cfg.lam)

    adv_t = torch.as_tensor(adv, device=cfg.device)
    ret_t = torch.as_tensor(returns, device=cfg.device)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)


    obs, actions, rewards, values, old_log_probs, dones = buffer.tensors(cfg.device)
    n = len(actions)

    pg_losses, vf_losses, ent_losses = [], [], []

    for _ in range(cfg.n_epochs):
        idxs = torch.randperm(n, device=cfg.device)

        for start in range(0, n, cfg.batch_size):
            b = idxs[start : start + cfg.batch_size]

            log_probs, values, entropy = policy.evaluate(
                obs[b], actions[b]
            )

            ratio = torch.exp(log_probs - old_log_probs[b])
            adv_b = adv_t[b]

            pg_loss = -torch.min(ratio * adv_b, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b).mean()

            vf_loss = F.mse_loss(values, ret_t[b])
            ent_loss = -entropy.mean()

            loss = pg_loss + cfg.vf_coef * vf_loss + cfg.ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            ent_losses.append(ent_loss.item())
    
    return {
        "pg_loss": float(np.mean(pg_losses)),
        "vf_loss": float(np.mean(vf_losses)),
        "entropy": float(np.mean(ent_losses)),
    }


def train(cfg: Config = Config())
