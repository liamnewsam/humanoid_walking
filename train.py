import torch
import numpy as np
from policy import Policy
from env import FlatWorld, STAND_C
from dataclasses import dataclass

import torch.nn.functional as F


@dataclass
class Config:

    lr: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95

    n_ppo_epochs: int = 4
    rollout_steps: int = 1000
    n_iterations: int = 50
    batch_size: int = 32

    clip_eps: float = 0.2
    vf_coef: float = 0.0001
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    command: int = STAND_C

    SHORT_HIST_LENGTH: int = 4
    LONG_HIST_LENGTH: int = 66


world = FlatWorld()


class RolloutBuffer:

    def __init__(self, capacity, obsDim, actDim, short_hist_len, long_hist_len, stand_obs, stand_action):
        self._ptr = 0
        self._ep_step_ctr = 0  # steps taken so far in the current episode
        self.capacity = capacity
        self.short_hist_len = short_hist_len
        self.long_hist_len = long_hist_len

        self.obs = np.zeros((capacity, obsDim), dtype=np.float32)
        self.actions = np.zeros((capacity, actDim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # command in effect when this transition was collected, so training evaluates each
        # transition under the command it was actually acted under (not whatever cfg.command
        # happens to be at update time)
        self.commands = np.zeros(capacity, dtype=np.int64)
        # steps since this transition's episode began; used to fabricate history near an
        # episode's start and to mark those steps as invalid for training (see ppo_update)
        self.ep_step = np.zeros(capacity, dtype=np.int64)

        self.stand_obs = np.asarray(stand_obs, dtype=np.float32)
        self.stand_action = np.asarray(stand_action, dtype=np.float32)

        # continuously-updated trailing real history, maintained across rollout boundaries
        # (unlike self.obs/self.actions, which get logically reused starting at index 0 every
        # rollout); _history_tail_* always reflects the most recent push, whenever it happened
        self._history_tail_obs = np.tile(self.stand_obs, (long_hist_len, 1))
        self._history_tail_actions = np.tile(self.stand_action, (long_hist_len, 1))
        # frozen snapshot of the above, taken once per rollout in clear(). This is what
        # _window_at actually reads from when it needs to look further back than index 0 of
        # the current rollout -- it must stay fixed for the rollout's whole lifetime (including
        # later training replay), not keep drifting as more of the *next* rollout gets pushed
        self.carry_obs = self._history_tail_obs.copy()
        self.carry_actions = self._history_tail_actions.copy()

    def _window_at(self, i, ep_step_value, hist_len):
        k = min(ep_step_value, hist_len)
        real_start = i - k

        if real_start >= 0:
            real_obs = self.obs[real_start:i]
            real_actions = self.actions[real_start:i]
        else:
            # this window reaches back before index 0 of the current rollout -- the episode is
            # continuing from a previous rollout, so pull the deficit from the carried-over tail
            deficit = -real_start
            real_obs = np.concatenate([self.carry_obs[-deficit:], self.obs[0:i]])
            real_actions = np.concatenate([self.carry_actions[-deficit:], self.actions[0:i]])

        pad = hist_len - k
        if pad <= 0:
            return real_actions, real_obs

        obs_pad = np.tile(self.stand_obs, (pad, 1))
        act_pad = np.tile(self.stand_action, (pad, 1))
        return np.concatenate([act_pad, real_actions]), np.concatenate([obs_pad, real_obs])

    def current_history(self, device):
        i = self._ptr
        shortAct, shortObs = self._window_at(i, self._ep_step_ctr, self.short_hist_len)
        longAct, longObs = self._window_at(i, self._ep_step_ctr, self.long_hist_len)
        longIO = np.concatenate([longObs, longAct], axis=1)

        return (
            torch.as_tensor(shortAct, device=device).unsqueeze(0),
            torch.as_tensor(shortObs, device=device).unsqueeze(0),
            torch.as_tensor(longIO, device=device).unsqueeze(0),
        )

    def batch_windows(self, idxs, device):
        shortActs, shortObss, longIOs = [], [], []
        for i in idxs:
            i = int(i)
            ep_step_value = int(self.ep_step[i])

            shortAct, shortObs = self._window_at(i, ep_step_value, self.short_hist_len)
            longAct, longObs = self._window_at(i, ep_step_value, self.long_hist_len)

            shortActs.append(shortAct)
            shortObss.append(shortObs)
            longIOs.append(np.concatenate([longObs, longAct], axis=1))

        return (
            torch.as_tensor(np.stack(shortActs), device=device),
            torch.as_tensor(np.stack(shortObss), device=device),
            torch.as_tensor(np.stack(longIOs), device=device),
        )

    def push(self, action, obs, reward, value, log_prob, done, command):
        i = self._ptr

        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = done
        self.ep_step[i] = self._ep_step_ctr
        self.commands[i] = command

        self._history_tail_obs[:-1] = self._history_tail_obs[1:]
        self._history_tail_obs[-1] = obs
        self._history_tail_actions[:-1] = self._history_tail_actions[1:]
        self._history_tail_actions[-1] = action

        self._ptr += 1
        if done:
            self._ep_step_ctr = 0
            self._history_tail_obs[:] = self.stand_obs
            self._history_tail_actions[:] = self.stand_action
        else:
            self._ep_step_ctr += 1

    @property
    def full(self):
        return self._ptr >= self.capacity

    def compute_gae(self, last_value, gamma, lam):

        n = self._ptr
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self.values[t + 1]
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae

        returns = adv + self.values[:n]

        return adv, returns

    def clear(self):
        # only resets storage for the next rollout; episode tracking (_ep_step_ctr) must not
        # reset here, since the episode may still be ongoing across this rollout boundary --
        # only push()'s `done` flag should ever restart it. Freeze this rollout's carry-over
        # snapshot now, before any new pushes start mutating _history_tail_*.
        self._ptr = 0
        self.carry_obs = self._history_tail_obs.copy()
        self.carry_actions = self._history_tail_actions.copy()


def ppo_update(policy, optimizer, buffer, last_value, cfg):
    adv, returns = buffer.compute_gae(last_value, cfg.gamma, cfg.lam)
    n = buffer._ptr

    # steps whose history window still contains fabricated (non-real) padding never had a
    # trustworthy input, so they're excluded from the loss entirely (their rewards/values are
    # still used above in compute_gae, since GAE needs the full, unbroken reward sequence)
    valid_idx = np.nonzero(buffer.ep_step[:n] >= buffer.long_hist_len)[0]
    if len(valid_idx) == 0:
        return None

    valid_idx_t = torch.as_tensor(valid_idx, dtype=torch.long, device=cfg.device)

    adv_t = torch.as_tensor(adv, device=cfg.device)[valid_idx_t]
    ret_t = torch.as_tensor(returns, device=cfg.device)[valid_idx_t]
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    actions_t = torch.as_tensor(buffer.actions[:n], device=cfg.device)[valid_idx_t]
    old_log_probs = torch.as_tensor(buffer.log_probs[:n], device=cfg.device)[valid_idx_t]
    commands_t = torch.as_tensor(buffer.commands[:n], device=cfg.device)[valid_idx_t]

    m = len(valid_idx)
    pg_losses, vf_losses, ent_losses = [], [], []

    for _ in range(cfg.n_ppo_epochs):
        perm = torch.randperm(m)

        for start in range(0, m, cfg.batch_size):
            local_b = perm[start: start + cfg.batch_size]
            global_b = valid_idx[local_b.numpy()]

            shortIn, shortOut, longIO = buffer.batch_windows(global_b, cfg.device)
            log_probs, values_pred, entropy = policy.evaluate(
                shortIn, shortOut, longIO, commands_t[local_b], actions_t[local_b]
            )

            ratio = torch.exp(log_probs - old_log_probs[local_b])
            adv_b = adv_t[local_b]

            pg_loss = -torch.min(ratio * adv_b, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b).mean()

            vf_loss = F.mse_loss(values_pred, ret_t[local_b])
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
        "n_valid": m,
    }


def train(cfg: Config = Config()):

    policy = Policy(
        world.obsDim, world.actDim, cfg.SHORT_HIST_LENGTH, cfg.LONG_HIST_LENGTH, world.standAction
    ).to(cfg.device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    buffer = RolloutBuffer(
        cfg.rollout_steps, world.obsDim, world.actDim, cfg.SHORT_HIST_LENGTH, cfg.LONG_HIST_LENGTH,
        world.standObs, world.standAction
    )

    obs, _ = world.reset()
    ep_reward = 0.0
    ep_rewards = []
    global_step = 0

    for iteration in range(1, cfg.n_iterations + 1):
        policy.eval()
        buffer.clear()

        while not buffer.full:

            action, log_prob, value = policy.act(*buffer.current_history(cfg.device), cfg.command)
            action = action.squeeze(0).cpu().numpy()

            obs_next, reward, terminated, truncated, _ = world.step(action, cfg.command)
            ep_reward += reward
            global_step += 1

            buffer.push(
                action, obs, reward, float(value.item()), float(log_prob.item()), terminated or truncated,
                cfg.command
            )

            obs = obs_next
            if terminated or truncated:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = world.reset()

        with torch.no_grad():
            _, _, last_val = policy(*buffer.current_history(cfg.device), cfg.command)
            last_value = float(last_val.item())

        policy.train()
        metrics = ppo_update(policy, optimizer, buffer, last_value, cfg)

        avg_reward = float(np.mean(ep_rewards[-10:])) if ep_rewards else ep_reward
        if metrics is None:
            print(
                f"iter {iteration:3d} | steps {global_step:6d} | "
                f"skipped update (no post-burn-in transitions) | avg_ep_reward {avg_reward:.3f}"
            )
        else:
            print(
                f"iter {iteration:3d} | steps {global_step:6d} | "
                f"pg {metrics['pg_loss']:.4f} vf {metrics['vf_loss']:.4f} ent {metrics['entropy']:.4f} | "
                f"valid {metrics['n_valid']:4d} | avg_ep_reward {avg_reward:.3f}"
            )

    world.close()
    return policy


if __name__ == "__main__":
    policy = train()
    torch.save(policy.state_dict(), "checkpoints/test1.pth")

