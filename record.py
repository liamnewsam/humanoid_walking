import mujoco
import mediapy as media
import torch

from train import RolloutBuffer, world
from env import WALK_C


def record_video(policy, duration_s, path="rollout.mp4", fps=30, deterministic=True, width=640, height=480,
                  command=WALK_C):
    """Roll the policy out in `world` for `duration_s` seconds and write an mp4 to `path`."""

    device = next(policy.parameters()).device
    n_steps = int(duration_s / world.control_dt)
    render_every = max(1, round(1.0 / (fps * world.control_dt)))

    buffer = RolloutBuffer(n_steps, world.obsDim, world.actDim, policy.shortHistLen, policy.longHistLen,
                            world.standObs, world.standAction)
    obs, _ = world.reset()

    cam = mujoco.MjvCamera()
    cam.trackbodyid = world.pelvisID
    cam.distance = 3.0
    cam.azimuth = 140
    cam.elevation = -20

    policy.eval()
    frames = []

    with mujoco.Renderer(world.model, height=height, width=width) as renderer:
        for step in range(n_steps):
            history = buffer.current_history(device)

            with torch.no_grad():
                if deterministic:
                    alpha, beta, value = policy(*history, command)
                    action = alpha / (alpha + beta)
                    log_prob = torch.zeros(1)
                else:
                    action, log_prob, value = policy.act(*history, command)

            action_np = action.squeeze(0).cpu().numpy()
            obs_next, reward, terminated, truncated, _ = world.step(action_np, command)

            buffer.push(
                action_np, obs, reward, float(value.item()), float(log_prob.item()), terminated or truncated,
                command
            )
            obs = obs_next

            if step % render_every == 0:
                renderer.update_scene(world.data, camera=cam)
                frames.append(renderer.render())

            if terminated or truncated:
                obs, _ = world.reset()

    media.write_video(path, frames, fps=fps)
    print(f"wrote {len(frames)} frames ({len(frames) / fps:.1f}s) to {path}")

    return path

'''
if __name__ == "__main__":
    from train import train

    policy = train()
    record_video(policy, duration_s=5.0, path="rollout.mp4")
'''