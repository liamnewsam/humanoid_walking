import mujoco
import numpy as np
import gymnasium as gym

WALK_C = 1
STAND_C = 0


def _quat_angle_diff(q1, q2):
    dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


class FlatWorld(gym.Env):

    FALL_HEIGHT = 0.5  # pelvis z below this counts as a fall
    FALL_PENALTY = -20.0

    # hold each commanded action for this many physics steps instead of resampling every
    # single one; at dt=0.002s this gives ~60Hz control on top of 500Hz physics, which keeps
    # exploration noise from being reissued to the (stiff, kp=500) actuators every 2ms
    CONTROL_DECIMATION = 8

    STAND_POSE_WEIGHT = 1.0
    STAND_ORIENT_WEIGHT = 1.0
    STAND_VEL_WEIGHT = 0.1
    STAND_ACTION_RATE_WEIGHT = 0.01
    STAND_HEIGHT_WEIGHT = 1.0
    STAND_FOOT_WEIGHT = 1.0
    STAND_REWARD_FLOOR = -5.0  # no single step should cost more than a fraction of FALL_PENALTY

    def __init__(self):
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)

        self.pelvisID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.footIDs = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"),
        ])
        self.standKeyID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")

        self.obsDim = self.model.nq + self.model.nv
        self.actDim = self.model.nu

        self.ctrlLow = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrlHigh = self.model.actuator_ctrlrange[:, 1].copy()

        # qpos indices each actuator drives, and the reference "stand" pose to track
        self.actuatedQposAdr = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]].copy()
        standQpos = self.model.key_qpos[self.standKeyID].copy()
        self.standJointQpos = standQpos[self.actuatedQposAdr]
        self.standPelvisQuat = standQpos[3:7].copy()
        self.standPelvisZ = float(standQpos[2])

        # foot body world z-height depends on the full kinematic chain, so it takes a forward
        # pass at the stand keyframe to read (unlike the pelvis, which is qpos[2] directly)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.standKeyID)
        mujoco.mj_forward(self.model, self.data)
        self.standFootZ = float(np.mean(self.data.xpos[self.footIDs, 2]))

        # the pose/action pair used to fabricate history before an episode has any real history
        self.standObs = np.concatenate([standQpos, np.zeros(self.model.nv)]).astype(np.float32)
        self.standAction = ((self.standJointQpos - self.ctrlLow) / (self.ctrlHigh - self.ctrlLow)).astype(np.float32)

        self.prevPelvisX = 0.0
        self.prevCtrl = self.standJointQpos.copy()

        self.control_dt = self.model.opt.timestep * self.CONTROL_DECIMATION

    @staticmethod
    def _build_model():
        spec = mujoco.MjSpec.from_file("unitree_g1/g1.xml")
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[20, 20, 0.1],
            pos=[0, 0, 0]
        )
        return spec.compile()

    def _obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.standKeyID)
        mujoco.mj_forward(self.model, self.data)

        self.prevPelvisX = float(self.data.xpos[self.pelvisID][0])
        self.prevCtrl = self.standJointQpos.copy()

        return self._obs(), {}

    def step(self, action, c):
        action = np.clip(action, 0.0, 1.0)
        ctrl = self.ctrlLow + action * (self.ctrlHigh - self.ctrlLow)
        self.data.ctrl[:] = ctrl

        for _ in range(self.CONTROL_DECIMATION):
            mujoco.mj_step(self.model, self.data)
            if self.data.xpos[self.pelvisID][2] < self.FALL_HEIGHT:
                break

        pelvisX = float(self.data.xpos[self.pelvisID][0])
        pelvisZ = float(self.data.xpos[self.pelvisID][2])

        if c == WALK_C:
            reward = (pelvisX - self.prevPelvisX) / self.control_dt
            self.prevPelvisX = pelvisX
        elif c == STAND_C:
            poseErr = np.mean((self.data.qpos[self.actuatedQposAdr] - self.standJointQpos) ** 2)
            orientErr = _quat_angle_diff(self.data.qpos[3:7], self.standPelvisQuat) ** 2
            velErr = np.mean(self.data.qvel ** 2)
            actionRateErr = np.mean((ctrl - self.prevCtrl) ** 2)
            heightErr = max(0.0, self.standPelvisZ - pelvisZ) ** 2
            footErr = np.mean((self.data.xpos[self.footIDs, 2] - self.standFootZ) ** 2)

            reward = -(
                self.STAND_POSE_WEIGHT * poseErr
                + self.STAND_ORIENT_WEIGHT * orientErr
                + self.STAND_VEL_WEIGHT * velErr
                + self.STAND_ACTION_RATE_WEIGHT * actionRateErr
                + self.STAND_HEIGHT_WEIGHT * heightErr
                + self.STAND_FOOT_WEIGHT * footErr
            )
            reward = max(reward, self.STAND_REWARD_FLOOR)
        else:
            raise ValueError(f"unknown command {c}")

        self.prevCtrl = ctrl

        terminated = pelvisZ < self.FALL_HEIGHT
        if terminated:
            reward = self.FALL_PENALTY
        truncated = False

        return self._obs(), reward, terminated, truncated, {}

    def close(self):
        pass
