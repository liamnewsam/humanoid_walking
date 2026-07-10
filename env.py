
import mujoco
import gymnasium as gym

class FlatWorld(gym.Env):

    def __init__(self):
        self.spec = mujoco.MjSpec.from_file("unitree_g1/g1.xml")

        self.spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[20, 20, 0.1],
            pos=[0, 0, 0]
        )

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        self.pelvisID = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def reset(self):
        self.spec = mujoco.MjSpec.from_file("unitree_g1/g1.xml")

        self.spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[20, 20, 0.1],
            pos=[0, 0, 0]
        )

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)

        return self.data.qpos[:]


    def step(self, action):

        self.data.qpos[:] = action
        mujoco.mj_forward(self.model, self.data)

        reward = self.reward()

        return reward, self.data.qpos[:]

    def reward(self):
        pelvisPos = self.data.xpos[self.pelvisID]
        return pelvisPos[0]


import time
world = FlatWorld()
print(world.data.qpos)
print(world.data.qvel)
for i in range(10):
    time.sleep(0.1)
    mujoco.mj_forward(world.model, world.data)
    
