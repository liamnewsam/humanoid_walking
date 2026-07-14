import mujoco
import mujoco.viewer
import mediapy as media
import matplotlib.pyplot as plt

spec = mujoco.MjSpec.from_file("unitree_g1/g1.xml")

spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[20, 20, 0.1],
    pos=[0, 0, 0]
)

model = spec.compile()
data = mujoco.MjData(model)

#mujoco.viewer.launch(model, data)

# Make renderer, render and show the pixels

pelvisID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
#with mujoco.Renderer(model) as renderer:
#  mujoco.mj_forward(model, data)
#  print(data.xpos[pelvisID])
#  renderer.update_scene(data)


import time
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        
        print(data.xpos[pelvisID])
        

        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))