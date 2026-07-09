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

mujoco.viewer.launch(model, data)

# Make renderer, render and show the pixels
'''
with mujoco.Renderer(model) as renderer:
  mujoco.mj_forward(model, data)
  renderer.update_scene(data)
  media.write_image("test.png", renderer.render())

'''