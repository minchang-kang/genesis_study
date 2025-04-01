import numpy as np

import genesis as gs
gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.URDF(file='/home/mars/genesis/genesis_study/urdf/berkeley_humanoid_description/urdf/robot.urdf',
                   pos=np.array([0,0,0.51])),
)

scene.build()

import IPython
IPython.embed()


for i in range(1000):
    scene.step()