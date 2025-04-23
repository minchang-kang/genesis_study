import numpy as np

import genesis as gs
gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.URDF(file='/home/mars/genesis/genesis_study/urdf/modeling_Indy7/urdf/modeling_Indy7.urdf',
                   pos=np.array([0,0,0.0]),
                   fixed=True,),
)

scene.build()
end_effector = robot.get_link('body_7')
dof_names = ["body_1","body_2","body_3","body_4","body_5","body_6","body_7"]
dofs_idx = [robot.get_joint(name).dof_idx_local for name in dof_names]

robot.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
    dofs_idx,
)
robot.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200]),
    dofs_idx,
)

qpos = robot.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.25]),
    quat=np.array([0, 1, 0, 0]),
)

# import IPython
# IPython.embed()


for i in range(1000):

    scene.step()