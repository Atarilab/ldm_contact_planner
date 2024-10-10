import time
import os
import mujoco
import mujoco.viewer
import numpy as np
from utils import scnGeomManager, axis_to_rotation_matrix
import pinocchio as pin

from robot_descriptions.loaders.mujoco import load_robot_description

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = "/home/michal/projects/tamp/tamp_warm_start/g1_description/g1_29dof.xml"
# xml_path = '/home/michal/projects/tamp/tamp_warm_start/unitree_g1/g1.xml'
model = mujoco.MjModel.from_xml_path(xml_path)

# model = load_robot_description("g1_mj_description", variant="scene")
data = mujoco.MjData(model)

sim_step = 0
sim_dt = 0.1
model.opt.timestep = sim_dt
for i in range(model.ngeom):
    model.geom_rgba[i][3] = 0.3
trajectory_length = 100
robot_base_pose = [-0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0] 
box_pose = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]

# joint_goal = [-0.5993399862484445,
#     0,
#     0,
#     0.6817972237155108,
#     0,
#     0,
#     -0.38197913009763645,
#     0,
#     0,
#     0.6373249995464898,
#     0,
#     0,
#     0,
#     0,
#     0.52,
#     -0.7342540738533904,
#     0,
#     0,
#     1.102012049830485,
#     0,
#     0,
#     0,
#     -0.7708047791999904,
#     0,
#     0,
#     1.0180126248198962,
#     0,
#     0,
#     0]


# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(0.02)

    viewer.sync()
    viewer.user_scn.ngeom = 0

    scn_geom_manager = scnGeomManager(viewer)

    data.qpos[0:7] = robot_base_pose[0:7]
    data.qpos[7:] = [0]*(model.nu)
    right_shoulder_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint")
    left_shoulder_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint")

    # data.qpos[7+right_shoulder_pitch_id] = -np.pi/2
    # data.qpos[7+left_shoulder_pitch_id] = np.pi/2

    # data.qpos[7:(7+model.nu)] = joint_goal

    mujoco.mj_fwdPosition(model, data)

    while viewer.is_running():
        if sim_step >= trajectory_length:
            sim_step = 0
            
        scn_geom_manager.clear()

        # scn_geom_manager.add_box(box_pose[0:3], [0.1,0.1,0.1], np.eye(3))

        # Add the World frame
        scn_geom_manager.add_frame([0.0,0.0,0.0], np.eye(3), 0.1)

        # for i in range(model.njnt):
        #     pos = data.xanchor[i]
        #     xmat = data.xmat[i]
        #     joint_axis = data.xaxis[i]
        #     scn_geom_manager.add_frame(pos, xmat, 0.1, joint_axis)
        #     # scn_geom_manager.add_frame(pos, xmat, 0.1)

        left_foot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_sole")

        print(left_foot_site_id)

        # Get the pose of the left foot site
        left_foot_site_pos = data.site_xpos[left_foot_site_id]
        left_foot_site_rot = data.site_xmat[left_foot_site_id].reshape(3, 3)

        scn_geom_manager.add_frame(left_foot_site_pos, left_foot_site_rot, 0.1)
        scn_geom_manager.add_box([1.0,1.0,0.0], [0.1,0.1,0.1])
        sim_step += 1
        
        mujoco.mj_fwdPosition(model, data)
        scn_geom_manager.update()
        viewer.sync()
        data.time += sim_dt
        
        time.sleep(0.01) # add to slow down the visualization