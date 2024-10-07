import time
import os
import mujoco
import mujoco.viewer
import numpy as np
from utils import scnGeomManager

from robot_descriptions.loaders.mujoco import load_robot_description

current_dir = os.path.dirname(os.path.abspath(__file__))

xml_path = os.path.join(current_dir, '..', 'g1_description/g1_29dof.xml')
model = mujoco.MjModel.from_xml_path(xml_path)

# model = load_robot_description("g1_mj_description", variant="scene")
data = mujoco.MjData(model)

sim_step = 0
sim_dt = 0.1
model.opt.timestep = sim_dt

trajectory_length = 100
robot_base_pose = [0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0] 
box_pose = [0.4, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]


# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(0.02)

    viewer.sync()
    viewer.user_scn.ngeom = 0

    scn_geom_manager = scnGeomManager(viewer)

    head_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_link")

    
    while viewer.is_running():
        if sim_step >= trajectory_length:
            sim_step = 0
            
        scn_geom_manager.clear()

        data.qpos[0:7] = robot_base_pose[0:7]
        data.qpos[7:] = [0.0]*(model.nq-7)

        scn_geom_manager.add_box(box_pose[0:3], [0.1,0.1,0.1], np.eye(3))

        head_position = data.xpos[head_link_id]
        head_orientation = data.xquat[head_link_id]
        print(head_position)

        head_orientation_mat = np.eye(3).flatten()
        mujoco.mju_quat2Mat(head_orientation_mat, head_orientation)

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1+scn_geom_manager.total_geoms],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=[1.0, 1.0, 1.1],
            pos=head_position,
            mat=head_orientation_mat,
            rgba=[0.0, 1.0, 0.0, 1.0]
        )

        sim_step += 1
        
        mujoco.mj_kinematics(model, data)
        scn_geom_manager.update()
        viewer.sync()
        data.time += sim_dt
        
        time.sleep(0.01) # add to slow down the visualization