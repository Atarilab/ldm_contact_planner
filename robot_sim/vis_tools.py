import mujoco
import mujoco.viewer
import numpy as np
import pinocchio
import time
from .utils import scnGeomManager

from robot_descriptions.loaders.mujoco import load_robot_description

def vis_robot(com_trajectory, joint_trajectory, timestep_trajectory: list=[]):

    assert len(com_trajectory) == len(joint_trajectory)

    model = load_robot_description("go2_mj_description")
    data = mujoco.MjData(model)

    sim_step = 0
    sim_dt = 0.1
    model.opt.timestep = sim_dt

    trajectory_length = len(com_trajectory)
    # com_positions = []
    # com_orientations = []
    # joint_positions = []
    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(0.02)
        # press_keys(['Tab'])
        # press_keys(['shift', 'Tab'])

        # Disable some rendering features for simplicity
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        viewer.sync()
        viewer.user_scn.ngeom = 0
        scn_geom_manager = scnGeomManager(viewer)
        
        while viewer.is_running():
            if sim_step >= trajectory_length:
                sim_step = 0
            
            scn_geom_manager.clear()
            scn_geom_manager.add_plane()
            data.qpos[0:3] = com_trajectory[sim_step][:3]
            rotation_matrix = pinocchio.rpy.rpyToMatrix(com_trajectory[sim_step][3], com_trajectory[sim_step][4], com_trajectory[sim_step][5])
            pin_quat = pinocchio.Quaternion(rotation_matrix)            
            data.qpos[3:7] = [pin_quat.w, pin_quat.x, pin_quat.y, pin_quat.z]
            data.qpos[7:] = joint_trajectory[sim_step]

            sim_step += 1
            
            mujoco.mj_kinematics(model, data)
            scn_geom_manager.update()
            viewer.sync()
            data.time += sim_dt
            if timestep_trajectory:
                if sim_step == trajectory_length - 1:
                    time.sleep(timestep_trajectory[sim_step]) # 
                else:
                    time.sleep(timestep_trajectory[-1]) # 
            else:
                time.sleep(2)

def vis_box(com_trajectory, timestep_trajectory: list=[]):

    world_model_xml = """
    <mujoco>
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
        rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"
        reflectance=".2"/>
    </asset>

    <worldbody>
        <light pos="0 0 1" mode="trackcom"/>
        <geom name="ground" type="plane" pos="0 0 0" size="20 20 .1" material="grid" solimp=".99 .99 .01" solref=".001 1"/>
    </worldbody>
    </mujoco>
    """

    model = mujoco.MjModel.from_xml_string(world_model_xml)
    data = mujoco.MjData(model)

    sim_step = 0
    sim_dt = 0.1
    model.opt.timestep = sim_dt

    trajectory_length = len(com_trajectory)
    # com_positions = []
    # com_orientations = []
    # joint_positions = []
    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(0.02)
        # press_keys(['Tab'])
        # press_keys(['shift', 'Tab'])

        # Disable some rendering features for simplicity
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        # viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        viewer.sync()
        viewer.user_scn.ngeom = 0
        scn_geom_manager = scnGeomManager(viewer)
        
        while viewer.is_running():
            if sim_step >= trajectory_length:
                sim_step = 0
                
            scn_geom_manager.clear()
            rotation_matrix = pinocchio.rpy.rpyToMatrix(com_trajectory[sim_step][3], com_trajectory[sim_step][4], com_trajectory[sim_step][5])

            scn_geom_manager.add_box(com_trajectory[sim_step][:3], [0.05,0.05,0.05], rotation_matrix)

            sim_step += 1
            
            mujoco.mj_kinematics(model, data)
            scn_geom_manager.update()
            viewer.sync()
            data.time += sim_dt
            if timestep_trajectory[0]:
                if sim_step == trajectory_length - 1:
                    # time.sleep(0.1) #
                    time.sleep(timestep_trajectory[-1]) # 
                else:
                    # time.sleep(0.1) #
                    time.sleep(timestep_trajectory[sim_step-1]) # 
            else:
                time.sleep(2)