import time
import mujoco
import mujoco.viewer
import numpy as np
from utils import scnGeomManager

class KinematicRobotVisualizer:
    def __init__(self, xml_path,  visualize_joints=True):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.scn_geom_manager = None
        self.static_objects = None
        self.dynamic_objects = None
        self.visible_frames = set()
        self.show_joints = visualize_joints

    def initialize_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        self.viewer.user_scn.ngeom = 0
        self.scn_geom_manager = scnGeomManager(self.viewer)

        for i in range(self.model.ngeom):
            self.model.geom_rgba[i][3] = 0.3

    def add_static_objects(self, objects):
        """
        Initializes static objects in the scene.

        Args:
            objects: List of tuples containing object types ('mesh', 'box') and parameters.
        """
        self.static_objects = objects

    def toggle_frame(self, frame_name):
        if frame_name in self.visible_frames:
            self.visible_frames.remove(frame_name)
        else:
            self.visible_frames.add(frame_name)

    def toggle_joint_frames(self):
        self.show_joints = not self.show_joints

    def update_scene(self):
        mujoco.mj_fwdPosition(self.model, self.data)
        self.scn_geom_manager.clear()
        self._add_world_frame()
        self._add_visible_frames()
        if self.show_joints:
            self._add_joint_frames()
        if self.static_objects is not None:
            self._initialize_static_objects()
        self.scn_geom_manager.update()
        self.viewer.sync()

    def _add_world_frame(self):
        self.scn_geom_manager.add_frame([0.0, 0.0, 0.0], np.eye(3), 0.1)

    def _add_visible_frames(self):
        for frame_name in self.visible_frames:
            self._add_site_frame(frame_name)

    def _add_joint_frames(self):
        for i in range(self.model.njnt):
            pos = self.data.xanchor[i]
            xmat = self.data.xmat[i]
            joint_axis = self.data.xaxis[i]
            self.scn_geom_manager.add_frame(pos, xmat, 0.1, joint_axis)

    def _add_site_frame(self, site_name):
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        pos = self.data.site_xpos[obj_id]
        rot = self.data.site_xmat[obj_id].reshape(3, 3)
        self.scn_geom_manager.add_frame(pos, rot, 0.1)

    def _initialize_static_objects(self):
        for obj_type, params in self.static_objects:
            if obj_type == 'box':
                pos, size, rotation, rgba = params
                self.scn_geom_manager.add_box(pos, size, rotation, rgba)
            elif obj_type == 'mesh':
                mesh_path, pos, rotation, rgba = params
                self.scn_geom_manager.add_mesh(mesh_path, pos, rotation, rgba)

    def visualize_configuration(self, base_pose, joint_positions):
        self.initialize_viewer()
        self.data.qpos[0:7] = base_pose
        self.data.qpos[7:] = joint_positions
        
        while self.viewer.is_running():
            self.update_scene()

    def visualize_trajectory(self, trajectory):
        self.initialize_viewer()
        trajectory_index = 0
        
        while self.viewer.is_running():
            if trajectory_index >= len(trajectory):
                trajectory_index = 0

            base_pose, joint_positions = trajectory[trajectory_index]
            self.data.qpos[0:7] = base_pose
            self.data.qpos[7:] = joint_positions 
            self.update_scene()
            trajectory_index += 1
            # time.sleep(dt_trajectory[trajectory_index])

# Usage example
if __name__ == "__main__":
    xml_path = "/home/michal/projects/tamp/tamp_warm_start/g1_description/g1_29dof.xml"
    
    visualizer = KinematicRobotVisualizer(xml_path)
    
    static_objects = [
        ('box', ([0, 0, 0.5], [0.1, 0.1, 0.1], np.eye(3), [0.5, 0.5, 0.5, 1.0])),
    ]
    visualizer.add_static_objects(static_objects)

    # Configure visualization
    # visualizer.toggle_joint_frames()
    visualizer.toggle_frame("left_sole")
    visualizer.toggle_frame("left_sole_FL")
    visualizer.toggle_frame("left_sole_FR")
    visualizer.toggle_frame("left_sole_RL")
    visualizer.toggle_frame("left_sole_RR")
    visualizer.toggle_frame("right_sole")
    visualizer.toggle_frame("right_sole_FL")
    visualizer.toggle_frame("right_sole_FR")
    visualizer.toggle_frame("right_sole_RL")
    visualizer.toggle_frame("right_sole_RR")
    visualizer.toggle_frame("left_palm")
    visualizer.toggle_frame("right_palm")

    # Visualize a single pose
    robot_base_pose = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]
    joint_positions = [0] * visualizer.model.nu  # Example joint positions
    visualizer.visualize_configuration(robot_base_pose, joint_positions)

    # Visualize a trajectory
    # trajectory = [(base_pose1, joint_positions1), (base_pose2, joint_positions2), ...]
    # visualizer.visualize_trajectory(trajectory)