from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d, quaternion_to_matrix, matrix_to_euler_angles
import torch
import numpy as np
from typing import Tuple
from aerial_gym.utils.math import *
from aerial_gym.utils.logging import CustomLogger
from gym.spaces import Dict, Box

logger = CustomLogger("position_setpoint_task_sim2real_px4_arm")

class PositionSetpointTaskSim2RealPX4Arm(BaseTask):
    def __init__(self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None):
        # Overwrite params logic
        if seed is not None: task_config.seed = seed
        if num_envs is not None: task_config.num_envs = num_envs
        if headless is not None: task_config.headless = headless
        if device is not None: task_config.device = device
        if use_warp is not None: task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device

        
        logger.info("Building environment for position setpoint task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
            )
        )
        
        # Build Environment
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        if not self.task_config.headless:
            self.sim_env.IGE_env.viewer.sync_frame_time = True

        # Buffers
        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.joint_actions = torch.zeros(
            (self.sim_env.num_envs, 3),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.raw_actions = torch.zeros_like(self.actions)
        self.prev_raw_actions = torch.zeros_like(self.actions)
        self.prev_dist = torch.zeros((self.sim_env.num_envs), device=self.device)
        self.counter = 0

        self.target_ee_position = torch.zeros((self.sim_env.num_envs, 3), device=self.device, requires_grad=False)

        self.target_ee_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.sim_env.num_envs, 1)
        
        # Target for EE Orientation: Horizontal (90 deg rotation around X-axis)
        # Quat [x, y, z, w] = [0.7071, 0.0, 0.0, 0.7071]
        # self.target_ee_quat = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.device).expand(self.sim_env.num_envs, 4)

        # Buffer for the drone's position error (to help it stay near the goal)
        self.prev_ee_dist = torch.zeros((self.sim_env.num_envs), device=self.device)

        
        # Observations
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)
        self.prev_pos_error = torch.zeros((self.sim_env.num_envs, 3), device=self.device, requires_grad=False)

        # Spaces
        self.observation_space = Dict({"observations": Box(low=-1.0, high=1.0, shape=(self.task_config.observation_space_dim,), dtype=np.float32)})
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.task_config.action_space_dim,), dtype=np.float32)
        self.num_envs = self.sim_env.num_envs
        
        # Task Obs Dictionary
        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

        self.counter = 0

        #=====TUNING=====#

        # --- TUNING LOGS ---
        self.tuning_step = 0
        self.log_size = 500  # Collect 500 steps then plot
        # Arrays to store history [Step, Value]
        self.log_cmd_roll = []
        self.log_act_roll = []
        self.log_cmd_pitch = []
        self.log_act_pitch = []
        self.log_cmd_yaw = []
        self.log_act_yaw = []
        self.log_cmd_arm_pos0 = []
        self.log_act_arm_pos0 = []
        self.log_cmd_arm_pos1 = []
        self.log_act_arm_pos1 = []
        self.log_cmd_arm_pos2 = []
        self.log_act_arm_pos2 = []

         #=====TUNING=====#

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        # Randomize EE target position: X, Y in [-2, 2], Z in [1, 2.5]
        self.target_ee_position[:, 0:2] = 0.0 
        self.target_ee_position[:, 2] = 0.0 
        # self.target_ee_position[:, 0:2] = ((torch.rand((self.sim_env.num_envs, 2), device=self.device) - 0.5) * 4.0)
        # self.target_ee_position[:, 2] = torch.rand((self.sim_env.num_envs), device=self.device) * 1.5 + 1.0

        # rand_quat = torch.randn((self.sim_env.num_envs, 4), device=self.device)
        # self.target_ee_quat = torch.nn.functional.normalize(rand_quat, p=2, dim=-1)

        random_roll = (torch.rand(self.sim_env.num_envs, device=self.device) * 0.4) - 0.2
        # Pitch: [-0.5, 0.5] 0.2
        random_pitch = (torch.rand(self.sim_env.num_envs, device=self.device) * 1.0) - 0.2
        # Yaw: [-3.14, 3.14]
        random_yaw = (torch.rand(self.sim_env.num_envs, device=self.device) * 6.28) - 3.14

        # 2. Convert to Quaternions (x, y, z, w)
        quats = quat_from_euler_xyz(random_roll, random_pitch, random_yaw)
        
        # 4. Update the buffer (Assign directly)
        self.target_ee_quat[:] = quats

        self.sim_env.reset()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.target_ee_position[env_ids, 0:2] = 0.0 
        self.target_ee_position[env_ids, 2] = 0.0
        # Randomize EE target position: X, Y in [-2, 2], Z in [1, 2.5]
        # self.target_ee_position[env_ids, 0:2] = (torch.rand((len(env_ids), 2), device=self.device) - 0.5) * 4.0
        # self.target_ee_position[env_ids, 2] = torch.rand((len(env_ids)), device=self.device) * 1.5 + 1.0

        num_resets = len(env_ids)

        random_roll = (torch.rand(num_resets, device=self.device) * 0.4) - 0.2
        # Pitch: [-0.5, 0.5]
        random_pitch = (torch.rand(num_resets, device=self.device) * 1.0) - 0.2
        # Yaw: [-3.14, 3.14]
        random_yaw = (torch.rand(num_resets, device=self.device) * 6.28) - 3.14

        # 2. Convert to Quaternions (x, y, z, w)
        quats = quat_from_euler_xyz(random_roll, random_pitch, random_yaw)
        
        # 4. Update the buffer (Assign directly)
        self.target_ee_quat[env_ids] = quats

        # rand_quat = torch.randn((len(env_ids), 4), device=self.device)
        # self.target_ee_quat[env_ids] = torch.nn.functional.normalize(rand_quat, p=2, dim=-1)
        self.sim_env.reset_idx(env_ids)
        self.prev_pos_error[env_ids] = 0.0

    def render(self, mode="human"):
        return None

    def step(self, actions):
        import sys
        self.counter += 1

        #=====ACTUAL CODE=====#

        self.raw_actions = torch.clamp(actions, -1, 1)

        # 2. Scale Actions
        self.actions = self.task_config.process_actions_for_task(
            actions, self.task_config.action_limit_min, self.task_config.action_limit_max
        )

        # 3. Split and Apply
        drone_actions = self.actions[:, :7]
        # joint_actions = self.joint_actions
        joint_actions = self.actions[:, 7:]
        # drone_actions[:, 0] = -9.81
        # drone_actions[:, 1] = 0
        # drone_actions[:, 2] = 0
        # drone_actions[:, 3] = 0
        # joint_actions[:, 0] = 1.0
        # joint_actions[:, 1] = 1.0
        # joint_actions[:, 1] = 1.0


        # print(f"Drone Actions: {self.actions[0, :7]}")
        # print(f"Joint Actions: {self.actions[0, 7:]}")


        self.sim_env.robot_manager.robot.set_dof_position_targets(joint_actions)
        self.sim_env.robot_manager.robot.set_dof_velocity_targets(torch.zeros_like(joint_actions))

        self.sim_env.step(actions=drone_actions)

        # print(f"\nCOMMANDED BODY RATES: {drone_actions[0, 1:]}")
        # print(f"""ACTUAL BODY RATES: {self.obs_dict["robot_body_angvel"][0,:]}""")
        # print("\n")
    
        # VISUALIZATION CALL
        if not self.task_config.headless:
            self.draw_debug_targets()

        ee_pos, ee_quat = calculate_ee_state(
            self.obs_dict["robot_position"],
            self.obs_dict["robot_orientation"],
            self.obs_dict["dof_state_tensor"][..., 0]
        )

        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        self.infos = {}  # self.obs_dict["infos"]

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        self.prev_actions = self.actions.clone()
        self.prev_raw_actions = self.raw_actions.clone()
        self.prev_pos_error = self.target_ee_position - ee_pos

        #=====ACTUUAL CODE=====#

        #=====TUNING=====#

        # period = 120  #120 for sine 100 for square
        # amplitude = 1.0
        # # target_val = -amplitude * np.sin((self.counter / period) * 2 * np.pi)
        # target_val = 1 if (self.counter % (period*2)) < period else -1
        # # Create override actions
        # # Drone: 0 thrust, 0 rates (It is fixed anyway)
        # drone_actions = torch.zeros((self.sim_env.num_envs, 4), device=self.device)

        # drone_actions[:, 0] = -9.81  # Zero Thrust (Keep motors idle)
        # drone_actions[:, 1] = target_val # Roll Rate
        # drone_actions[:, 2] = target_val*0   # Pitch Rate
        # drone_actions[:, 3] = target_val*0   # Yaw Rate
        
        # # Arm: Joint 1 = target, Joint 2 = 0
        # joint_actions = torch.zeros((self.sim_env.num_envs, 3), device=self.device)
        # joint_actions[:, 0] = (target_val-1.57)*0
        # joint_actions[:, 1] = target_val*0
        # joint_actions[:, 2] = target_val*0

        
        # # Apply to Robot
        # # 1. Position Target
        # self.sim_env.robot_manager.robot.set_dof_position_targets(joint_actions)
        # # 2. Velocity Target (Must be 0 for Position Control)
        # self.sim_env.robot_manager.robot.set_dof_velocity_targets(torch.zeros_like(joint_actions))

        # # 3. Step Physics
        # self.sim_env.step(actions=drone_actions)

        # # --- DATA LOGGING FOR TUNING ---
        # # Only log Environment 0
        # self.tuning_step += 1

        # self.log_cmd_arm_pos0.append(target_val-1.57)
        # self.log_cmd_arm_pos1.append(target_val)
        # self.log_cmd_arm_pos2.append(target_val)

        
        # # 1. Log Rates (Roll Rate / Rate X)
        # # drone_actions: [Thrust, RollRate, PitchRate, YawRate] -> Index 1 is RollRate
        # cmd_roll = drone_actions[0, 1].item() 
        # act_roll = self.obs_dict["robot_euler_angles"][0, 0].item()
        # cmd_pitch = drone_actions[0, 2].item() 
        # act_pitch = self.obs_dict["robot_euler_angles"][0, 1].item()
        # cmd_yaw= drone_actions[0, 3].item() 
        # act_yaw= self.obs_dict["robot_euler_angles"][0, 2].item()

        
        # self.log_cmd_roll.append(cmd_roll)
        # self.log_act_roll.append(act_roll)
        # self.log_cmd_pitch.append(cmd_pitch)
        # self.log_act_pitch.append(act_pitch)
        # self.log_cmd_yaw.append(cmd_yaw)
        # self.log_act_yaw.append(act_yaw)

        # # 2. Log Arm Joint 1 Position
        # # joint_actions: [Joint1, Joint2] -> Index 0
        # cmd_joint0 = self.actions[0, 4].item() # 4 is start of arm actions
        # act_joint0 = self.obs_dict["dof_state_tensor"][0, 0, 0].item() # Env 0, Joint 0, Pos
        # cmd_joint1 = self.actions[0, 5].item() # 4 is start of arm actions
        # act_joint1 = self.obs_dict["dof_state_tensor"][0, 1, 0].item() # Env 0, Joint 0, Po
        # cmd_joint2 = self.actions[0, 6].item() # 4 is start of arm actions
        # act_joint2 = self.obs_dict["dof_state_tensor"][0, 2, 0].item() # Env 0, Joint 0, Po
        
        # # self.log_cmd_arm_pos.append(cmd_joint)
        # self.log_act_arm_pos0.append(act_joint0)
        # self.log_act_arm_pos1.append(act_joint1)
        # self.log_act_arm_pos2.append(act_joint2)


        # # 3. Plot and Reset every N steps
        # if self.tuning_step % self.log_size == 0:
        #     self.plot_tuning_data()
        #     # Clear logs
        #     self.log_cmd_roll = []
        #     self.log_act_roll = []
        #     self.log_cmd_pitch = []
        #     self.log_act_pitch = []
        #     self.log_cmd_yaw = []
        #     self.log_act_yaw = []
        #     self.log_cmd_arm_pos0 = []
        #     self.log_act_arm_pos0 = []
        #     self.log_cmd_arm_pos1 = []
        #     self.log_act_arm_pos1 = []
        #     self.log_cmd_arm_pos2 = []
        #     self.log_act_arm_pos2 = []
        #=====TUNING=====#

        return self.get_return_tuple()

    def draw_debug_targets(self):

        viewer = self.sim_env.IGE_env.viewer.viewer
        gym = self.sim_env.IGE_env.gym
        if viewer is None:
            return

        gym.clear_lines(viewer)

        # from isaacgym.torch_utils import quat_to_ma
        import numpy as np

        # Crosshair size
        s = 0.2

        # Compute FK EE pose
        ee_pos, ee_quat = calculate_ee_state(
            self.obs_dict["robot_position"],
            self.obs_dict["robot_orientation"],
            self.obs_dict["dof_state_tensor"][..., 0]
        )

        # Draw for only first few envs (performance)
        num_draw = min(4, self.num_envs)

        for i in range(num_draw):

            # --- Target pose ---
            target_pos = self.target_ee_position[i].cpu().numpy()
            target_quat = self.target_ee_quat[i]

            # --- FK EE pose ---
            ee_p = ee_pos[i].cpu().numpy()
            ee_q = ee_quat[i]

            env_handle = self.sim_env.IGE_env.env_handles[i]

            # Convert quaternions → rotation matrices
            R_target = quaternion_to_matrix(target_quat.unsqueeze(0))[0].cpu().numpy()
            R_ee     = quaternion_to_matrix(ee_q.unsqueeze(0))[0].cpu().numpy()

            # Local axes scaled
            local_axes = np.eye(3) * s

            # Rotate axes
            target_axes = (R_target @ local_axes.T).T
            ee_axes     = (R_ee     @ local_axes.T).T

            # Build line vertices
            verts_target = []
            verts_ee     = []

            for axis in target_axes:
                p1 = target_pos - axis
                p2 = target_pos + axis
                verts_target.append([*p1, *p2])

            for axis in ee_axes:
                p1 = ee_p - axis
                p2 = ee_p + axis
                verts_ee.append([*p1, *p2])

            verts_target = np.array(verts_target, dtype=np.float32)
            verts_ee     = np.array(verts_ee, dtype=np.float32)

            # Axis colors (X=Red, Y=Green, Z=Blue (North))
            colors = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=np.float32)

            # Draw target frame
            gym.add_lines(viewer, env_handle, 3, verts_target, colors)

            # Draw EE frame
            gym.add_lines(viewer, env_handle, 3, verts_ee, colors)

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (self.task_obs, self.rewards, self.terminations, self.truncations, {})

    def process_obs_for_task(self):
        # 1. Drone Ground Truths
        obs_drone_pos = self.target_ee_position - self.obs_dict["robot_position"] 
        
        # Drone 6D Orientation (Keep this if you still want drone state)
        quat_raw = self.obs_dict["robot_orientation"]
        quat_norm = torch.nn.functional.normalize(quat_raw, p=2, dim=-1, eps=1e-8)
        or_quat = quat_norm[:, [3, 0, 1, 2]] # xyzw -> wxyz
        obs_drone_rot6d = matrix_to_rotation_6d(quaternion_to_matrix(or_quat))

        # print("actual attitude: ", self.obs_dict["robot_euler_angles"][0])

        # 2. End Effector States (Using your new calculate_ee_state)
        ee_pos, ee_quat = calculate_ee_state(
            self.obs_dict["robot_position"],
            self.obs_dict["robot_orientation"],
            self.obs_dict["dof_state_tensor"][..., 0]
        )
        
        # Calculate Position Error
        obs_ee_pos_err = self.target_ee_position - ee_pos

        # --- NEW: CALCULATE RELATIVE ROTATION MATRIX ---
        
        # 1. Convert EE Quat to Matrix (Make sure to normalize first)
        ee_quat = torch.nn.functional.normalize(ee_quat, p=2, dim=-1, eps=1e-8)
        ee_quat_p3d = ee_quat[:, [3, 0, 1, 2]] # xyzw -> wxyz
        R_ee = quaternion_to_matrix(ee_quat_p3d) # Shape: (Num_Envs, 3, 3)

        # 2. Convert Target Quat to Matrix
        target_quat = torch.nn.functional.normalize(self.target_ee_quat, p=2, dim=-1, eps=1e-8)
        target_quat_p3d = target_quat[:, [3, 0, 1, 2]] # xyzw -> wxyz
        R_target = quaternion_to_matrix(target_quat_p3d) # Shape: (Num_Envs, 3, 3)

        # 3. Compute Relative Rotation: R_error = R_ee.T @ R_target
        # This gives the error in the End-Effector's local frame
        R_ee_transposed = R_ee.transpose(1, 2)
        R_relative = torch.bmm(R_ee_transposed, R_target) # Batch Matrix Multiply
        
        # 4. Flatten to (Num_Envs, 9)
        obs_relative_rot_matrix = R_relative.reshape(self.num_envs, 9)
        
        # 0-3: Drone Pos Error
        self.task_obs["observations"][:, 0:3]   = obs_drone_pos / 10.0
        
        # 3-9: Drone Rot 6D
        self.task_obs["observations"][:, 3:9]   = obs_drone_rot6d
        
        # 9-12: Drone Lin Vel
        self.task_obs["observations"][:, 9:12]  = self.obs_dict["robot_linvel"] / 10.0
        
        # 12-15: Drone Ang Vel
        self.task_obs["observations"][:, 12:15] = self.obs_dict["robot_body_angvel"]
        
        # 15-18: EE Pos Error
        self.task_obs["observations"][:, 15:18] = obs_ee_pos_err / 10.0
        
        # 18-27: Relative Rotation Matrix (9 values)
        self.task_obs["observations"][:, 18:27] = obs_relative_rot_matrix

        # 27-29: Joint Pos (Shifted from 30)
        self.task_obs["observations"][:, 27:30] = self.obs_dict["dof_state_tensor"][..., 0] 
        
        # 29-31: Joint Vel (Shifted from 32)
        self.task_obs["observations"][:, 30:33] = self.obs_dict["dof_state_tensor"][..., 1] 
        
        # 31-37: Previous Actions (Shifted from 34)
        # self.task_obs["observations"][:, 31:37] = self.prev_raw_actions

        # print(f"traget_ee_position: {self.target_ee_position[0]}")
        # print("OBSERVATIONS (drone_pos_rel): ", self.task_obs["observations"][0, 0:3])
        # # print("OBSERVATIONS (drne_ori): ", self.task_obs["observations"][0, 3:9])
        # # print(f"OBSERVATIONS (Drone_att_euler): {self.obs_dict["robot_euler_angles"][0]}")
        # print("actual attitude: ", self.obs_dict["robot_euler_angles"][0])
        # print("OBSERVATIONS (lin vel: ", self.task_obs["observations"][0, 9:12])
        # print("OBSERVATIONS (ang vel): ", self.task_obs["observations"][0, 12:15])
        # print("OBSERVATIONS (ee_pos_rel): ", self.task_obs["observations"][0, 15:18])
        # print("OBSERVATIONS (ee_ori): ", self.task_obs["observations"][0, 18:27])
        # print("OBSERVATIONS (joint pos): ", self.task_obs["observations"][0, 27:30])
        # print("OBSERVATIONS (joint vel): ", self.task_obs["observations"][0, 30:33])
        
        # Convert the first matrix in the batch (R_target[0]) to Euler angles
        target_euler_rad = matrix_to_euler_angles(R_target[0], "XYZ")
        # Convert from radians to degrees for easy reading
        target_euler_deg = torch.rad2deg(target_euler_rad)
        # print(f"target_ee_orientation (Roll, Pitch, Yaw) in degrees: {target_euler_rad}")

        # Do the same for the actual End-Effector orientation
        actual_euler_rad = matrix_to_euler_angles(R_ee[0], "XYZ")
        actual_euler_deg = torch.rad2deg(actual_euler_rad)
        # print(f"actual_ee_orientation (Roll, Pitch, Yaw) in degrees: {actual_euler_rad}")




        # print("\n")

        # self.task_obs["observations"] = torch.nan_to_num(
        #     self.task_obs["observations"], 
        #     nan=0.0, 
        #     posinf=0.0, 
        #     neginf=0.0
        # )

    def compute_rewards_and_crashes(self, obs_dict):
        # 1. First, calculate the End Effector Position using FK
        ee_pos, ee_quat = calculate_ee_state(
            obs_dict["robot_position"],
            obs_dict["robot_orientation"],
            obs_dict["dof_state_tensor"][..., 0]
        )

        ee_vel = calculate_ee_velocity(
            obs_dict["robot_linvel"], 
            obs_dict["robot_body_angvel"], 
            obs_dict["robot_orientation"], 
            obs_dict["dof_state_tensor"][..., 0], 
            obs_dict["dof_state_tensor"][..., 1], 
        )

        # return compute_reward(
        #         ee_pos,
        #         ee_quat,
        #         self.target_ee_position,
        #         self.target_ee_quat,
        #         obs_dict["robot_orientation"],
        #         obs_dict["robot_linvel"],
        #         obs_dict["robot_body_angvel"],
        #         obs_dict["dof_state_tensor"][..., 1], # joint_vel
        #         obs_dict["dof_state_tensor"][..., 0], # joint_angles <--- ADDED THIS
        #         self.actions,
        #         self.prev_actions,
        #         self.raw_actions,
        #         self.prev_raw_actions,
        #         obs_dict["crashes"],
        #         self.task_config.crash_dist
        #     )

        robot_position = obs_dict["robot_position"]
        robot_linvel = obs_dict["robot_linvel"]
        target_position = self.target_ee_position
        robot_orientation = obs_dict["robot_orientation"]
        angular_velocity = obs_dict["robot_body_angvel"]
        joint_velocity = obs_dict["dof_state_tensor"][..., 1]
        action_input = self.actions

        pos_error_frame = target_position - ee_pos #robot_position

        return compute_reward(
            pos_error_frame,
            robot_orientation,
            robot_linvel,
            angular_velocity,
            joint_velocity,
            ee_vel,
            ee_quat,
            self.target_ee_quat,
            obs_dict["crashes"],
            action_input.clone(),
            self.prev_actions,
            self.prev_pos_error,
            self.task_config.crash_dist
        )

        # return compute_reward(
        #     pos_error=pos_error_frame,
        #     ee_quat=ee_quat,
        #     target_ee_quat=self.target_ee_quat,
        #     actions=action_input.clone(),
        #     prev_actions=self.prev_actions,
        #     crashes=self.obs_dict["crashes"],
        #     crash_dist=self.task_config.crash_dist
        # )

    def plot_tuning_data(self):
        import matplotlib.pyplot as plt
        
        # Create a figure with 2 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        
        # Plot Drone Roll Rate
        ax1.plot(self.log_cmd_roll, 'g--', label='Commanded Roll Rate', linewidth=2)
        ax1.plot(self.log_act_roll, 'r-', label='Actual Roll Rate', alpha=0.7)
        ax1.set_title("Drone Roll Rate Response")
        ax1.set_ylabel("Rad/s")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.log_cmd_pitch, 'g--', label='Commanded Pitch Rate', linewidth=2)
        ax2.plot(self.log_act_pitch, 'r-', label='Actual Pitch Rate', alpha=0.7)
        ax2.set_title("Drone Pitch Rate Response")
        ax2.set_ylabel("Rad/s")
        ax2.legend()
        ax2.grid(True)

        ax3.plot(self.log_cmd_yaw, 'g--', label='Commanded Yaw Rate', linewidth=2)
        ax3.plot(self.log_act_yaw, 'r-', label='Actual Yaw Rate', alpha=0.7)
        ax3.set_title("Drone Yaw Rate Response")
        ax3.set_ylabel("Rad/s")
        ax3.legend()
        ax3.grid(True)

        # Plot Arm Joint Position
        # ax1.plot(self.log_cmd_arm_pos0, 'g--', label='Commanded Joint 0', linewidth=2)
        # ax1.plot(self.log_act_arm_pos0, 'r-', label='Actual Joint 0', alpha=0.7)
        # ax1.set_title("Arm Joint 0 Position Response")
        # ax1.set_ylabel("Radians")
        # ax1.legend()
        # ax1.grid(True)

        # ax2.plot(self.log_cmd_arm_pos1, 'g--', label='Commanded Joint 1', linewidth=2)
        # ax2.plot(self.log_act_arm_pos1, 'r-', label='Actual Joint 1', alpha=0.7)
        # ax2.set_title("Arm Joint 1 Position Response")
        # ax2.set_ylabel("Radians")
        # ax2.legend()
        # ax2.grid(True)
        
        # ax3.plot(self.log_cmd_arm_pos2, 'g--', label='Commanded Joint 2', linewidth=2)
        # ax3.plot(self.log_act_arm_pos2, 'r-', label='Actual Joint 2', alpha=0.7)
        # ax3.set_title("Arm Joint 2 Position Response")
        # ax3.set_ylabel("Radians")
        # ax3.legend()
        # ax3.grid(True)
        
        print("Displaying Plot... Close window to continue simulation.")
        plt.show() # This pauses the sim until you close the window
# ========================================================
# REWARD FUNCTIONS - FIXED JIT SYNTAX
# ========================================================

@torch.jit.script
def calculate_ee_state(drone_pos: torch.Tensor, drone_quat: torch.Tensor, joint_angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates End Effector position based on the updated X500 PX4 4-Link Arm URDF.
    
    URDF SPECS:
    - Base Mount: (0, 0, -0.025). Pitch = 180 deg.
    - Joint 1 (Shoulder): (0, 0, 0.03796) from Mount. Pitch = 90 deg. Axis = (0, 1, 0)
    - Joint 2 (Elbow): (0.03, 0, 0.23682) from J1. Axis = (0, 1, 0)
    - Joint 3 (Twist): (0, 0, 0.21599) from J2. Axis = (0, 0, 1)
    """
    theta1 = joint_angles[:, 0]
    theta2 = joint_angles[:, 1]
    theta3 = joint_angles[:, 2]

    # URDF Lengths and Offsets
    j1_x = 0.03
    j1_z = 0.23682
    j2_z = 0.21599
    
    # Base mount is Z = -0.025.
    # L1_to_L2 is Z = 0.03796 in L1 frame. 
    # Because L1 is pitched 180 deg (upside down), L1's Z points in the -Z direction of the drone.
    # Total Base Z Offset = -0.025 - 0.03796 = -0.06296
    base_z_offset = -0.06296

    # --- POSITION (Local X-Z Plane) ---
    # Derived from transformation matrices passing through the 180 and 90 degree pitch offsets
    ee_x_local = j1_x * torch.sin(theta1) - j1_z * torch.cos(theta1) - j2_z * torch.cos(theta1 + theta2)
    ee_y_local = torch.zeros_like(theta1)
    ee_z_local = base_z_offset + j1_x * torch.cos(theta1) + j1_z * torch.sin(theta1) + j2_z * torch.sin(theta1 + theta2)
    
    # Construct Local Offset Vector
    local_offset = torch.zeros_like(drone_pos)
    local_offset[:, 0] = ee_x_local
    local_offset[:, 2] = ee_z_local
    
    # Apply Drone Orientation to get World Position
    world_ee_pos = drone_pos + quat_apply(drone_quat, local_offset)
    
    # --- ORIENTATION ---
    # A. Swing Quaternion (Rotation around +Y axis)
    # Total static pitch offset = 180 deg + 90 deg = 270 deg (1.5 * PI)
    pi_1_5 = 4.71238898  # 1.5 * math.pi
    half_angle_swing = (pi_1_5 + theta1 + theta2) * 0.5
    
    swing_quat = torch.zeros_like(drone_quat)
    swing_quat[:, 1] = torch.sin(half_angle_swing)  # Y-axis
    swing_quat[:, 3] = torch.cos(half_angle_swing)  # W
    
    # B. Twist Quaternion (Rotation around +Z axis by theta3)
    half_angle_twist = theta3 * 0.5
    twist_quat = torch.zeros_like(drone_quat)
    twist_quat[:, 2] = torch.sin(half_angle_twist)  # Z-axis
    twist_quat[:, 3] = torch.cos(half_angle_twist)  # W

    # Combine rotations: Swing first, then Twist
    arm_local_quat = quat_mul(swing_quat, twist_quat)

    # Combine with Drone Rotation
    world_ee_quat = quat_mul(drone_quat, arm_local_quat)

    return world_ee_pos, world_ee_quat

@torch.jit.script
def calculate_ee_velocity(
    drone_linvel: torch.Tensor,      # World Frame
    drone_angvel: torch.Tensor,      # Body Frame
    drone_quat: torch.Tensor,        # [x,y,z,w]
    joint_angles: torch.Tensor, 
    joint_velocities: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates End Effector linear velocity based on the updated URDF chain.
    """
    theta1 = joint_angles[:, 0]
    theta2 = joint_angles[:, 1]
    d_theta1 = joint_velocities[:, 0]
    d_theta2 = joint_velocities[:, 1]
    
    # URDF Lengths and Offsets
    j1_x = 0.03
    j1_z = 0.23682
    j2_z = 0.21599
    base_z_offset = -0.06296

    # --- 1. Velocity due to Joint Motion (Relative to Drone) ---
    # Partial derivatives of the position equations w.r.t time (Chain Rule)
    
    # dX/dt
    v_x_local = (j1_x * torch.cos(theta1) * d_theta1 + 
                 j1_z * torch.sin(theta1) * d_theta1 + 
                 j2_z * torch.sin(theta1 + theta2) * (d_theta1 + d_theta2))
    
    v_y_local = torch.zeros_like(theta1)
    
    # dZ/dt
    v_z_local = (-j1_x * torch.sin(theta1) * d_theta1 + 
                 j1_z * torch.cos(theta1) * d_theta1 + 
                 j2_z * torch.cos(theta1 + theta2) * (d_theta1 + d_theta2))
    
    # Construct Local Velocity Vector
    v_arm_local = torch.zeros_like(drone_linvel)
    v_arm_local[:, 0] = v_x_local
    v_arm_local[:, 2] = v_z_local
    
    # Rotate joint velocity to World Frame
    v_arm_world = quat_apply(drone_quat, v_arm_local)

    # --- 2. Velocity due to Drone Rotation (Tangential Velocity) ---
    # V_tangential = omega (Body) cross r (Body)
    p_local = torch.zeros_like(drone_linvel)
    p_local[:, 0] = j1_x * torch.sin(theta1) - j1_z * torch.cos(theta1) - j2_z * torch.cos(theta1 + theta2)
    p_local[:, 2] = base_z_offset + j1_x * torch.cos(theta1) + j1_z * torch.sin(theta1) + j2_z * torch.sin(theta1 + theta2)

    # Tangential Velocity in Body Frame
    v_tan_body = torch.cross(drone_angvel, p_local, dim=1)
    
    # Rotate Tangential to World
    v_tan_world = quat_apply(drone_quat, v_tan_body)
    
    # --- 3. Total EE Velocity in World Frame ---
    # V_ee = V_drone_linear + V_arm_extension_rotated + V_tangential_rotation
    ee_vel_world = drone_linvel + v_arm_world + v_tan_world
    
    return ee_vel_world

# @torch.jit.script
# def calculate_ee_state(drone_pos: torch.Tensor, drone_quat: torch.Tensor, joint_angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Calculates End Effector position based on X500 3-DOF Arm URDF.
    
#     URDF SPECS:
#     - Mount: (0, 0.1, -0.05) from Drone Center.
#     - Joint 1 (Shoulder): (0, 0, 0.0012) from Mount. Axis: (-1, 0, 0).
#     - Joint 2 (Elbow): (0, 0, -0.1795) from J1. Axis: (-1, 0, 0).
#     - Joint 3 (Twist): (0, 0, -0.2) from J2. Axis: (0, 0, 1).
#     """
#     theta1 = joint_angles[:, 0]
#     theta2 = joint_angles[:, 1]
#     theta3 = joint_angles[:, 2]

    
    
#     # URDF offsets
#     l1 = 0.1795
#     l2 = 0.20
#     mount_z = -0.05
#     mount_y = 0.1
#     j1_z_offset = 0.0012
#     z_offset = mount_z + j1_z_offset

#     theta_swing = theta1 + theta2
    
#     # In Local Frame (X-forward, Z-up):
#     # Axis is -X. Positive Theta rotates Y towards -Z.
#     # Standard pendulum math with phase shift.
    
#     # Joint 1 Position relative to Mount
#     # y1 = -l1 * sin(theta1)
#     # z1 = -l1 * cos(theta1)
    
#     # Joint 2 Position relative to Joint 1
#     # y2 = -l2 * sin(theta1 + theta2)
#     # z2 = -l2 * cos(theta1 + theta2)
    
#     ee_y_local = mount_y - l1 * torch.sin(theta1) - l2 * torch.sin(theta1 + theta2)
#     ee_z_local = z_offset - l1 * torch.cos(theta1) - l2 * torch.cos(theta1 + theta2)
    
#     # Construct Local Offset Vector
#     local_offset = torch.zeros_like(drone_pos)
#     local_offset[:, 1] = ee_y_local
#     local_offset[:, 2] = ee_z_local
    
#     # Apply Drone Orientation
#     world_ee_pos = drone_pos + quat_apply(drone_quat, local_offset)
    
#     #### ORIENTATION #####
#     # A. Swing Quaternion (Rotation around -X axis by theta_swing)
#     half_angle_swing = theta_swing * 0.5
#     swing_quat = torch.zeros_like(drone_quat)
#     # Axis is (-1, 0, 0) -> x = sin(a/2)*-1
#     swing_quat[:, 0] = -torch.sin(half_angle_swing) 
#     swing_quat[:, 3] = torch.cos(half_angle_swing)
    
#     # B. Twist Quaternion (Rotation around +Z axis by theta3)
#     # This is applied in the LOCAL frame of the arm tip (which is already swung)
#     half_angle_twist = theta3 * 0.5
#     twist_quat = torch.zeros_like(drone_quat)
#     # Axis is (0, 0, 1) -> z = sin(a/2)*1
#     twist_quat[:, 2] = torch.sin(half_angle_twist)
#     twist_quat[:, 3] = torch.cos(half_angle_twist)

#     # Construct the local quaternion for the arm's rotation [x, y, z, w]
#     # Formula: [sin(a/2)*axis_x, sin(a/2)*axis_y, sin(a/2)*axis_z, cos(a/2)]
#     # Axis is (-1, 0, 0)
#     arm_local_quat = torch.zeros_like(drone_quat)
#     arm_local_quat = quat_mul(swing_quat, twist_quat)

#     # Combine rotations: (Drone Rotation) * (Arm Rotation)
#     world_ee_quat = quat_mul(drone_quat, arm_local_quat)

#     return world_ee_pos, world_ee_quat

# @torch.jit.script
# def calculate_ee_velocity(
#     drone_linvel: torch.Tensor,      # World Frame
#     drone_angvel: torch.Tensor,      # Body Frame
#     drone_quat: torch.Tensor,        # [x,y,z,w]
#     joint_angles: torch.Tensor, 
#     joint_velocities: torch.Tensor,
# ) -> torch.Tensor:
    
#     # --- 1. Velocity due to Joint Motion (Relative to Drone) ---
#     theta1 = joint_angles[:, 0]
#     theta2 = joint_angles[:, 1]
#     d_theta1 = joint_velocities[:, 0]
#     d_theta2 = joint_velocities[:, 1]
    
#     # Lengths (Must match URDF)
#     l1 = 0.18
#     l2 = 0.20

#     mount_z = -0.05
#     mount_y = 0.1
#     j1_z_offset = 0.0012
#     z_offset = mount_z + j1_z_offset

#     # Derivatives of position w.r.t time (Chain rule)
#     # y = -l1*sin(t1) - l2*sin(t1+t2)
#     # z = mount - l1*cos(t1) - l2*cos(t1+t2)
    
#     # dy/dt = -l1*cos(t1)*dt1 - l2*cos(t1+t2)*(dt1+dt2)
#     v_y_local = -l1 * torch.cos(theta1) * d_theta1 - l2 * torch.cos(theta1 + theta2) * (d_theta1 + d_theta2)
    
#     # dz/dt = l1*sin(t1)*dt1 + l2*sin(t1+t2)*(dt1+dt2)
#     v_z_local = l1 * torch.sin(theta1) * d_theta1 + l2 * torch.sin(theta1 + theta2) * (d_theta1 + d_theta2)
    
#     # Construct Local Velocity Vector
#     v_arm_local = torch.zeros_like(drone_linvel)
#     v_arm_local[:, 1] = v_y_local
#     v_arm_local[:, 2] = v_z_local
    
#     # Rotate joint velocity to World Frame
#     v_arm_world = quat_apply(drone_quat, v_arm_local)

#     # --- 2. Velocity due to Drone Rotation (Tangential Velocity) ---
#     # V_tangential = omega (cross) r
#     # We need omega in World Frame to cross with r (World Frame)
#     # Or omega (Body) cross r (Body) then rotate. Let's do Body frame.
    
#     # Cross product in Body Frame: omega_body x r_body
#     # Note: ee_pos_rel_drone passed in is usually World Frame. We need Local.
#     # Let's reconstruct local position quickly to be safe or pass it from FK.
#     # Re-calculating local P for strict correctness:
#     p_y_local = mount_y - l1 * torch.sin(theta1) - l2 * torch.sin(theta1 + theta2)
#     p_z_local = z_offset - l1 * torch.cos(theta1) - l2 * torch.cos(theta1 + theta2)
    
#     p_local = torch.zeros_like(drone_linvel)
#     p_local[:, 1] = p_y_local
#     p_local[:, 2] = p_z_local

#     # Tangential Velocity in Body Frame
#     v_tan_body = torch.cross(drone_angvel, p_local, dim=1)
    
#     # Rotate Tangential to World
#     v_tan_world = quat_apply(drone_quat, v_tan_body)
    
#     # --- 3. Total EE Velocity in World Frame ---
#     # V_ee = V_drone_linear + V_arm_extension_rotated + V_tangential_rotation
#     ee_vel_world = drone_linvel + v_arm_world + v_tan_world
    
#     return ee_vel_world

# @torch.jit.script
# def calculate_ee_velocity(
#     drone_linvel: torch.Tensor,      # World Frame
#     drone_angvel: torch.Tensor,      # Body Frame
#     drone_quat: torch.Tensor,        # [x,y,z,w]
#     joint_angles: torch.Tensor, 
#     joint_velocities: torch.Tensor,
#     ee_pos_dummy: torch.Tensor       # Unused, kept for function signature compatibility
# ) -> torch.Tensor:
    
#     #####################################
#     ######### SINGLE LINK ARM ###########
#     #####################################

#     theta1 = joint_angles[:, 0]
#     # theta2 does not affect linear velocity
    
#     d_theta1 = joint_velocities[:, 0]
    
#     # Lengths
#     l_arm = 0.25
#     mount_z = -0.10
    
#     # --- 1. Velocity due to Arm Swing (Relative to Drone) ---
#     # Position: 
#     # y = -L * sin(t1)
#     # z = mount - L * cos(t1)
    
#     # Velocity (Derivative w.r.t time):
#     # dy/dt = -L * cos(t1) * dt1
#     # dz/dt =  L * sin(t1) * dt1
    
#     v_y_local = -l_arm * torch.cos(theta1) * d_theta1
#     v_z_local = l_arm * torch.sin(theta1) * d_theta1
    
#     v_arm_local = torch.zeros_like(drone_linvel)
#     v_arm_local[:, 1] = v_y_local
#     v_arm_local[:, 2] = v_z_local
    
#     # Rotate to World Frame
#     v_arm_world = quat_apply(drone_quat, v_arm_local)

#     # --- 2. Tangential Velocity (Drone Rotation) ---
#     # r = Position of EE in Drone Frame
#     p_y_local = -l_arm * torch.sin(theta1)
#     p_z_local = mount_z - l_arm * torch.cos(theta1)
    
#     p_local = torch.zeros_like(drone_linvel)
#     p_local[:, 1] = p_y_local
#     p_local[:, 2] = p_z_local
    
#     # v_tan = omega_body x r_local
#     v_tan_body = torch.cross(drone_angvel, p_local, dim=1)
#     v_tan_world = quat_apply(drone_quat, v_tan_body)
    
#     # --- 3. Total ---
#     ee_vel_world = drone_linvel + v_arm_world + v_tan_world
    
#     return ee_vel_world

# @torch.jit.script
# def calculate_ee_state(
#     drone_pos: torch.Tensor, 
#     drone_quat: torch.Tensor, 
#     joint_angles: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor]:

#     #####################################
#     ######### SINGLE LINK ARM ###########
#     #####################################
    
#     # Joint 1: Shoulder (Swing) around -X axis
#     theta1 = joint_angles[:, 0]
    
#     # Joint 2: Twist around Z axis (Does not affect position!)
#     theta2 = joint_angles[:, 1]
    
#     # Dimensions
#     l_arm = 0.25
#     mount_z = -0.10
    
#     # --- POSITION CALCULATION ---
#     # The arm is a pendulum in the Y-Z plane.
#     # At theta1 = 0, arm is straight down (-Z).
#     # Rotation is around -X axis. Positive theta moves tip to -Y.
    
#     ee_y_local = -l_arm * torch.sin(theta1)
#     ee_z_local = mount_z - (l_arm * torch.cos(theta1))
    
#     # Local Offset Vector relative to Drone CoM
#     local_offset = torch.zeros_like(drone_pos)
#     local_offset[:, 1] = ee_y_local
#     local_offset[:, 2] = ee_z_local
    
#     # Transform to World Frame
#     world_ee_pos = drone_pos + quat_apply(drone_quat, local_offset)

#     # --- ORIENTATION CALCULATION ---
#     # We need the orientation of the End Effector.
#     # Q_ee = Q_drone * Q_shoulder * Q_twist
    
#     # 1. Shoulder Quat (Rotation around -X)
#     # Axis = [-1, 0, 0]
#     half_angle_1 = theta1 / 2.0
#     s1 = torch.sin(half_angle_1)
#     shoulder_quat = torch.zeros_like(drone_quat)
#     shoulder_quat[:, 0] = -s1        # x (-1 * sin)
#     shoulder_quat[:, 3] = torch.cos(half_angle_1) # w
    
#     # 2. Twist Quat (Rotation around Z of the previous link)
#     # Axis = [0, 0, 1]
#     half_angle_2 = theta2 / 2.0
#     s2 = torch.sin(half_angle_2)
#     twist_quat = torch.zeros_like(drone_quat)
#     twist_quat[:, 2] = s2           # z
#     twist_quat[:, 3] = torch.cos(half_angle_2) # w
    
#     # Combine: Q_local = Q_shoulder * Q_twist
#     local_arm_quat = quat_mul(shoulder_quat, twist_quat)
    
#     # Combine with Drone: Q_world = Q_drone * Q_local
#     world_ee_quat = quat_mul(drone_quat, local_arm_quat)

#     return world_ee_pos, world_ee_quat

@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * (torch.exp(-exp * x * x) - 1)

def compute_reward(
                     pos_error,
                     quats,
                     linvels_err,
                     angvels_err,
                     jointvels_err,
                     eevels_err,
                     ee_quat,
                     target_ee_quat,
                     crashes,
                     action_input,
                     prev_action,
                     prev_pos_error,
                     crash_dist):

    target_dist = torch.norm(pos_error[:, :3], dim=1)

    prev_target_dist = torch.norm(prev_pos_error, dim=1)

    #1. ee pos reward
    pos_error[:,2] = pos_error[:,2]*13.
    pos_error[:,0:2] = pos_error[:,0:2]
    pos_reward = torch.sum(exp_func(pos_error[:, :3], 10.0, 10.0), dim=1) + torch.sum(exp_func(pos_error[:, :3], 2.0, 2.0), dim=1)
    # print(f"\nPOS: {pos_error}")
    # print(f"POS R: {pos_reward}")
    #2. drone level reward
    ups = quat_axis(quats, 2)
    tiltage = 1 - ups[..., 2]
    upright_reward = exp_func(tiltage, 2.5, 5.0) + exp_func(tiltage, 2.5, 2.0)
    # print(f"TILT E: {tiltage}")
    # print(f"UPRIGHT R: {upright_reward}")

    #3. drone yaw reward
    forw = quat_axis(quats, 0)
    alignment = 1 - forw[..., 0]
    alignment_reward = exp_func(alignment, 4., 5.0) + exp_func(alignment, 2., 2.0)
    # print(f"ALIGN E: {alignment}")
    # print(f"ALIGN R: {alignment_reward}")

    #4. drone angvel reward
    # angvels_err[:,2]*=2
    angvel_reward = torch.sum(exp_func(angvels_err, 1.5 , 2.0), dim=1)
    angvel_yaw_reward = exp_func(angvels_err[:,2], 1.5 , 2.0) + exp_func(angvels_err[:,2], 1.5 , 10.0)
    # print(f"ANGVEL E: {angvels_err}")
    # print(f"ANGVEL R: {angvel_reward}")
    #5. drone linvel reward
    vel_reward = torch.sum(exp_func(linvels_err, 1., 5.0), dim=1)
    # print(f"LINVEL E: {linvels_err}")
    # print(f"LINVEL R: {linvel_reward}")

    #6. ee vel reward
    eevel_reward = torch.sum(exp_func(eevels_err, 0.5, 5.0) + exp_func(eevels_err, 0.5, 2.0), dim=1)
    # print(f"EEVEL E: {eevels_err}")
    # print(f"EEVEL R: {eevel_reward}")

    #7. joint vel reward
    jointvel_reward = torch.sum(exp_func(jointvels_err, 2.0, 5.0), dim=1) #0#exp_func(jointvels_err, 2., 1.0) + exp_func(jointvels_err, 2., 0.5)
    # print(f"JOINTVEL E: {jointvels_err}")
    # print(f"JOINTVEL R: {jointvel_reward}")

    #8. ee ori reward
    quat_dot = torch.abs(torch.sum(ee_quat * target_ee_quat, dim=1))
    quat_dot = torch.clamp(quat_dot, 0.0, 1.0)
    ee_angle_error = 2.0 * torch.acos(quat_dot) 
    # ee_err = abs(ee_angle_error)
    # ee_err = ee_angle_error * ee_angle_error
    # eeori_reward = exp_func(ee_err, 4., 5.0) + exp_func(ee_err, 2., 2.0)
    eeori_reward = exp_func(ee_angle_error, 4., 1.) + exp_func(ee_angle_error, 4., 5.0)

    # print(f"EEORI E: {ee_angle_error}")
    # print("\n")
    # print(f"EEORI R: {eeori_reward}")

    #9.hover reward
    action_acc = action_input[:,0]
    action_cost = exp_penalty_func(action_acc*1.926/4, 0.01, 10.0)
    # print(f"HOVER acc: {action_acc}")
    # print(f"HOVER R: {action_cost}")

    #10. moving closer reward
    closer_by_dist = prev_target_dist - target_dist
    towards_goal_reward = torch.where(closer_by_dist >= 0, 50*closer_by_dist, 100*closer_by_dist)
    # towards_goal_reward = torch.where(closer_by_dist >= 0, 25*closer_by_dist, 50*closer_by_dist)
    # print(f"DISTANCE E: {closer_by_dist}")
    # print(f"DISTANCE R: {towards_goal_reward}")


    #11. action smoothness reward
    action_diff = action_input - prev_action
    # action_diff_penalty_drone = exp_penalty_func(action_diff[:, 0], 0.5, 6.0)
    action_diff_penalty_acc = torch.sum(exp_penalty_func(action_diff[:, 0:3], 1.0, 5.0), dim=1)
    action_diff_penalty_rad = torch.sum(exp_penalty_func(action_diff[:, 3:], 1.0, 12.0), dim=1)
    action_diff_penalty_yaw = exp_penalty_func(action_diff[:, 6], 4.0, 12.0)

    # action_diff_penalty_joints = torch.sum(exp_penalty_func(action_diff[:, 7:], 1.0, 2.0), dim=1)
    # print(f"ACTION SMOOTH E: {action_difference}")
    # print(f"ACTION SMOOTH R: {action_difference_penalty}")

    #forw is for yaw, will have to change that, add joint vel, ee_vel_err, ee_ori_error

    # 12. action magnitude reward (penalty)
    # Action space:[0:3] lin acc, [3:6] body rates, [6] yaw, [7:10] joint pos
    
    # Penalize large linear accelerations (only x,y)
    action_mag_penalty_acc = torch.sum(exp_penalty_func(action_input[:, 0:2], 2.0, 1.0), dim=1)
    
    # Penalize large body rates (often needs a stricter penalty to prevent aggressive spinning)
    # action_input[:,5] = action_input[:,5]*5
    action_mag_penalty_rates = torch.sum(exp_penalty_func(action_input[:, 3:5], 2.0, 5.0), dim=1)
    action_mag_penalty_yaw_rate = exp_penalty_func(action_input[:, 5], 2.0, 2.0) + exp_penalty_func(action_input[:, 5], 2.0, 10.0)
    
    # Penalize large yaw commands
    # Note: action_input[:, 6] is a 1D tensor, so no need for torch.sum()
    action_mag_penalty_yaw = exp_penalty_func(action_input[:, 6], 1.0, 5.0)

    # Optional: Combine the angular magnitudes for cleaner code in the final equation
    action_mag_penalty = action_mag_penalty_rates + action_mag_penalty_yaw*0 + action_mag_penalty_acc + action_mag_penalty_yaw_rate

    reward = towards_goal_reward + (pos_reward * (eevel_reward + vel_reward*0 + angvel_reward*0 + 0*jointvel_reward + angvel_yaw_reward + action_diff_penalty_acc + action_diff_penalty_rad + action_diff_penalty_yaw + 5*eeori_reward + action_mag_penalty) + (action_mag_penalty + angvel_reward*0 + vel_reward*0 + jointvel_reward*0 + angvel_yaw_reward + eevel_reward + 15*eeori_reward + upright_reward + 50 * pos_reward + action_diff_penalty_rad + action_diff_penalty_yaw + action_diff_penalty_acc)) / 100.0
    
    # reward = towards_goal_reward + (pos_reward * (eevel_reward + vel_reward + angvel_reward + jointvel_reward + action_difference_penalty + 2*eeori_reward + 0*alignment_reward) + (angvel_reward + vel_reward + jointvel_reward + eevel_reward + 12*eeori_reward + alignment_reward*0 + upright_reward + 50 * pos_reward + action_cost)) / 100.0
    # print(f"\nTOTAL REW: {reward}").
    # print(f"POS SCALED REW: {pos_reward * (eevel_reward + vel_reward + angvel_reward + action_difference_penalty + eeori_reward + 0*alignment_reward)/100}")
    # print(f"REST REW: {(angvel_reward + vel_reward + jointvel_reward + eevel_reward + upright_reward + 1.3 * pos_reward + action_cost)/100}")



    crashes[:] = torch.where(target_dist > crash_dist, torch.ones_like(crashes), crashes)

    return reward, crashes

# =========================================================================================================#
# @torch.jit.script
# def compute_reward(
#     pos_error: torch.Tensor,        # Target - EE_Pos
#     ee_quat: torch.Tensor,          # EE Orientation
#     target_ee_quat: torch.Tensor,   # Target EE Orientation
#     actions: torch.Tensor,          # Current actions (Drone: 0-3, Joints: 4-5)
#     prev_actions: torch.Tensor,     # Previous actions
#     crashes: torch.Tensor,
#     crash_dist: float
# ) -> Tuple[torch.Tensor, torch.Tensor]:

#     # --- Hyperparameters from Table I in the Paper ---
#     # Weights (w)
#     w_pos   = 4.0
#     w_ori   = 1.0
#     w_ds    = 0.5   # Drone action smoothing
#     w_js    = 1.0   # Joint action smoothing
#     w_dmag  = 0.1   # Drone action magnitude

#     # Scaling factors (alpha)
#     # Note: The paper uses a negative exponential: w * exp(-alpha * error)
#     a_pos   = 1.2
#     a_ori   = 1.0
#     a_ds    = 1.0
#     a_js    = 1.0
#     a_dmag  = 1.0

#     # --- 1. Position Reward (Eq 6) ---
#     # pos_error is (Target - EE), so norm is the distance
#     dist = torch.norm(pos_error, dim=1)
#     r_pos = w_pos * torch.exp(-a_pos * dist)

#     # --- 2. Orientation Reward (Eq 7) ---
#     # Geodesic distance: 2 * acos(|q . q_goal|)
#     # Dot product of quaternions
#     quat_dot = torch.sum(ee_quat * target_ee_quat, dim=1)
#     # Clamp to handle numerical instability ensuring inputs to acos are in [-1, 1]
#     quat_dot = torch.clamp(torch.abs(quat_dot), 0.0, 1.0)
    
#     # Orientation error (Geodesic angle)
#     e_ori = 2.0 * torch.acos(quat_dot)
#     r_ori = w_ori * torch.exp(-a_ori * e_ori)

#     # --- 3. Action Smoothing Rewards (Eq 8 & 9) ---
#     # Assuming indices: 0-3 are Drone (Thrust/Rates), 4-5 are Joints
    
#     # Drone Smoothing (L2 norm squared of difference)
#     drone_actions = actions[:, :4]
#     drone_prev = prev_actions[:, :4]
#     diff_drone = drone_actions - drone_prev
#     # Sum of squares along dim 1
#     norm_diff_drone_sq = torch.sum(diff_drone ** 2, dim=1) 
#     r_ds = w_ds * torch.exp(-a_ds * norm_diff_drone_sq)

#     # Joint Smoothing (L1 norm of difference - as per Eq 9 text)
#     joint_actions = actions[:, 4:]
#     joint_prev = prev_actions[:, 4:]
#     diff_joint = torch.abs(joint_actions - joint_prev)
#     norm_diff_joint = torch.sum(diff_joint, dim=1)
#     r_js = w_js * torch.exp(-a_js * norm_diff_joint)

#     # --- 4. Action Magnitude Reward (Eq 10) ---
#     # Penalizes large drone commands (energy/oscillation)
#     # L2 norm squared of current drone actions
#     norm_drone_sq = torch.sum(drone_actions ** 2, dim=1)
#     r_dmag = w_dmag * torch.exp(-a_dmag * norm_drone_sq)

#     # --- Total Reward (Eq 11) ---
#     total_reward = r_pos + r_ori + r_ds + r_js + r_dmag

#     # --- Crash Logic ---
#     # If the distance is too large, we consider it a crash (divergence)
#     # Or if Z height is too low (floor collision check should happen outside or passed in 'crashes')
    
#     # Update crashes based on distance divergence
#     crashes = torch.where(dist > crash_dist, torch.ones_like(crashes), crashes)
    
#     # Apply heavy penalty for crashing (standard RL practice not explicitly detailed in Eq 11 
#     # but implied by episode resets in Section IV.B.3)
#     # We use -10.0 or reset reward to 0.0 depending on preference. 
#     # A negative value helps the critic learn "death is bad".
#     total_reward = torch.where(crashes > 0.0, -5.0 * torch.ones_like(total_reward), total_reward)

#     return total_reward, crashes

# =========================================================================================================#

# @torch.jit.script
# def compute_reward(
#     ee_pos: torch.Tensor,
#     ee_quat: torch.Tensor,
#     target_pos: torch.Tensor,
#     target_ee_quat: torch.Tensor,
#     drone_quats: torch.Tensor,
#     drone_linvel: torch.Tensor,
#     drone_angvel: torch.Tensor,
#     joint_vel: torch.Tensor,
#     joint_angles: torch.Tensor, # Added this input
#     action_curr: torch.Tensor,
#     action_prev: torch.Tensor,
#     actions_raw: torch.Tensor,
#     actions_prev_raw: torch.Tensor,
#     crashes: torch.Tensor,
#     crash_dist: float
# ) -> Tuple[torch.Tensor, torch.Tensor]:

#     # --- 0. PRE-CALCULATIONS ---
    
#     # 1. Calculate Exact EE Velocity (World Frame)
#     # We pass a dummy tensor for the last arg or recalculate it inside as I did above.
#     ee_world_vel = calculate_ee_velocity(
#         drone_linvel, 
#         drone_angvel, 
#         drone_quats, 
#         joint_angles, 
#         joint_vel, 
#         torch.zeros_like(ee_pos) # Dummy, we recalc inside function
#     )
    
#     # Distance
#     dist = torch.norm(ee_pos - target_pos, dim=1)
#     dist_sq = dist * dist
#     # print("ERR: " + str(dist))
    
#     gate_80 = 1.0 / (1.0 + 80.0 * dist_sq)
#     gate_50 = 1.0 / (1.0 + 50.0 * dist_sq)
#     gate_05 = 1.0 / (1.0 + 0.5 * dist_sq)
#     gate_yaw = 1.0 / (1.0 + 0.8 * (dist - 2.5)*(dist - 2.5))
#     # Norms
#     ee_vel_norm = torch.norm(ee_world_vel, dim=1) # <--- USING ANALYTIC EE VELOCITY
#     ee_vel_sq = ee_vel_norm * ee_vel_norm
    
#     quat_dot = torch.abs(torch.sum(ee_quat * target_ee_quat, dim=1))
#     quat_dot = torch.clamp(quat_dot, 0.0, 1.0)
#     ee_angle_error = 2.0 * torch.acos(quat_dot) 
#     ee_angle_sq = ee_angle_error * ee_angle_error

#     drone_vel_norm = torch.norm(drone_linvel, dim=1)
#     drone_angvel_rp = torch.norm(drone_angvel[:, 0:2], dim=1) # Roll and Pitch rates
#     drone_angvel_yaw = torch.abs(drone_angvel[:, 2])          # Yaw rate
    
#     # --- REWARD EQUATIONS ---

#     # 1. EE Position Reward
#     # 1/(1 + 4.6(ee_dist)^2) + 0.2/(1 + 200(ee_dist)^2)
#     r_ee_pos = (1.0 / (1.0 + 2.0 * dist_sq)) + (0.6 / (1.0 + 100.0 * dist_sq))

#     # 2. EE Orientation Reward
#     # (1/(2 + 4(err)^2) + 1/(1 + 600(err)^2)) * gate_80
#     r_term_ori = (1.0 / (2.0 + 4.0 * ee_angle_sq)) + (1.0 / (1.0 + 600.0 * ee_angle_sq))
#     r_ee_ori = r_term_ori * gate_80

#     # 3. EE Velocity Reward (Updated with analytic vel)
#     # (1/(2 + 4(v)^2) + 1/(1 + 300(v)^2)) * gate_1500
#     r_term_vel = (1.0 / (2.0 + 4.0 * ee_vel_sq)) + (1.0 / (1.0 + 300.0 * ee_vel_sq))
#     r_ee_vel = r_term_vel * gate_05

#     # 4. Drone Orientation Reward (Uprightness)
#     # e^(-0.7(1-drone_quat_z)) - 0.5
#     # Calculate Z-axis of the drone body in world frame
#     # (Strictly speaking, for standard quat, axis 2 (Z) is calculated as below)
#     # But often in Isaac Gym, simply taking quat[..., 2] (the z component) 
#     # roughly correlates to tilt if yaw is aligned. 
#     # Let's use the proper Z-projection for robustness:
#     drone_up_z = 2.0 * (drone_quats[:, 1] * drone_quats[:, 3] - drone_quats[:, 0] * drone_quats[:, 2]) # This is Z component of X-axis... wait.
#     # Simpler method used in Aerial Gym utils:
#     drone_up_z = quat_axis(drone_quats, 2)[:, 2] # Project Z-axis of body onto Z-axis of world
    
#     r_drone_ori = torch.exp(-0.7 * (1.0 - drone_up_z))- 0.5

#     # 5. Drone Linear Velocity Reward
#     # (e^(-1.2(drone_vel_norm))) * gate_80
#     r_drone_linvel = torch.exp(-1.2 * (drone_vel_norm-2.0)) * gate_50 - 5.0 * drone_vel_norm + 10

#     # 6. Drone AngVel (Roll/Pitch) Reward
#     # 1.6*e^(-0.3(drone_r,p_norm)) - 1.0
#     r_angvel_rp = 1.6 * torch.exp(-0.3 * drone_angvel_rp) - 1.0

#     # 7. Drone AngVel (Yaw) Reward
#     # Requirement: Reward 0 yaw rate when >1m away. Within 1m, less/no penalty.
#     # Logic: We create a 'strictness' coefficient that relaxes when dist < 1.0
#     # Sigmoid centers at 1.0. 
#     # If dist > 1.0, strictness -> High. If dist < 1.0, strictness -> Low.
#     # yaw_strictness = 10.0 * torch.sigmoid(5.0 * (dist - 1.0)) # Scales from ~0 to 10
#     # r_angvel_yaw = torch.exp(-yaw_strictness * (drone_angvel_yaw**2))
#     r_angvel_yaw = (-0.5 * abs(drone_angvel_yaw) + 0.1)*gate_yaw

#     # 8. Drone Actions Reward (Minimize control effort)
#     # "rewarded if near 0" -> exp(-x^2)
#     # Applied only to drone accel (0), roll rate (1), pitch rate (2)
#     action_mag_sq = torch.sum(actions_raw[:, 0:3]**2, dim=1)
#     r_actions = torch.exp(-2.0 * action_mag_sq)

#     # 9. Action Smoothness
#     action_diff_sq = torch.sum((actions_raw - actions_prev_raw)**2, dim=1)
#     r_smoothness = torch.exp(-1.2 * action_diff_sq)

#     # 10. Joint Velocity Reward
#     joint_vel_sq = torch.sum(joint_vel**2, dim=1)
#     r_joint_vel = torch.exp(-1.5 * joint_vel_sq)

#     # --- TOTAL SUM ---
#     total_reward = (
#         r_ee_pos*1.5 + 
#         # r_ee_ori + 
#         r_ee_vel*0.8 + 
#         r_drone_ori + 
#         r_drone_linvel*0.03 + 
#         r_angvel_rp*0.8 + 
#         # r_angvel_yaw*3.0 + 
#         r_actions*0.2 + 
#         r_smoothness*0.3 + 
#         r_joint_vel*0.5
#     )

#     # --- CRASH PENALTY ---
#     # Overwrite reward if crashed
#     crashes[:] = torch.where(ee_pos[:, 2] < 0.1, torch.ones_like(crashes), crashes)
#     crashes[:] = torch.where(dist > crash_dist, torch.ones_like(crashes), crashes)
    
#     total_reward = torch.where(crashes > 0.0, -5.0 * torch.ones_like(total_reward), total_reward)

#     return total_reward, crashes

# @torch.jit.script
# def quat_axis(q, axis=2):
#     # type: (Tensor, int) -> Tensor
#     basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
#     basis_vec[:, axis] = 1
#     return quat_rotate(q, basis_vec)

# # Standard Aerial Gym Quat Rotate
# @torch.jit.script
# def quat_rotate(q, v):
#     shape = q.shape
#     q_w = q[:, -1]
#     q_vec = q[:, :3]
#     a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
#     b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
#     c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
#     return a + b + c

#=========================================================================================================#