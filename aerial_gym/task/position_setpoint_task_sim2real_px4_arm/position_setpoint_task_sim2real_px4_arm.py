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

        # Buffers
        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.joint_actions = torch.zeros(
            (self.sim_env.num_envs, 2),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.raw_actions = torch.zeros_like(self.actions)
        self.prev_raw_actions = torch.zeros_like(self.actions)
        self.prev_dist = torch.zeros((self.sim_env.num_envs), device=self.device)
        self.counter = 0

        self.target_ee_position = torch.zeros((self.sim_env.num_envs, 3), device=self.device, requires_grad=False)

        self.target_ee_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(self.sim_env.num_envs, 4)
        
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
        self.sim_env.reset()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.target_ee_position[:, 0:2] = 0.0 
        self.target_ee_position[:, 2] = 0.0
       # Randomize EE target position: X, Y in [-2, 2], Z in [1, 2.5]
        # self.target_ee_position[env_ids, 0:2] = (torch.rand((len(env_ids), 2), device=self.device) - 0.5) * 4.0
        # self.target_ee_position[env_ids, 2] = torch.rand((len(env_ids)), device=self.device) * 1.5 + 1.0

        # rand_quat = torch.randn((len(env_ids), 4), device=self.device)
        # self.target_ee_quat[env_ids] = torch.nn.functional.normalize(rand_quat, p=2, dim=-1)
        self.sim_env.reset_idx(env_ids)
        self.prev_pos_error[env_ids] = 0.0

    def render(self, mode="human"):
        return None

    # def step(self, actions):
    #     self.counter += 1
        
    #     # 1. Scale Actions
    #     self.actions = self.task_config.process_actions_for_task(
    #         actions, self.task_config.action_limit_min, self.task_config.action_limit_max
    #     )
        
    #     # 2. SPLIT ACTIONS
    #     # Drone (Accel, Rates) = Indices 0-3
    #     # Arm (Positions)      = Indices 4-5
    #     drone_actions = self.actions[:, :4]
    #     joint_actions = self.actions[:, 4:]

    #     # 3. Apply Arm Targets (Before Physics Step!)
    #     self.sim_env.robot_manager.robot.set_dof_position_targets(joint_actions)

    #     # 4. Step Physics
    #     # This sends drone_actions to LeeRatesController -> Physics Engine
    #     self.sim_env.step(actions=drone_actions)
        
    #     # 5. Rewards
    #     self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)
        
    #     self.truncations[:] = torch.where(self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0)
    #     reset_envs = self.sim_env.post_reward_calculation_step()
    #     if len(reset_envs) > 0: self.reset_idx(reset_envs)

    #     self.prev_actions = self.actions.clone()
    #     ee_pos, _ = calculate_ee_state(
    #         self.obs_dict["robot_position"],
    #         self.obs_dict["robot_orientation"],
    #         self.obs_dict["dof_state_tensor"][..., 0]
    #     )
    #     self.prev_pos_error = self.target_ee_position - ee_pos
    #     # === ADD THIS DEBUG BLOCK ===
    #     if torch.isnan(self.task_obs["observations"]).any():
    #         print("!!! NAN DETECTED IN OBSERVATIONS !!!")
    #         # Find which column has the NaN
    #         nan_mask = torch.isnan(self.task_obs["observations"])
    #         indices = nan_mask.nonzero(as_tuple=True)
    #         print(f"NaN found in columns: {torch.unique(indices[1])}")
            
    #         # Print the raw robot state from sim to see if physics exploded
    #         print(f"Robot Position: {self.obs_dict['robot_position'][0]}")
    #         print(f"Robot LinVel: {self.obs_dict['robot_linvel'][0]}")
    #         print(f"DOF Positions: {self.obs_dict['dof_state_tensor'][0, :, 0]}")
            
    #         # Stop the program so you can read the error
    #         import sys
    #         sys.exit()
    #     # ============================

    #     return self.get_return_tuple()

    def step(self, actions):
        import sys
        self.counter += 1

        self.raw_actions = torch.clamp(actions, -1, 1)
        
        # 1. Check RL Agent Output
        if torch.isnan(actions).any():
            print(f"[DEBUG] Step {self.counter}: RL Agent produced NaNs in actions")
            sys.exit()

        # 2. Scale Actions
        self.actions = self.task_config.process_actions_for_task(
            actions, self.task_config.action_limit_min, self.task_config.action_limit_max
        )
        
        if torch.isnan(self.actions).any():
            print(f"[DEBUG] Step {self.counter}: NaNs found AFTER scaling actions")
            sys.exit()

        # 3. Split and Apply
        drone_actions = self.actions[:, :4]
        # joint_actions = self.joint_actions
        joint_actions = self.actions[:, 4:]
        # drone_actions[:, 0] = -9.81
        # drone_actions[:, 1] = 0
        # drone_actions[:, 2] = 0
        # drone_actions[:, 3] = 0
        # joint_actions[:, 0] = 0.0
        # joint_actions[:, 1] = 0.0

        # print(f"Actions: {self.actions[0, :]}")

        # === DEBUG: FORCE ALL MOTOR AND ARM COMMANDS TO ZERO ===
        # drone_actions = torch.zeros((self.sim_env.num_envs, 4), device=self.device)
        # joint_actions = torch.zeros((self.sim_env.num_envs, 2), device=self.device)
        # =======================================================


        self.sim_env.robot_manager.robot.set_dof_position_targets(joint_actions)

        # 4. STEP PHYSICS
        self.sim_env.step(actions=drone_actions)

        # print(f"\nCOMMANDED BODY RATES: {drone_actions[0, 1:]}")
        # print(f"""ACTUAL BODY RATES: {self.obs_dict["robot_body_angvel"][0,:]}""")
        # print("\n")
    
        # VISUALIZATION CALL
        if not self.task_config.headless:
            self.draw_debug_targets()

        
        # 5. POST-PHYSICS CHECK (Did the engine explode?)
        if torch.isnan(self.obs_dict["robot_position"]).any():
            print(f"[DEBUG] Step {self.counter}: PHYSICS EXPLODED!")
            print(f"Sample drone_actions: {drone_actions[0]}")
            print(f"Sample joint_targets: {joint_actions[0]}")
            # Check for zero/nan mass in controller
            mass = self.sim_env.robot_manager.robot_masses[0]
            print(f"Controller Mass: {mass}")
            sys.exit()

        # 6. FK CHECK
        ee_pos, ee_quat = calculate_ee_state(
            self.obs_dict["robot_position"],
            self.obs_dict["robot_orientation"],
            self.obs_dict["dof_state_tensor"][..., 0]
        )
        
        if torch.isnan(ee_pos).any() or torch.isnan(ee_quat).any():
            print(f"[DEBUG] Step {self.counter}: FK produced NaNs")
            print(f"Joint Angles: {self.obs_dict['dof_state_tensor'][0, :, 0]}")
            sys.exit()

        # 7. REWARD CHECK
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)
        if torch.isnan(self.rewards).any():
            print(f"[DEBUG] Step {self.counter}: Reward function produced NaNs")
            sys.exit()

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
        self.prev_pos_error = self.target_ee_position - ee_pos#self.obs_dict['robot_position']

        return self.get_return_tuple()

    def draw_debug_targets(self):

        viewer = self.sim_env.IGE_env.viewer.viewer
        gym = self.sim_env.IGE_env.gym
        if viewer is None:
            return
        
        # We clear lines every frame so they don't "stack" and slow down the sim
        gym.clear_lines(viewer)

        # Define a small crosshair size (e.g., 20cm)
        s = 0.2 

        ee_pos, _ = calculate_ee_state(
            self.obs_dict["robot_position"],
            self.obs_dict["robot_orientation"],
            self.obs_dict["dof_state_tensor"][..., 0]
        )
        
        # We will draw a cross for the target position of the first few environments
        # Drawing for all 4096 envs will crash your frame rate, so we pick the first 1-4.
        for i in range(min(4, self.num_envs)):
            target = self.target_ee_position[i].cpu().numpy()
            ee_target = ee_pos[i].cpu().numpy()
            env_handle = self.sim_env.IGE_env.env_handles[i]

            # Lines: {start_x, start_y, start_z, end_x, end_y, end_z}
            # 3 Lines (X, Y, Z axes)
            verts = np.array([
                [target[0]-s, target[1], target[2], target[0]+s, target[1], target[2]], # X-line
                [target[0], target[1]-s, target[2], target[0], target[1]+s, target[2]], # Y-line
                [target[0], target[1], target[2]-s, target[0], target[1], target[2]+s]  # Z-line
            ], dtype=np.float32)

            verts_fk = np.array([
                [ee_target[0]-s, ee_target[1], ee_target[2], ee_target[0]+s, ee_target[1], ee_target[2]], # X-line
                [ee_target[0], ee_target[1]-s, ee_target[2], ee_target[0], ee_target[1]+s, ee_target[2]], # Y-line
                [ee_target[0], ee_target[1], ee_target[2]-s, ee_target[0], ee_target[1], ee_target[2]+s]  # Z-line
            ], dtype=np.float32)

            # Colors: {R, G, B}
            # Let's make the target bright Red
            colors = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32)
            colors_fk = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32)

            gym.add_lines(viewer, env_handle, 3, verts, colors)
            gym.add_lines(viewer, env_handle, 3, verts_fk, colors_fk)


    def get_return_tuple(self):
        self.process_obs_for_task()
        return (self.task_obs, self.rewards, self.terminations, self.truncations, {})

    def process_obs_for_task(self):
        # 1. Drone Ground Truths (with noise if you want)
        obs_drone_pos = self.target_ee_position - self.obs_dict["robot_position"] # Drone rel to EE target
        
        # Drone 6D Orientation
        quat_raw = self.obs_dict["robot_orientation"]
        quat_norm = torch.nn.functional.normalize(quat_raw, p=2, dim=-1, eps=1e-8)
        or_quat = quat_norm[:, [3, 0, 1, 2]] # xyzw -> wxyz
        obs_drone_rot6d = matrix_to_rotation_6d(quaternion_to_matrix(or_quat))

        # 2. End Effector States
        ee_pos, ee_quat = calculate_ee_state(
            self.obs_dict["robot_position"],
            self.obs_dict["robot_orientation"],
            self.obs_dict["dof_state_tensor"][..., 0]
        )
        
        obs_ee_pos_err = self.target_ee_position - ee_pos
        ee_quat_p3d = ee_quat[:, [3, 0, 1, 2]]
        obs_ee_rot6d = matrix_to_rotation_6d(quaternion_to_matrix(ee_quat_p3d))

        target_quat_p3d = self.target_ee_quat[:, [3, 0, 1, 2]]
        obs_target_rot6d = matrix_to_rotation_6d(quaternion_to_matrix(target_quat_p3d))



        # 3. Fill the Tensor (Indices must match observation_space_dim)
        self.task_obs["observations"][:, 0:3]   = obs_drone_pos/10
        self.task_obs["observations"][:, 3:9]   = obs_drone_rot6d
        self.task_obs["observations"][:, 9:12]  = self.obs_dict["robot_linvel"]/10
        self.task_obs["observations"][:, 12:15] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 15:18] = obs_ee_pos_err/10
        self.task_obs["observations"][:, 18:24] = obs_ee_rot6d
        self.task_obs["observations"][:, 24:30] = obs_target_rot6d
        self.task_obs["observations"][:, 30:32] = self.obs_dict["dof_state_tensor"][..., 0] # Joint Pos
        self.task_obs["observations"][:, 32:34] = self.obs_dict["dof_state_tensor"][..., 1] # Joint Vel
        self.task_obs["observations"][:, 34:40] = self.prev_raw_actions

        # print("OBSERVATIONS SAMPLE: ", self.task_obs["observations"][0,0:3])

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
            torch.zeros_like(ee_pos) # Dummy, we recalc inside function
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
# ========================================================
# REWARD FUNCTIONS - FIXED JIT SYNTAX
# ========================================================



@torch.jit.script
def calculate_ee_state(drone_pos: torch.Tensor, drone_quat: torch.Tensor, joint_angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates End Effector position based on X500 Arm URDF.
    Joint Axis: (-1, 0, 0). 
    Home Config: Arm hanging down.
    """
    theta1 = joint_angles[:, 0]
    theta2 = joint_angles[:, 1]
    
    # URDF offsets
    l1 = 0.18
    l2 = 0.20
    mount_z = -0.12
    
    # In Local Frame (X-forward, Z-up):
    # Axis is -X. Positive Theta rotates Y towards -Z.
    # Standard pendulum math with phase shift.
    
    # Joint 1 Position relative to Mount
    # y1 = -l1 * sin(theta1)
    # z1 = -l1 * cos(theta1)
    
    # Joint 2 Position relative to Joint 1
    # y2 = -l2 * sin(theta1 + theta2)
    # z2 = -l2 * cos(theta1 + theta2)
    
    ee_y_local = -l1 * torch.sin(theta1) - l2 * torch.sin(theta1 + theta2)
    ee_z_local = mount_z - l1 * torch.cos(theta1) - l2 * torch.cos(theta1 + theta2)
    
    # Construct Local Offset Vector
    local_offset = torch.zeros_like(drone_pos)
    local_offset[:, 1] = ee_y_local
    local_offset[:, 2] = ee_z_local
    
    # Apply Drone Orientation
    world_ee_pos = drone_pos + quat_apply(drone_quat, local_offset)
    
    # Orientation is not critical for position task, returning drone quat for now 
    # (calculating composed quaternion for 2-link arm is expensive and often unnecessary for simple Pos task)
    return world_ee_pos, drone_quat

@torch.jit.script
def calculate_ee_velocity(
    drone_linvel: torch.Tensor,      # World Frame
    drone_angvel: torch.Tensor,      # Body Frame
    drone_quat: torch.Tensor,        # [x,y,z,w]
    joint_angles: torch.Tensor, 
    joint_velocities: torch.Tensor,
    ee_pos_rel_drone: torch.Tensor   # The local offset calculated in FK
) -> torch.Tensor:
    
    # --- 1. Velocity due to Joint Motion (Relative to Drone) ---
    theta1 = joint_angles[:, 0]
    theta2 = joint_angles[:, 1]
    d_theta1 = joint_velocities[:, 0]
    d_theta2 = joint_velocities[:, 1]
    
    # Lengths (Must match URDF)
    l1 = 0.18
    l2 = 0.20
    
    # Derivatives of position w.r.t time (Chain rule)
    # y = -l1*sin(t1) - l2*sin(t1+t2)
    # z = mount - l1*cos(t1) - l2*cos(t1+t2)
    
    # dy/dt = -l1*cos(t1)*dt1 - l2*cos(t1+t2)*(dt1+dt2)
    v_y_local = -l1 * torch.cos(theta1) * d_theta1 - l2 * torch.cos(theta1 + theta2) * (d_theta1 + d_theta2)
    
    # dz/dt = l1*sin(t1)*dt1 + l2*sin(t1+t2)*(dt1+dt2)
    v_z_local = l1 * torch.sin(theta1) * d_theta1 + l2 * torch.sin(theta1 + theta2) * (d_theta1 + d_theta2)
    
    # Construct Local Velocity Vector
    v_arm_local = torch.zeros_like(drone_linvel)
    v_arm_local[:, 1] = v_y_local
    v_arm_local[:, 2] = v_z_local
    
    # Rotate joint velocity to World Frame
    v_arm_world = quat_apply(drone_quat, v_arm_local)

    # --- 2. Velocity due to Drone Rotation (Tangential Velocity) ---
    # V_tangential = omega (cross) r
    # We need omega in World Frame to cross with r (World Frame)
    # Or omega (Body) cross r (Body) then rotate. Let's do Body frame.
    
    # Cross product in Body Frame: omega_body x r_body
    # Note: ee_pos_rel_drone passed in is usually World Frame. We need Local.
    # Let's reconstruct local position quickly to be safe or pass it from FK.
    # Re-calculating local P for strict correctness:
    p_y_local = -l1 * torch.sin(theta1) - l2 * torch.sin(theta1 + theta2)
    p_z_local = -0.12 - l1 * torch.cos(theta1) - l2 * torch.cos(theta1 + theta2)
    
    p_local = torch.zeros_like(drone_linvel)
    p_local[:, 1] = p_y_local
    p_local[:, 2] = p_z_local

    # Tangential Velocity in Body Frame
    v_tan_body = torch.cross(drone_angvel, p_local, dim=1)
    
    # Rotate Tangential to World
    v_tan_world = quat_apply(drone_quat, v_tan_body)
    
    # --- 3. Total EE Velocity in World Frame ---
    # V_ee = V_drone_linear + V_arm_extension_rotated + V_tangential_rotation
    ee_vel_world = drone_linvel + v_arm_world + v_tan_world
    
    return ee_vel_world


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
    angvels_err[:,2]*=5
    angvel_reward = torch.sum(exp_func(angvels_err, .75 , 10.0), dim=1)
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

    #8. drone ori reward
    quat_dot = torch.abs(torch.sum(ee_quat * target_ee_quat, dim=1))
    quat_dot = torch.clamp(quat_dot, 0.0, 1.0)
    ee_angle_error = 2.0 * torch.acos(quat_dot) 
    # ee_err = abs(ee_angle_error)
    ee_err = ee_angle_error * ee_angle_error
    # eeori_reward = exp_func(ee_err, 4., 5.0) + exp_func(ee_err, 2., 2.0)
    eeori_reward = exp_func(ee_err, 4., 1.0) + exp_func(ee_err, 4., 5.0)

    # print(f"EEORI E: {ee_angle_error}")
    # print(f"EEORI R: {eeori_reward}")

    #5.hover reward
    action_acc = action_input[:,0]
    action_cost = exp_penalty_func(action_acc*1.9651/4, 0.01, 10.0)
    # print(f"HOVER acc: {action_acc}")
    # print(f"HOVER R: {action_cost}")

    #6. moving closer reward
    closer_by_dist = prev_target_dist - target_dist
    towards_goal_reward = torch.where(closer_by_dist >= 0, 50*closer_by_dist, 100*closer_by_dist)
    # towards_goal_reward = torch.where(closer_by_dist >= 0, 25*closer_by_dist, 50*closer_by_dist)
    # print(f"DISTANCE E: {closer_by_dist}")
    # print(f"DISTANCE R: {towards_goal_reward}")


    #7. action smoothness reward
    action_difference = action_input - prev_action
    action_difference_penalty = torch.sum(exp_penalty_func(action_difference, 0.5, 6.0), dim=1)
    # print(f"ACTION SMOOTH E: {action_difference}")
    # print(f"ACTION SMOOTH R: {action_difference_penalty}")

    #forw is for yaw, will have to change that, add joint vel, ee_vel_err, ee_ori_error

    # reward = towards_goal_reward + (pos_reward * (alignment_reward + vel_reward + angvel_reward + action_difference_penalty) + (angvel_reward + vel_reward + upright_reward + pos_reward + action_cost)) / 100.0
    
    reward = towards_goal_reward + (pos_reward * (eevel_reward + vel_reward + angvel_reward + jointvel_reward + action_difference_penalty + eeori_reward + 0.0*alignment_reward) + (angvel_reward + vel_reward + jointvel_reward + eevel_reward + 25 * eeori_reward + upright_reward + 100 * pos_reward + action_cost)) / 100.0
    # print(f"\nTOTAL REW: {reward}")
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
#     action_curr: torch.Tensor,
#     action_prev: torch.Tensor,
#     crashes: torch.Tensor,
#     crash_dist: float
# ) -> Tuple[torch.Tensor, torch.Tensor]:
    
#     # 1. Distance Reward (Positional)
#     dist = torch.norm(ee_pos - target_pos, dim=1)
#     # Reward: Higher when closer. 
#     # Use additive structure: 1.0 / (1.0 + dist) is very stable.
#     r_pos = 1.0 / (1.0 + dist**2)

#     # 1.2. ORIENTATION REWARD
#     quat_diff = torch.sum(ee_quat * target_ee_quat, dim=1)
#     ori_error = 1.0 - quat_diff**2
#     r_ori = 1.0 * torch.exp(-10.0 * ori_error) # Bonus for aligning correctly

#     # 2. Stability / Upright Reward
#     # We want the drone to stay relatively flat, but it needs to tilt to move.
#     # Allow 0.2 rad tilt without penalty, then penalize.
#     ups = quat_axis(drone_quats, 2)
#     tilt_error = 1.0 - ups[:, 2] # 0 is perfectly upright
#     r_stability = torch.exp(-5.0 * tilt_error)

#     # 3. Penalties
#     # Velocity Penalty (prevent high speed oscillation)
#     # if dist<=0.0
#     p_vel = -0.01 * torch.norm(drone_linvel, dim=1)
#     p_angvel = -0.01 * torch.norm(drone_angvel, dim=1)
#     p_arm_vel = -0.005 * torch.norm(joint_vel, dim=1) # Discourage arm flailing
    
#     # Action Smoothness
#     p_action_diff = -0.1 * torch.norm(action_curr - action_prev, dim=1)

#     # 4. Crash Logic
#     # Hit floor (Z < 0.1) or fly too far
#     crashes[:] = torch.where(ee_pos[:, 2] < 0.1, torch.ones_like(crashes), crashes)
#     crashes[:] = torch.where(dist > crash_dist, torch.ones_like(crashes), crashes)
    
#     # 5. Total Reward (ADDITIVE IS KEY)
#     # Weight the position reward highest (e.g., 5.0)
#     total_reward = (5.0 * r_pos) + (1.0 * r_stability) + p_vel + p_angvel + p_arm_vel + p_action_diff + r_ori
    
#     # Apply Crash Penalty
#     total_reward = torch.where(crashes > 0.0, -20.0 * torch.ones_like(total_reward), total_reward)

#     return total_reward, crashes

# @torch.jit.script
# def calculate_ee_state(drone_pos: torch.Tensor, drone_quat: torch.Tensor, joint_angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     # Joint 1 and 2 rotate around X-axis (from your URDF)
#     # This means movement happens in the Y-Z plane of the drone
#     theta1 = joint_angles[:, 0]
#     theta2 = joint_angles[:, 1]
    
#     # Link lengths from URDF
#     l1 = 0.18
#     l2 = 0.20
#     mount_z = -0.12 # 12cm below COM
    
#     # Calculate EE position in the LOCAL DRONE FRAME
#     # Assuming 0 radians is pointing straight down
#     ee_y_local = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)
#     ee_z_local = mount_z - (l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2))
    
#     # Create the local vector [x=0, y, z]
#     local_offset = torch.zeros_like(drone_pos)
#     local_offset[:, 1] = ee_y_local
#     local_offset[:, 2] = ee_z_local

#     arm_quat_local = torch.zeros_like(drone_quat)
#     arm_quat_local[:, 0] = torch.sin((theta1 + theta2) / 2.0) # Rotation around X
#     arm_quat_local[:, 3] = torch.cos((theta1 + theta2) / 2.0)
    
#     # Rotate the local offset by the drone's current orientation
#     # and add to drone's world position
#     world_ee_pos = drone_pos + quat_apply(drone_quat, local_offset)
#     world_ee_quat = quat_mul(drone_quat, arm_quat_local)
#     return world_ee_pos, world_ee_quat



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
#     joint_angles: torch.Tensor,
#     action_curr: torch.Tensor,
#     action_prev: torch.Tensor,
#     crashes: torch.Tensor,
#     crash_dist: float
# ) -> Tuple[torch.Tensor, torch.Tensor]:
    
#     # 1.1. EE POSITION REWARD (Linear + Exponential)
#     dist = torch.norm(ee_pos - target_pos, dim=1)
    
#     # Linear "Gravity" to the goal (Max dist 10m)
#     r_dist_linear = (10.0 - torch.clamp(dist, max=10.0)) * 0.1
#     # Exponential "Precision" reward
#     r_dist_exp = 2.0 * torch.exp(-5.0 * dist**2) + 5.0 * torch.exp(-20.0 * dist**2)
#     r_dist = r_dist_linear + r_dist_exp

#     # 1.2. ORIENTATION REWARD
#     quat_diff = torch.sum(ee_quat * target_ee_quat, dim=1)
#     ori_error = 1.0 - quat_diff**2
#     r_ori = 3.0 * torch.exp(-10.0 * ori_error) # Bonus for aligning correctly
    
#     # 2. DRONE STABILITY (Orientation)
#     ups = quat_axis(drone_quats, 2)
#     tiltage = 1.0 - ups[:, 2] # 0 when level, 1 when vertical
#     r_stability = torch.exp(-3.0 * tiltage**2)
    
#     # 3. VELOCITY PENALTIES (Dampening)
#     p_drone_vel = torch.norm(drone_linvel, dim=1) * -0.1
#     p_drone_angvel = torch.norm(drone_angvel, dim=1) * -0.1
#     p_joint_vel = torch.norm(joint_vel, dim=1) * -0.05
    
#     # 4. ACTION PENALTIES
#     # Penalty for using too much thrust/effort
#     p_action_mag = torch.norm(action_curr, dim=1) * -0.01
#     # Penalty for jerky movements (Action Smoothness)
#     p_action_smooth = torch.norm(action_curr - action_prev, dim=1) * -0.1
    
#     # 5. JOINT LIMIT PENALTY
#     # Penalize if arm is within 0.1 rad of 1.57 limit
#     # limit_dist = 1.57 - torch.abs(joint_angles)
#     # p_joint_limit = torch.sum(torch.where(limit_dist < 0.1, -1.0, 0.0), dim=1)
    
#     # r_floor_penalty = torch.where(ee_pos[:, 2] < 0.5, -20.0, 0.0)

#     # TOTAL SUM
#     # Task reward (EE Pos) is multiplied by stability so it learns to stabilize TO reach the goal
#     total_reward = (r_dist*r_ori) * r_stability
#     total_reward += p_drone_vel + p_drone_angvel + p_joint_vel + p_action_smooth + p_action_mag#+ p_joint_limit + r_floor_penalty

#     # CRASH LOGIC
#     # If drone flies away or hits the floor (Z < 0.1)
#     crashes[:] = torch.where(dist > crash_dist, torch.ones_like(crashes), crashes)
#     crashes[:] = torch.where(ee_pos[:, 2] < 0.0, torch.ones_like(crashes), crashes)
    
#     # Massive crash penalty
#     total_reward = torch.where(crashes > 0.0, -50.0 * torch.ones_like(total_reward), total_reward)
    
#     return total_reward, crashes