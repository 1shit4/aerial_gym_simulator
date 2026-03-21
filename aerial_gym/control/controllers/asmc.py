import torch
import math
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *
from aerial_gym.control.controllers.base_lee_controller import BaseLeeController, extract_vee_map, asmc_sigmoid

class ASMCAttitudeController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)
        
    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)
        
        # 1. Load PID and ASMC parameters from config to device tensors
        self.dt = self.cfg.asmc_dt
        # self.max_int_att = self.cfg.asmc_max_int_att
        # self.max_rate = self.cfg.asmc_max_rate
        self.max_torque = self.cfg.asmc_max_torque
        
        # PID gains
        self.kp_att = torch.tensor(self.cfg.asmc_kp_att, device=self.device).expand(self.num_envs, -1)
        self.ki_att = torch.tensor(self.cfg.asmc_ki_att, device=self.device).expand(self.num_envs, -1)
        self.kd_att = torch.tensor(self.cfg.asmc_kd_att, device=self.device).expand(self.num_envs, -1)
        
        # ASMC gains
        self.Lam = torch.tensor(self.cfg.asmc_Lam, device=self.device).expand(self.num_envs, -1)
        self.Phi = torch.tensor(self.cfg.asmc_Phi, device=self.device).expand(self.num_envs, -1)
        self.alpha0 = torch.tensor(self.cfg.asmc_alpha0, device=self.device).expand(self.num_envs, -1)
        self.alpha1 = torch.tensor(self.cfg.asmc_alpha1, device=self.device).expand(self.num_envs, -1)
        self.M_bar = self.cfg.asmc_M_bar
        self.v_t = self.cfg.asmc_v_t
        
        # 2. Initialize State Tracking Tensors (Equivalent to C++ state variables)
        # Attitude PID States
        # self.int_error_att = torch.zeros((self.num_envs, 3), device=self.device)
        # self.last_att_error = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Rate tracking for angular acceleration (alpha_desired)
        self.rate_cmd_prev = torch.zeros((self.num_envs, 3), device=self.device)
        
        # ASMC Adaptive Gain States
        self.Kp0_t = torch.zeros((self.num_envs, 3), device=self.device)
        self.Kp1_t = torch.zeros((self.num_envs, 3), device=self.device)

        # self.max_tilt = 30.0 * math.pi / 180.0
        self.max_tilt = 45.0 * math.pi / 180.0

        self.alloc_coeffs = torch.tensor(self.cfg.asmc_alloc_coeffs, device=self.device)
        
        # Forward Allocation Matrix Coefficients (norm -> Nm)
        # Pre-calculated from the inverse matrix for efficiency
        c0, c1, c2, c3 = self.cfg.asmc_alloc_coeffs
        self.alloc_coeffs_fwd = torch.tensor([
            1/c0,            # Pitch coeff
            1/c1,            # Yaw coeff
            1/c2,            # Roll coeff
            -c3 / (c2 * c1)  # Roll-from-Yaw coupling coeff
        ], device=self.device)

        # self.q_des_state = torch.zeros((self.num_envs, 4), device=self.device)
        # self.q_des_state[:, 3] = 1.0 # Initialize w = 1.0 (flat/level)

    def reset_idx(self, env_ids):
        """Resets states for environments that have terminated/reset."""
        super().reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # Reset PID states
        # self.int_error_att[env_ids] = 0.0
        # self.last_att_error[env_ids] = 0.0
        self.rate_cmd_prev[env_ids] = 0.0
        
        # Reset ASMC Adaptive gains
        self.Kp0_t[env_ids] = 0.0
        self.Kp1_t[env_ids] = 0.0

        # self.q_des_state[env_ids] = self.robot_orientation[env_ids].clone()

    def update(self, command_actions):
        """
        ASMC cascade controller
        :param command_actions: tensor (num_envs, 4) -> [thrust, roll, pitch, yaw] desired
        :return: m*g normalized thrust and interial normalized torques (num_envs, 6)
        """
        self.reset_commands()
        
        # -----------------------------------------------------------------
        # 0. Set Thrust (Directly passed from command to Z-axis wrench)
        # -----------------------------------------------------------------
        # self.wrench_command[:, 2] = (
        #     (command_actions[:, 0] - self.gravity[:, 2]) * self.mass.squeeze()
        # )


        ####### RL OUTPUT ATTITUDE ##############################################################################
        
        # -----------------------------------------------------------------
        # 1. Outer Loop: Attitude to Rate (PID Control)
        # -----------------------------------------------------------------
        # Desired Attitude from RL Agent: [Roll, Pitch, Yaw]
        # attitude_desired = command_actions[:, 1:4]
        # attitude_current = self.robot_euler_angles
        
        # # Error Calculation + Yaw Wrapping
        # attitude_error = attitude_desired - attitude_current
        # attitude_error[:, 2] = torch.atan2(torch.sin(attitude_error[:, 2]), torch.cos(attitude_error[:, 2]))
        
        # # Integrator (with Anti-windup clamp)
        # self.int_error_att += attitude_error * self.dt
        # self.int_error_att = torch.clamp(self.int_error_att, -self.max_int_att, self.max_int_att)
        
        # # Derivative
        # att_error_dot = (attitude_error - self.last_att_error) / self.dt
        # self.last_att_error = attitude_error.clone()
        
        # # PID Law
        # rate_cmd = (self.kp_att * attitude_error) + (self.ki_att * self.int_error_att) + (self.kd_att * att_error_dot)
        
        # # Rate Saturation
        # rate_cmd = torch.clamp(rate_cmd, -self.max_rate, self.max_rate)
        
        # # Compute Desired Angular Acceleration (alpha_desired) via finite differences
        # alpha_des = (rate_cmd - self.rate_cmd_prev) / self.dt
        # self.rate_cmd_prev = rate_cmd.clone()

        # # -----------------------------------------------------------------
        # # 2. Inner Loop: Rate to Torque (Adaptive Sliding Mode Control)
        # # -----------------------------------------------------------------
        # # Current State
        # q_curr = self.robot_orientation
        # omega_curr = self.robot_body_angvel
        # R_curr = quat_to_rotation_matrix(q_curr)
        
        # # Desired State
        # q_des = quat_from_euler_xyz(attitude_desired[:, 0], attitude_desired[:, 1], attitude_desired[:, 2])
        # R_des = quat_to_rotation_matrix(q_des)
        # omega_des = rate_cmd

        ################################################################################################

        ######### RL OUTPUT CTBR #######################################################################
        # omega_des = command_actions[:, 1:4]
        # alpha_des = (omega_des - self.rate_cmd_prev) / self.dt
        # self.rate_cmd_prev = omega_des.clone()

        # # -----------------------------------------------------------------
        # # 1. KINEMATIC INTEGRATION: omega_des -> q_des
        # # -----------------------------------------------------------------
        # # Calculate angle of rotation: theta = ||omega|| * dt
        # omega_norm = torch.norm(omega_des, dim=1, keepdim=True) + 1e-8 # add epsilon to avoid div by zero
        # theta = omega_norm * self.dt
        
        # # Calculate rotation axis: v = omega / ||omega||
        # axis = omega_des / omega_norm
        
        # # Create Delta Quaternion: [x, y, z, w]
        # sin_half_theta = torch.sin(theta / 2.0)
        # delta_q = torch.cat([
        #     axis * sin_half_theta,          # x, y, z
        #     torch.cos(theta / 2.0)          # w
        # ], dim=1)
        
        # # Integrate: q_des_state = q_des_state * delta_q
        # self.q_des_state = quat_mul(self.q_des_state, delta_q)
        
        # # Normalize to prevent floating point drift
        # self.q_des_state = self.q_des_state / torch.norm(self.q_des_state, dim=1, keepdim=True)

        # # -----------------------------------------------------------------
        # # 2. ASMC Logic
        # # -----------------------------------------------------------------
        # # Current State
        # q_curr = self.robot_orientation
        # omega_curr = self.robot_body_angvel
        # R_curr = quat_to_rotation_matrix(q_curr)
        
        # # Desired State (from our integration)
        # q_des = self.q_des_state
        # R_des = quat_to_rotation_matrix(q_des)

        ################################################################################################

        ######### RL OUTPUT ACCTBR+YAW #################################################################
        """
        ASMC cascade controller (Acceleration + Yaw + Rates)
        :param command_actions: tensor (num_envs, 7+) 
            -> [ax, ay, az, p, q, r, yaw, ...joints]
        :return: m*g normalized thrust and interial normalized torques (num_envs, 6)
        """
        self.reset_commands()
        
        # -----------------------------------------------------------------
        # 0. Extract Commands from Action Space
        # -----------------------------------------------------------------
        accel_world = command_actions[:, 0:3]   # Desired world linear accelerations
        omega_des   = command_actions[:, 3:6]   # Desired body rates
        yaw_des     = command_actions[:, 6]     # Desired yaw
        
        # 1. Z-Acceleration -> Thrust (Replicating C++ Normalized Logic)
        # -----------------------------------------------------------------
        # hover_thrust_norm = 0.811 # From your C++ script
        # gravity_mag = torch.abs(self.gravity[:, 2]) # Usually 9.81
        
        # # C++ Logic: thrust_correction = a_z / gravity
        # thrust_correction = accel_world[:, 2] / gravity_mag
        
        # # In Aerial Gym (Z is UP), so we add correction to hover thrust. 
        # # (In C++ NED frame Z is DOWN, which is why your script subtracted it).
        # thrust_norm = hover_thrust_norm + thrust_correction
        
        # # Clamp thrust to safe range [0.05, 0.95] exactly like C++
        # thrust_norm = torch.clamp(thrust_norm, 0.05, 0.95)
        
        # # Convert normalized thrust back to physical Newtons for the simulator.
        # # Physics fact: hover_thrust_norm * Max_Motor_Thrust = mass * gravity
        # # Therefore: Max_Motor_Thrust = (mass * gravity) / hover_thrust_norm
        # max_thrust_newtons = (self.mass.squeeze() * gravity_mag) / hover_thrust_norm
        
        # # Final physical force applied to the Z-axis of the drone
        # self.wrench_command[:, 2] = thrust_norm * max_thrust_newtons
        # # -----------------------------------------------------------------
        # # 2. XY-Acceleration & Yaw -> Desired Roll/Pitch (Attitude)
        # # -----------------------------------------------------------------
        # # Use current yaw to rotate world accelerations into the body frame
        # current_yaw = self.robot_euler_angles[:, 2]
        # c_psi = torch.cos(current_yaw)
        # s_psi = torch.sin(current_yaw)
        
        # accel_world_x = accel_world[:, 0]
        # accel_world_y = accel_world[:, 1]
        
        # # Rotate acceleration: R_yaw^T * Global
        # accel_body_x = (accel_world_x * c_psi) + (accel_world_y * s_psi)
        # accel_body_y = (-accel_world_x * s_psi) + (accel_world_y * c_psi)
        
        # # Compute Desired Roll and Pitch matching C++ std::atan2 logic
        # roll_des  =  torch.atan2(accel_body_y, gravity_mag)
        # pitch_des = -torch.atan2(accel_body_x, gravity_mag)
        
        # # Saturate tilt to 30 degrees
        # roll_des  = torch.clamp(roll_des,  -self.max_tilt, self.max_tilt)
        # pitch_des = torch.clamp(pitch_des, -self.max_tilt, self.max_tilt)
        
        # -----------------------------------------------------------------
        # 1 & 2. Acceleration + Yaw -> Desired Attitude (Geometric Method)
        # -----------------------------------------------------------------
        # print(f"recieved lin acc: {accel_world[0, :]}")
        # print(f"recieved body rates: {omega_des[0, :]}")
        # print(f"recieved yaw des: {yaw_des[0]}")
        gravity_mag = torch.abs(self.gravity[:, 2]).unsqueeze(1) # [num_envs, 1] Usually 9.81
        
        # a_ref - g (ENU Frame: opposing gravity means adding 9.81 to Z)
        a_des = accel_world.clone()
        a_des[:, 2] += gravity_mag.squeeze()
        
        # Total required physical thrust (Magnitude of a_des)
        total_accel_norm = torch.norm(a_des, dim=1, keepdim=True)
        # print(f"accel cmd: {total_accel_norm}")
        
        # Calculate Physical Thrust Force (F = m * a)
        self.wrench_command[:, 2] = (total_accel_norm.squeeze() * self.mass.squeeze())
        # print(f"thrust_cmd: {self.wrench_command[0, 2]}")
        
        # Eq 17: z_d = a_des / ||a_des||
        z_d = a_des / (total_accel_norm + 1e-8) # Add epsilon to prevent div by zero
        
        # Eq 18: x_c = [cos(yaw), sin(yaw), 0]
        x_c = torch.zeros_like(a_des)
        x_c[:, 0] = torch.cos(yaw_des)
        x_c[:, 1] = torch.sin(yaw_des)
        
        # Eq 19: y_d = normalize(z_d x x_c)
        y_d_unnorm = torch.cross(z_d, x_c, dim=1)
        y_d = y_d_unnorm / (torch.norm(y_d_unnorm, dim=1, keepdim=True) + 1e-8)
        
        # Eq 20: x_d = y_d x z_d
        x_d = torch.cross(y_d, z_d, dim=1)
        
        # Eq 21: Construct Raw Rotation Matrix R = [x_d, y_d, z_d]
        R_raw = torch.stack((x_d, y_d, z_d), dim=2) # Shape: [num_envs, 3, 3]
        
        # --- EULER CLAMPING LOGIC (Matching C++ Parity) ---
        # Extract Roll: atan2(R(2, 1), R(2, 2))
        roll_raw = torch.atan2(R_raw[:, 2, 1], R_raw[:, 2, 2])
        # print(f"output_roll: {roll_raw[0]}")
        
        # Extract Pitch: asin(-R(2, 0))
        # Note: We clamp the input to asin between [-1, 1] because float32 inaccuracies 
        # can sometimes yield -1.000001, which causes torch.asin to return NaN!
        sin_pitch = torch.clamp(-R_raw[:, 2, 0], -1.0, 1.0)
        pitch_raw = torch.asin(sin_pitch)
        # print(f"output_pitch: {pitch_raw[0]}")

        
        # Clamp Roll and Pitch to max_tilt (e.g. 45 degrees)
        roll_clamped = torch.clamp(roll_raw, -self.max_tilt, self.max_tilt)
        pitch_clamped = torch.clamp(pitch_raw, -self.max_tilt, self.max_tilt)

        # yaw_raw = torch.atan2(R_raw[:, 1, 0], R_raw[:, 0, 0])
        
        # Convert clamped Euler angles back to Quaternion and Final Rotation Matrix
        # (We use the user's commanded yaw_des directly, bypassing extraction, which is mathematically safer)
        q_des = quat_from_euler_xyz(roll_clamped, pitch_clamped, yaw_des)
        R_des = quat_to_rotation_matrix(q_des)
        # -----------------------------------------------------------------
        # 3. ASMC Target Formulation
        # -----------------------------------------------------------------
        # Compute Desired Angular Acceleration (alpha_desired) via finite differences
        alpha_des = (omega_des - self.rate_cmd_prev) / self.dt
        self.rate_cmd_prev = omega_des.clone()
        # print(f"calc alpha_des: {alpha_des[0, :]}")


        # Current State
        q_curr = self.robot_orientation
        omega_curr = self.robot_body_angvel
        R_curr = quat_to_rotation_matrix(q_curr)
        ################################################################################################
        # print("===Entering ASMC===")


        
        # a) Attitude Error (Variational)
        R_curr_T = R_curr.transpose(1, 2)
        R_des_T = R_des.transpose(1, 2)
        
        # errQ = 0.5 * vee(R_des^T * R_curr - R_curr^T * R_des)
        R_err_mat = torch.bmm(R_des_T, R_curr) - torch.bmm(R_curr_T, R_des)
        errQ = 0.5 * extract_vee_map(R_err_mat)
        # print(f"attitude err: {errQ[0, :]}")

        
        # b) Angular Velocity Error
        # errAngVel = omega_curr - (R_curr^T * R_des * omega_des)
        R_cT_R_d = torch.bmm(R_curr_T, R_des)
        mapped_omega_des = torch.bmm(R_cT_R_d, omega_des.unsqueeze(2)).squeeze(2)
        errAngVel = omega_curr - mapped_omega_des
        # print(f"angvel err: {errAngVel[0, :]}")

        
        # c) Sliding Surface (s)
        sv = errAngVel + (self.Phi * errQ)
        sv_norm = torch.norm(sv, dim=1).unsqueeze(1) # shape (num_envs, 1) to broadcast with (num_envs, 3)
        
        # d) Adaptive Gain Update
        # Kp0_dot = norm(sv) - alpha0 * Kp0
        Kp0_dot = sv_norm - (self.alpha0 * self.Kp0_t)
        Kp1_dot = sv_norm - (self.alpha1 * self.Kp1_t)
        
        self.Kp0_t = torch.clamp((Kp0_dot * self.dt), 0.001, 0.3)
        self.Kp1_t = torch.clamp((Kp1_dot * self.dt), 0.001, 0.3)

        # self.Kp0_t = torch.clamp(self.Kp0_t + (Kp0_dot * self.dt), 0.001, 0.3)
        # self.Kp1_t = torch.clamp(self.Kp1_t + (Kp1_dot * self.dt), 0.001, 0.3)
        
        rho = self.Kp0_t + (self.Kp1_t * torch.abs(errQ))
        
        # e) Switching Term (delTau)
        delTau = rho * asmc_sigmoid(sv, self.v_t)
        
        # f) Control Law
        term1 = -self.Lam * sv
        term2 = self.M_bar * (alpha_des - (self.Phi * errAngVel))
        des_tq = term1 + term2 - delTau
        # print(f"term1: {term1[0]}")
        # print(f"term2: {term2[0]}")
        # print(f"delTau: {delTau[0]}")
        # print(f"des torque: {des_tq[0, :]}")

        # Saturation
        des_tq = torch.clamp(des_tq, -self.max_torque, self.max_torque)

        #########TITL PRIORITISED CONTROL ALLOCATION#############################################################
        # print("===ENtering Tilt Prioritization==")
        raw_roll_nm  = des_tq[:, 0]
        raw_pitch_nm = des_tq[:, 1]
        raw_yaw_nm   = des_tq[:, 2]
        # print(f"roll torque Nm b4: {raw_roll_nm[0]}")
        # print(f"pitch torque Nm b4: {raw_pitch_nm[0]}")
        # print(f"yaw torque Nm b4: {raw_yaw_nm[0]}")

        # Step A: Convert N-m to normalized commands using the allocation matrix
        norm_pitch_cmd = self.alloc_coeffs[0] * raw_pitch_nm
        norm_yaw_cmd   = self.alloc_coeffs[1] * raw_yaw_nm
        norm_roll_cmd  = (self.alloc_coeffs[2] * raw_roll_nm) + (self.alloc_coeffs[3] * raw_yaw_nm)

        # print(f"roll torque norm b4: {norm_roll_cmd[0]}")
        # print(f"pitch torque norm b4: {norm_pitch_cmd[0]}")
        # print(f"yaw torque norm b4: {norm_yaw_cmd[0]}")

        # Step B: Apply Tilt-Prioritization to the normalized commands
        # 1. TILT PRIORITY: Clamp Roll and Pitch strictly
        norm_roll_final  = torch.clamp(norm_roll_cmd, -1.0, 1.0)
        norm_pitch_final = torch.clamp(norm_pitch_cmd, -1.0, 1.0)

        # 2. CALCULATE REMAINING HEADROOM FOR YAW
        used_effort = torch.maximum(torch.abs(norm_roll_final), torch.abs(norm_pitch_final))
        yaw_headroom = torch.clamp(1.0 - used_effort, min=0.0)
        # print(f"yaw headroom: {yaw_headroom[0]}")
        
        # 3. CLAMP YAW to use only the remaining headroom
        norm_yaw_final = torch.clamp(norm_yaw_cmd, -yaw_headroom, yaw_headroom)
        # print(f"yaw torque norm after: {norm_yaw_final[0]}")

        # Recombine into the final normalized torque vector that PX4 would see
        des_tq_normalized = torch.stack((norm_roll_final, norm_pitch_final, norm_yaw_final), dim=1)
        c_fwd = self.alloc_coeffs_fwd
        final_pitch_nm = c_fwd[0] * norm_pitch_final
        final_yaw_nm   = c_fwd[1] * norm_yaw_final
        final_roll_nm  = (c_fwd[2] * norm_roll_final) + (c_fwd[3] * norm_yaw_final)

        physical_torque_nm = torch.stack((final_roll_nm, final_pitch_nm, final_yaw_nm), dim=1)

        # -----------------------------------------------------------------
        # 5. Apply to Vehicle Wrench (Back-convert to Physical SI Units)
        # -----------------------------------------------------------------
        # To send to the simulator, we must convert the normalized commands back to N-m.
        # This requires the FORWARD allocation matrix, which is the inverse of the one we used.
        
        # -----------------------------------------------------------------
        # 3. Apply to Vehicle Wrench
        # -----------------------------------------------------------------
        self.wrench_command[:, 3:6] = physical_torque_nm
        # print(f"Final CTBR: {self.wrench_command[0, 2:6]}")
        
        return self.wrench_command