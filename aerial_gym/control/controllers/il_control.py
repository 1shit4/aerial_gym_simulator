import torch
import math
import pytorch3d.transforms as p3d_transforms
from aerial_gym.utils.math import *
from aerial_gym.control.controllers.base_lee_controller import BaseLeeController, extract_vee_map

class ILAttitudeController(BaseLeeController):
    def __init__(self, config, num_envs, device):
        super().__init__(config, num_envs, device)
        
    def init_tensors(self, global_tensor_dict=None):
        super().init_tensors(global_tensor_dict)
        
        # 1. Load Parameters from Config
        self.dt = self.cfg.il_dt
        self.max_tilt = self.cfg.il_max_tilt
        
        # Jbar, Kp, Kd are diagonal, so we store them as vectors (shape: 3) for fast element-wise math
        self.Jbar = torch.tensor(self.cfg.il_Jbar, device=self.device).expand(self.num_envs, -1)
        self.Kp_eta = torch.tensor(self.cfg.il_Kp_eta, device=self.device).expand(self.num_envs, -1)
        self.Kd_eta = torch.tensor(self.cfg.il_Kd_eta, device=self.device).expand(self.num_envs, -1)
        self.torque_limits = torch.tensor(self.cfg.il_torque_limits, device=self.device).expand(self.num_envs, -1)
        
        # Filter coefficient
        self.alpha_filter = self.cfg.il_alpha_filter
        
        # Allocation Coefficients (from N-m to Normalized)
        self.alloc_coeffs = torch.tensor(self.cfg.il_alloc_coeffs, device=self.device)
        c0, c1, c2, c3 = self.cfg.il_alloc_coeffs
        
        # Inverse Allocation (from Normalized to N-m) for simulator wrench mapping
        self.alloc_coeffs_fwd = torch.tensor([
            1.0 / c0,              # Pitch coeff
            1.0 / c1,              # Yaw coeff
            1.0 / c2,              # Roll coeff
            c3 / (c2 * c1)         # Roll-from-Yaw coupling coeff
        ], device=self.device)

        # 2. State Tracking Tensors (Delay-based memory)
        self.is_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.prev_tau = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_omega = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_omega_ff_current = torch.zeros((self.num_envs, 3), device=self.device)
        
        # EMA Filter States
        self.filtered_omega_dot = torch.zeros((self.num_envs, 3), device=self.device)
        self.filtered_tau = torch.zeros((self.num_envs, 3), device=self.device)

    def reset_idx(self, env_ids):
        """Resets states for environments that have terminated/reset."""
        super().reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        self.is_initialized[env_ids] = False
        self.prev_tau[env_ids] = 0.0
        self.prev_omega[env_ids] = 0.0
        self.prev_omega_ff_current[env_ids] = 0.0
        
        self.filtered_omega_dot[env_ids] = 0.0
        self.filtered_tau[env_ids] = 0.0

    def update(self, command_actions):
        """
        IL cascade controller (Acceleration + Yaw + Rates)
        :param command_actions: tensor (num_envs, 7+) -> [ax, ay, az, p, q, r, yaw]
        :return: m*g normalized thrust and inertial physical torques (num_envs, 6)
        """
        self.reset_commands()
        
        # -----------------------------------------------------------------
        # 0. Extract Commands
        # -----------------------------------------------------------------
        accel_world = command_actions[:, 0:3]   # Desired world linear accelerations
        omega_des   = command_actions[:, 3:6]   # Desired body rates
        yaw_des     = command_actions[:, 6]     # Desired yaw
        
        # -----------------------------------------------------------------
        # 1 & 2. Acceleration + Yaw -> Desired Attitude & Thrust
        # -----------------------------------------------------------------
        gravity_mag = torch.abs(self.gravity[:, 2]).unsqueeze(1) 
        
        # a_ref - g (ENU Frame: opposing gravity means adding 9.81 to Z)
        a_des = accel_world.clone()
        a_des[:, 2] += gravity_mag.squeeze()
        
        # Total Thrust
        total_accel_norm = torch.norm(a_des, dim=1, keepdim=True)
        self.wrench_command[:, 2] = (total_accel_norm.squeeze() * self.mass.squeeze())
        
        # Geometric Attitude Extraction
        z_d = a_des / (total_accel_norm + 1e-8) 
        
        x_c = torch.zeros_like(a_des)
        x_c[:, 0] = torch.cos(yaw_des)
        x_c[:, 1] = torch.sin(yaw_des)
        
        y_d_unnorm = torch.cross(z_d, x_c, dim=1)
        y_d = y_d_unnorm / (torch.norm(y_d_unnorm, dim=1, keepdim=True) + 1e-8)
        
        x_d = torch.cross(y_d, z_d, dim=1)
        
        R_raw = torch.stack((x_d, y_d, z_d), dim=2) 
        
        # Euler Clamping
        roll_raw = torch.atan2(R_raw[:, 2, 1], R_raw[:, 2, 2])
        sin_pitch = torch.clamp(-R_raw[:, 2, 0], -1.0, 1.0)
        pitch_raw = torch.asin(sin_pitch)
        
        roll_clamped = torch.clamp(roll_raw, -self.max_tilt, self.max_tilt)
        pitch_clamped = torch.clamp(pitch_raw, -self.max_tilt, self.max_tilt)
        
        q_des = quat_from_euler_xyz(roll_clamped, pitch_clamped, yaw_des)
        R_des = quat_to_rotation_matrix(q_des)
        
        # -----------------------------------------------------------------
        # 3. Inner Loop: Adaptive Torque Law
        # -----------------------------------------------------------------
        q_curr = self.robot_orientation
        omega_curr = self.robot_body_angvel
        R_curr = quat_to_rotation_matrix(q_curr)
        
        # a) SO(3) Attitude Error
        R_curr_T = R_curr.transpose(1, 2)
        R_des_T = R_des.transpose(1, 2)
        
        R_err_mat = torch.bmm(R_des_T, R_curr) - torch.bmm(R_curr_T, R_des)
        e_R = 0.5 * extract_vee_map(R_err_mat)

        # b) Map Rate Feedforward
        R_cT_R_d = torch.bmm(R_curr_T, R_des)
        omega_ff_current = torch.bmm(R_cT_R_d, omega_des.unsqueeze(2)).squeeze(2)
        
        # c) Rate Error
        e_omega = omega_curr - omega_ff_current

        # d) Finite Differences
        raw_omega_dot_meas = torch.zeros_like(omega_curr)
        omega_ff_dot = torch.zeros_like(omega_curr)
        
        mask = self.is_initialized.unsqueeze(1)
        
        raw_omega_dot_meas = torch.where(mask, (omega_curr - self.prev_omega) / self.dt, torch.zeros_like(omega_curr))
        omega_ff_dot       = torch.where(mask, (omega_ff_current - self.prev_omega_ff_current) / self.dt, torch.zeros_like(omega_curr))

        # e) First-Order Low-Pass Filtering (EMA)
        self.filtered_omega_dot = torch.where(mask, 
                                              (self.alpha_filter * raw_omega_dot_meas) + ((1.0 - self.alpha_filter) * self.filtered_omega_dot), 
                                              torch.zeros_like(omega_curr))
                                              
        self.filtered_tau = torch.where(mask, 
                                        (self.alpha_filter * self.prev_tau) + ((1.0 - self.alpha_filter) * self.filtered_tau), 
                                        torch.zeros_like(omega_curr))

        # f) Delay-based Lumped Uncertainty Estimate
        Hhat_bar = self.filtered_tau - (self.Jbar * self.filtered_omega_dot)

        # Precompute terms for printing/debugging
        kp_term = -self.Kp_eta * e_R
        kd_term = -self.Kd_eta * e_omega
        Ia_term = self.Jbar * (omega_ff_dot + kp_term + kd_term)

        # g) Adaptive Torque Law
        torque_unsat = Ia_term + Hhat_bar

        # ====================================================================
        # DEBUG LOGGING (Uncomment to debug Env 0)
        # ====================================================================
        # print(f"Att Err: [{e_R[0, 0]:.2f}, {e_R[0, 1]:.2f}, {e_R[0, 2]:.2f}]")
        # print(f"w_d: [{omega_ff_current[0, 0]:.2f}, {omega_ff_current[0, 1]:.2f}, {omega_ff_current[0, 2]:.2f}]")
        # print(f"w_curr: [{omega_curr[0, 0]:.2f}, {omega_curr[0, 1]:.2f}, {omega_curr[0, 2]:.2f}]")
        # print(f"w_err: [{e_omega[0, 0]:.2f}, {e_omega[0, 1]:.2f}, {e_omega[0, 2]:.2f}]")
        # print(f"a_act: [{self.filtered_omega_dot[0, 0]:.2f}, {self.filtered_omega_dot[0, 1]:.2f}, {self.filtered_omega_dot[0, 2]:.2f}]")
        # print(f"a_d: [{omega_ff_dot[0, 0]:.2f}, {omega_ff_dot[0, 1]:.2f}, {omega_ff_dot[0, 2]:.2f}]")
        # print(f"prev_tau: [{self.filtered_tau[0, 0]:.2f}, {self.filtered_tau[0, 1]:.2f}, {self.filtered_tau[0, 2]:.2f}]")
        # print(f"H_pred: [{Hhat_bar[0, 0]:.2f}, {Hhat_bar[0, 1]:.2f}, {Hhat_bar[0, 2]:.2f}]")
        # print(f"-kp*att_err: [{kp_term[0, 0]:.2f}, {kp_term[0, 1]:.2f}, {kp_term[0, 2]:.2f}]")
        # print(f"-kd*w_err: [{kd_term[0, 0]:.2f}, {kd_term[0, 1]:.2f}, {kd_term[0, 2]:.2f}]")
        # print(f"Ia: [{Ia_term[0, 0]:.2f}, {Ia_term[0, 1]:.2f}, {Ia_term[0, 2]:.2f}]")
        # print(f"tau_unsat: [{torque_unsat[0, 0]:.2f}, {torque_unsat[0, 1]:.2f}, {torque_unsat[0, 2]:.2f}]")
        # ====================================================================

        # Saturation
        des_tq = torch.clamp(torque_unsat, -self.torque_limits, self.torque_limits)

        # -----------------------------------------------------------------
        # 4. PX4 Normalized Allocation & Tilt Prioritization
        # -----------------------------------------------------------------
        raw_roll_nm  = des_tq[:, 0]
        raw_pitch_nm = des_tq[:, 1]
        raw_yaw_nm   = des_tq[:, 2]

        # Convert N-m to normalized commands using YOUR allocation matrix
        norm_pitch_cmd = self.alloc_coeffs[0] * raw_pitch_nm
        norm_yaw_cmd   = self.alloc_coeffs[1] * raw_yaw_nm
        norm_roll_cmd  = (self.alloc_coeffs[2] * raw_roll_nm) - (self.alloc_coeffs[3] * raw_yaw_nm)

        # TILT PRIORITY
        norm_roll_final  = torch.clamp(norm_roll_cmd, -1.0, 1.0)
        norm_pitch_final = torch.clamp(norm_pitch_cmd, -1.0, 1.0)

        # Yaw Headroom
        used_effort = torch.maximum(torch.abs(norm_roll_final), torch.abs(norm_pitch_final))
        yaw_headroom = torch.clamp(1.0 - used_effort, min=0.0)
        norm_yaw_final = torch.clamp(norm_yaw_cmd, -yaw_headroom, yaw_headroom)

        # Back-convert to Physical Torque (N-m) for the Simulator Physics Engine
        c_fwd = self.alloc_coeffs_fwd
        final_pitch_nm = c_fwd[0] * norm_pitch_final
        final_yaw_nm   = c_fwd[1] * norm_yaw_final
        final_roll_nm  = (c_fwd[2] * norm_roll_final) + (c_fwd[3] * norm_yaw_final)

        physical_torque_nm = torch.stack((final_roll_nm, final_pitch_nm, final_yaw_nm), dim=1)

        # -----------------------------------------------------------------
        # 5. Apply to Vehicle Wrench & Save Memory
        # -----------------------------------------------------------------
        self.wrench_command[:, 3:6] = physical_torque_nm
        
        self.prev_tau = des_tq.clone()
        self.prev_omega = omega_curr.clone()
        self.prev_omega_ff_current = omega_ff_current.clone()
        self.is_initialized[:] = True
        
        return self.wrench_command