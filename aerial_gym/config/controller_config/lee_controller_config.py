import numpy as np


class control:
    """
    Control parameters
    controller:
        lee_position_control: command_actions = [x, y, z, yaw] in environment frame scaled between -1 and 1
        lee_velocity_control: command_actions = [vx, vy, vz, yaw_rate] in vehicle frame scaled between -1 and 1
        lee_attitude_control: command_actions = [thrust, roll, pitch, yaw_rate] in vehicle frame scaled between -1 and 1
    kP: gains for position
    kV: gains for velocity
    kR: gains for attitude
    kOmega: gains for angular velocity
    """

    num_actions = 7
    max_inclination_angle_rad = np.pi / 3.0
    max_yaw_rate = 6.0 #np.pi / 3.0 , 10

    K_pos_tensor_max = [3.0, 3.0, 2.0]  # used for lee_position_control only
    K_pos_tensor_min = [2.0, 2.0, 1.0]  # used for lee_position_control only

    K_vel_tensor_max = [
        3.0,
        3.0,
        3.0,
    ]  # used for lee_position_control, lee_velocity_control only
    K_vel_tensor_min = [2.0, 2.0, 2.0]

    K_rot_tensor_max = [
        1.2,
        1.2,
        0.6,
    ]  # used for lee_position_control, lee_velocity_control and lee_attitude_control
    K_rot_tensor_min = [0.8, 0.8, 0.4]

    # K_angvel_tensor_max = [
    #     16,
    #     16.0,
    #     10,
    # ]  # used for lee_position_control, lee_velocity_control and lee_attitude_control
    K_angvel_tensor_max = [0.2, 0.2, 0.2]
    K_angvel_tensor_min = [0.1, 0.1, 0.1] #[16, 16.0, 10]

    randomize_params = False

    # ---------------------------------------------
    # ASMC (Adaptive Sliding Mode Control) Config
    # ---------------------------------------------
    # Note: dt should match your simulation control step (e.g., 0.01 for 100Hz)
    asmc_dt = 0.004 
    
    # Attitude -> Rate PID Gains
    asmc_kp_att = [2.0, 2.0, 2.0]
    asmc_ki_att = [0.0, 0.0, 0.0]
    asmc_kd_att = [0.0, 0.0, 0.0]
    asmc_max_int_att = 0.5
    asmc_max_rate = 2.0
    
    # Rate -> Torque ASMC Gains
    asmc_Lam = [0.12, 0.12, 1.0]
    asmc_Phi = [4.0, 4.0, 4.0]
    asmc_alpha0 = [2.0, 2.0, 2.0]
    asmc_alpha1 = [2.0, 2.0, 2.0]
    
    asmc_M_bar = 0.03
    asmc_v_t = 0.1
    asmc_max_torque = 1.0

    asmc_alloc_coeffs = [0.2372, 2.5801, 0.2372, -0.1229]
