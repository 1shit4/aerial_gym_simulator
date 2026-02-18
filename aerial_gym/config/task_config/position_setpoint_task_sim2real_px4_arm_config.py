import torch

class task_config:
    seed = 56
    sim_name = "base_sim_4ms"
    env_name = "empty_env"
    robot_name = "x500arm"       # Must match robot_registry key
    controller_name = "lee_rates_control" # Must match controller_registry key
    args = {}
    num_envs = 4096
    use_warp = False
    headless = True
    device = "cuda:0"
    
    # Dimensions
    num_motors = 4
    num_joints = 2
    action_space_dim = num_motors + num_joints # 6
    
    # Observation: 15 (PX4 Standard) + 3 (EE rel pos) + 2 (Arm Pos) + 2 (Arm Vel)
    observation_space_dim = 40
    
    privileged_observation_space_dim = 0
    episode_len_steps = 950 # Short episodes for initial training  *0.01 = __s
    return_state_before_reset = False
    reward_parameters = {}
    crash_dist = 5.5

    # Limits
    # Drone: Accel [-9.8, 15], Rates [-3.14, 3.14]
    drone_min = [-9.8, -6.0, -6.0, -1.0]
    drone_max = [9.8,  6.0,  6.0,  1.0]
    # drone_max = [15.0, 3.14, 3.14, 3.14]
    # drone_min = [-9.8, -3.14, -3.14, -3.14]

    
    # Arm: Radians 90degs for j1, 120degs for j2
    arm_min = [-1.57, -2.094]
    arm_max = [1.57, 2.094]
    
    # Combine Limits
    action_limit_min = torch.tensor(drone_min + arm_min, device=device, dtype=torch.float32)
    action_limit_max = torch.tensor(drone_max + arm_max, device=device, dtype=torch.float32)

    def process_actions_for_task(actions, min_limit, max_limit):
        actions_clipped = torch.clamp(actions, -1, 1)
        actions_clipped[:,0] = actions_clipped[:,0]*9.81
        actions_clipped[:,1] = actions_clipped[:,1]*8.0
        actions_clipped[:,2] = actions_clipped[:,2]*8.0
        actions_clipped[:,3] = actions_clipped[:,3]*6.0
        actions_clipped[:,4] = actions_clipped[:,4]*1.57
        actions_clipped[:,5] = actions_clipped[:,5]*2.094

        # rescaled_actions = actions_clipped * (max_limit - min_limit)/2 + (max_limit + min_limit)/2
        return actions_clipped