import torch
import numpy as np

EVAL = False

if EVAL == False:
    class task_config:
        seed = 56
        sim_name = "base_sim_4ms"
        env_name = "empty_env"
        robot_name = "x500arm"       # Must match robot_registry key
        controller_name = "il_control" # Must match controller_registry key
        args = {}
        num_envs = 2
        use_warp = False
        headless = True
        device = "cuda:0"
        
        # Dimensions
        action_space_dim = 10 # 3 acc, 3 br, 1 yaw, 3 joint pos
        
        # Observation: 15 (PX4 Standard) + 3 (EE rel pos) + 2 (Arm Pos) + 2 (Arm Vel)
        observation_space_dim = 33
        
        privileged_observation_space_dim = 0
        episode_len_steps = 500 # Short episodes for initial training  *0.01 = __s
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
            # actions_clipped = torch.clamp(actions, -1, 1)
            # actions_clipped[:,0] = actions_clipped[:,0]*9.81
            # actions_clipped[:,1] = actions_clipped[:,1]*9.81
            # actions_clipped[:,2] = actions_clipped[:,2]*9.81
            # actions_clipped[:,3] = actions_clipped[:,3]*8.0 #8.0 for rate
            # actions_clipped[:,4] = actions_clipped[:,4]*8.0 #8.0 for rate
            # actions_clipped[:,5] = actions_clipped[:,5]*6.0 #np.pi / 3.0  #6.0
            # actions_clipped[:,6] = actions_clipped[:,6]*np.pi/2 #np.pi / 3.0  #6.0
            # actions_clipped[:,7] = actions_clipped[:,7]*1.57 -1.57
            # actions_clipped[:,8] = actions_clipped[:,8]*2.094
            # actions_clipped[:,9] = actions_clipped[:,9]*3.14
            actions_clipped = torch.clamp(actions, -1, 1)
            actions_clipped[:,0] = actions_clipped[:,0]*5
            actions_clipped[:,1] = actions_clipped[:,1]*5
            actions_clipped[:,2] = actions_clipped[:,2]*9.81
            actions_clipped[:,3] = actions_clipped[:,3]*5 #8.0 for rate
            actions_clipped[:,4] = actions_clipped[:,4]*5 #8.0 for rate
            actions_clipped[:,5] = actions_clipped[:,5]*np.pi/5  #6.0
            actions_clipped[:,6] = actions_clipped[:,6]*np.pi/2 #np.pi / 3.0  #6.0
            actions_clipped[:,7] = actions_clipped[:,7]*1.57 -1.57
            actions_clipped[:,8] = actions_clipped[:,8]*2.094
            actions_clipped[:,9] = actions_clipped[:,9]*3.14


            # rescaled_actions = actions_clipped * (max_limit - min_limit)/2 + (max_limit + min_limit)/2
            return actions_clipped

else:
    class task_config:
        class task_config:
            seed = 56
            sim_name = "base_sim_4ms"
            env_name = "empty_env"
            robot_name = "x500arm"       # Must match robot_registry key
            controller_name = "il_control" # Must match controller_registry key
            args = {}
            num_envs = 2
            use_warp = False
            headless = True
            device = "cuda:0"
            
            # Dimensions
            action_space_dim = 10 # 3 acc, 3 br, 1 yaw, 3 joint pos
            
            # Observation: 15 (PX4 Standard) + 3 (EE rel pos) + 2 (Arm Pos) + 2 (Arm Vel)
            observation_space_dim = 33
            
            privileged_observation_space_dim = 0
            episode_len_steps = 500 # Short episodes for initial training  *0.01 = __s
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
                # actions_clipped = torch.clamp(actions, -1, 1)
                # actions_clipped[:,0] = actions_clipped[:,0]*9.81
                # actions_clipped[:,1] = actions_clipped[:,1]*9.81
                # actions_clipped[:,2] = actions_clipped[:,2]*9.81
                # actions_clipped[:,3] = actions_clipped[:,3]*8.0 #8.0 for rate
                # actions_clipped[:,4] = actions_clipped[:,4]*8.0 #8.0 for rate
                # actions_clipped[:,5] = actions_clipped[:,5]*6.0 #np.pi / 3.0  #6.0
                # actions_clipped[:,6] = actions_clipped[:,6]*np.pi/2 #np.pi / 3.0  #6.0
                # actions_clipped[:,7] = actions_clipped[:,7]*1.57 -1.57
                # actions_clipped[:,8] = actions_clipped[:,8]*2.094
                # actions_clipped[:,9] = actions_clipped[:,9]*3.14
                actions_clipped = torch.clamp(actions, -1, 1)
                actions_clipped[:,0] = actions_clipped[:,0]*5
                actions_clipped[:,1] = actions_clipped[:,1]*5
                actions_clipped[:,2] = actions_clipped[:,2]*9.81
                actions_clipped[:,3] = actions_clipped[:,3]*5 #8.0 for rate
                actions_clipped[:,4] = actions_clipped[:,4]*5 #8.0 for rate
                actions_clipped[:,5] = actions_clipped[:,5]*np.pi/5  #6.0
                actions_clipped[:,6] = actions_clipped[:,6]*np.pi/2 #np.pi / 3.0  #6.0
                actions_clipped[:,7] = actions_clipped[:,7]*1.57 -1.57
                actions_clipped[:,8] = actions_clipped[:,8]*2.094
                actions_clipped[:,9] = actions_clipped[:,9]*3.14

                # print("Actions clipped: ", actions_clipped[0, :])

                # rescaled_actions = actions_clipped * (max_limit - min_limit)/2 + (max_limit + min_limit)/2
                return actions_clipped
