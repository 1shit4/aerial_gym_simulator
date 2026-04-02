import torch
import numpy as np

EVAL = False

if EVAL == False:
    class task_config:
        seed = 56 #16 #26 #36 #46 #56
        sim_name = "base_sim"
        env_name = "empty_env"
        robot_name = "x500"
        controller_name = "il_control"
        args = {}
        num_envs = 8192
        use_warp = False
        headless = True
        device = "cuda:0"
        privileged_observation_space_dim = 0
        action_space_dim = 7
        observation_space_dim = 15
        episode_len_steps = 500
        return_state_before_reset = False
        reward_parameters = { }
        crash_dist = 5.5

        action_limit_max = torch.ones(action_space_dim,device=device) * 8.0
        action_limit_min = torch.ones(action_space_dim,device=device) * 0.0

        # action_limit_max = torch.tensor([15.0, 3.14, 3.14, 3.14], device="cuda:0", dtype=torch.float32)
        # action_limit_min = torch.tensor([-9.8, -3.14, -3.14, -3.14], device="cuda:0", dtype=torch.float32)
        def process_actions_for_task(actions, min_limit, max_limit):
            actions_clipped = torch.clamp(actions, -1, 1)

            # rescaled_command_actions = actions_clipped * (max_limit - min_limit)/2 + (max_limit + min_limit)/2
            actions_clipped[:,0] = actions_clipped[:,0]*5
            actions_clipped[:,1] = actions_clipped[:,1]*5
            actions_clipped[:,2] = actions_clipped[:,2]*9.81
            actions_clipped[:,3] = actions_clipped[:,3]*5 #8.0 for rate
            actions_clipped[:,4] = actions_clipped[:,4]*5 #8.0 for rate
            actions_clipped[:,5] = actions_clipped[:,5]*np.pi/5  #6.0
            actions_clipped[:,6] = actions_clipped[:,6]*np.pi/2 #np.pi / 3.0  #6.0

            return actions_clipped
        
else:
    class task_config:
        seed = 41
        sim_name = "base_sim"
        env_name = "empty_env"
        robot_name = "x500"
        controller_name = "il_control"
        args = {}
        num_envs = 8192
        use_warp = False
        headless = True
        device = "cuda:0"
        privileged_observation_space_dim = 0
        action_space_dim = 7
        observation_space_dim = 15
        episode_len_steps = 10000
        return_state_before_reset = False
        reward_parameters = { }

        crash_dist = 5.5

        # action_limit_max = torch.ones(action_space_dim,device=device) * 8.0
        # action_limit_min = torch.ones(action_space_dim,device=device) * 0.0

        action_limit_max = torch.tensor([15.0, 3.14, 3.14, 3.14], device="cuda:0", dtype=torch.float32)
        action_limit_min = torch.tensor([-9.8, -3.14, -3.14, -3.14], device="cuda:0", dtype=torch.float32)

        def process_actions_for_task(actions, min_limit, max_limit):
            actions_clipped = torch.clamp(actions, -1, 1)

            # rescaled_command_actions = actions_clipped * (max_limit - min_limit)/2 + (max_limit + min_limit)/2
            actions_clipped[:,0] = actions_clipped[:,0]*5
            actions_clipped[:,1] = actions_clipped[:,1]*5
            actions_clipped[:,2] = actions_clipped[:,2]*9.81
            actions_clipped[:,3] = actions_clipped[:,3]*5 #8.0 for rate
            actions_clipped[:,4] = actions_clipped[:,4]*5 #8.0 for rate
            actions_clipped[:,5] = actions_clipped[:,5]*np.pi/5  #6.0
            actions_clipped[:,6] = actions_clipped[:,6]*np.pi/2 #np.pi / 3.0  #6.0

            return actions_clipped
