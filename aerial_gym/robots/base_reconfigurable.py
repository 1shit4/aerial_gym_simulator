from aerial_gym.robots.base_multirotor import BaseMultirotor
import torch

from aerial_gym.utils.math import torch_rand_float_tensor, pd_control

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("reconfigurable_robot_class")


class BaseReconfigurable(BaseMultirotor):
    def __init__(self, robot_config, controller_name, env_config, device):
        super().__init__(
            robot_config=robot_config,
            controller_name=controller_name,
            env_config=env_config,
            device=device,
        )

        self.joint_init_state_min = torch.tensor(
            self.cfg.reconfiguration_config.init_state_min, device=self.device, dtype=torch.float32
        ).T.expand(self.num_envs, -1, -1)

        self.joint_init_state_max = torch.tensor(
            self.cfg.reconfiguration_config.init_state_max, device=self.device, dtype=torch.float32
        ).T.expand(self.num_envs, -1, -1)

        self.init_joint_response_params(self.cfg)

    def init_joint_response_params(self, cfg):
        self.joint_stiffness_min = torch.tensor(
            self.cfg.reconfiguration_config.min_stiffness, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        
        self.joint_stiffness_max = torch.tensor(
            self.cfg.reconfiguration_config.max_stiffness, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        
        self.joint_stiffness = torch.zeros((self.num_envs, cfg.reconfiguration_config.num_dofs), device=self.device, dtype=torch.float32)

        self.joint_damping = torch.tensor(
            cfg.reconfiguration_config.damping, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)

        # self.joint_damping_min = torch.tensor(
        #     cfg.reconfiguration_config.min_damping, device=self.device, dtype=torch.float32
        # ).expand(self.num_envs, -1)
        
        # self.joint_damping_max = torch.tensor(
        #     cfg.reconfiguration_config.max_damping, device=self.device, dtype=torch.float32
        # ).expand(self.num_envs, -1)

        # self.joint_damping = torch.zeros((self.num_envs, cfg.reconfiguration_config.num_dofs), device=self.device, dtype=torch.float32)

    def init_tensors(self, global_tensor_dict):
        super().init_tensors(global_tensor_dict)
        self.dof_states = global_tensor_dict["dof_state_tensor"]
        self.dof_control_mode = global_tensor_dict["dof_control_mode"]

        self.dof_effort_tensor = torch.zeros_like(self.dof_states[..., 0])
        self.dof_position_setpoint_tensor = torch.zeros_like(self.dof_states[..., 0])
        self.dof_velocity_setpoint_tensor = torch.zeros_like(self.dof_states[..., 0])

        self.dof_states_position = self.dof_states[..., 0]
        self.dof_states_velocity = self.dof_states[..., 1]

        global_tensor_dict["dof_position_setpoint_tensor"] = self.dof_position_setpoint_tensor
        global_tensor_dict["dof_velocity_setpoint_tensor"] = self.dof_velocity_setpoint_tensor
        global_tensor_dict["dof_effort_tensor"] = self.dof_effort_tensor

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.dof_states[env_ids, :] = torch_rand_float_tensor(
            lower=self.joint_init_state_min[env_ids],
            upper=self.joint_init_state_max[env_ids],
        )

        self.randomize_joint_params(env_ids)

    def randomize_joint_params(self, env_ids):
        """
        Generates new random stiffness and damping values for the specified environments.
        """
        num_resets = len(env_ids)
        if num_resets == 0:
            return

        # Generate random numbers between 0 and 1
        rand_stiffness = torch.rand((num_resets, self.joint_stiffness.shape[1]), device=self.device)
        # rand_damping = torch.rand((num_resets, self.joint_damping.shape[1]), device=self.device)

        # Scale to[min, max] using formula: rand * (max - min) + min
        self.joint_stiffness[env_ids] = rand_stiffness * (self.joint_stiffness_max[env_ids] - self.joint_stiffness_min[env_ids]) + self.joint_stiffness_min[env_ids]
        # self.joint_damping[env_ids] = rand_damping * (self.joint_damping_max[env_ids] - self.joint_damping_min[env_ids]) + self.joint_damping_min[env_ids]

    def call_arm_controller(self):
        """
        Call the controller for the arm of the quadrotor. This function is called every simulation step.
        """
        if self.dof_control_mode == "effort":
            # custom nonlinear response can be implemented here as per your needs.
            # currently, a simple PD controller is implemented
            pos_err = self.dof_position_setpoint_tensor - self.dof_states_position
            vel_err = self.dof_velocity_setpoint_tensor - self.dof_states_velocity
            self.dof_effort_tensor[:] = pd_control(
                pos_err,
                vel_err,
                self.joint_stiffness,
                self.joint_damping,
            )
        else:
            return

    def set_dof_position_targets(self, dof_pos_target):
        self.dof_position_setpoint_tensor[:] = dof_pos_target

    def set_dof_velocity_targets(self, dof_vel_target):
        self.dof_velocity_setpoint_tensor[:] = dof_vel_target

    def step(self, action_tensor):
        """
        Update the state of the quadrotor. This function is called every simulation step.
        """
        super().update_states()
        if action_tensor.shape[0] != self.num_envs:
            raise ValueError("Action tensor does not have the correct number of environments")
        self.action_tensor[:] = action_tensor
        # calling controller leads to control allocation happening, and
        super().call_controller()
        super().simulate_drag()
        super().apply_disturbance()
        self.call_arm_controller()
