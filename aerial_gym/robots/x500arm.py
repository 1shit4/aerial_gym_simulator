from aerial_gym.robots.base_reconfigurable import BaseReconfigurable
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import pd_control
import torch

logger = CustomLogger("x500arm_robot_class")

class X500Arm(BaseReconfigurable):
    def __init__(self, robot_config, controller_name, env_config, device):
        # 1. Initialize the Parent Class
        # This triggers the chain: BaseReconfigurable -> BaseMultirotor -> BaseRobot
        # It sets up all the tensors (positions, velocities) automatically.
        super().__init__(
            robot_config=robot_config,
            controller_name=controller_name,
            env_config=env_config,
            device=device,
        )

    def init_joint_response_params(self, cfg):
        """
        This function is called automatically during __init__.
        It loads the P (stiffness) and D (damping) gains from your config file
        and prepares them as GPU tensors so we can calculate physics in parallel.
        """
        # Read 'stiffness' from the config file
        self.joint_stiffness = torch.tensor(
            cfg.reconfiguration_config.stiffness, 
            device=self.device, 
            dtype=torch.float32
        ).expand(self.num_envs, -1)

        # Read 'damping' from the config file
        self.joint_damping = torch.tensor(
            cfg.reconfiguration_config.damping, 
            device=self.device, 
            dtype=torch.float32
        ).expand(self.num_envs, -1)

    def call_arm_controller(self):
        """
        This is called every single physics step (e.g., 100 times per second).
        It calculates the forces (torques) the motors should apply to move the arm.
        """
        
        # We only calculate torques manually if we selected "effort" mode in the config.
        # If you selected "position" mode, Nvidia PhysX handles this internally.
        if self.dof_control_mode == "effort":
            
            # 1. Calculate the error: Where do we want to be vs Where are we?
            # These tensors are automatically updated by BaseReconfigurable
            pos_err = self.dof_position_setpoint_tensor - self.dof_states_position
            vel_err = self.dof_velocity_setpoint_tensor - self.dof_states_velocity

            # 2. Apply the PD Control formula:
            # Torque = (P * position_error) + (D * velocity_error)
            self.dof_effort_tensor[:] = pd_control(
                pos_err,
                vel_err,
                self.joint_stiffness,
                self.joint_damping,
            )
            
            # Note: The BaseReconfigurable.step() function will take this 
            # self.dof_effort_tensor and send it to the Physics engine.
            
        else:
            return