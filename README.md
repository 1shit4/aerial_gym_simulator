# Goal-Conditioned Aerial-Manipulator RL in Aerial Gym

**Aerial Gym extensions for coordinated quadrotor–arm control, end-effector pose tracking, and controller-in-the-loop reinforcement learning.**

This repository is part of the [Robust Goal-Directed Aerial Manipulation project](https://github.com/1shit4/aerial-manipulation-rl-control). The larger research objective is to move an aerial manipulator's end effector toward a target pose while maintaining stable flight despite the coupled motion of the vehicle and arm. This repository implements the **RL simulation and training components** of that investigation.

The code builds on the open-source [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator), developed at NTNU. The underlying GPU simulation, environment infrastructure, and general-purpose controllers are from Aerial Gym; the work documented here concerns the **X500 aerial-manipulator setup, custom learning tasks, PX4-oriented policy interfaces, and disturbance-compensating controller integrations**.

## Aerial-Manipulator RL Task

The primary task is [`position_setpoint_task_sim2real_px4_arm`](aerial_gym/task/position_setpoint_task_sim2real_px4_arm/position_setpoint_task_sim2real_px4_arm.py). It uses an X500-style quadrotor with a **three-joint manipulator**, modeled in [`resources/robots/x500arm/`](resources/robots/x500arm/) and configured by [`x500arm_config.py`](aerial_gym/config/robot_config/x500arm_config.py).

The policy receives vehicle and manipulator states together with the end-effector goal. Its actions command **both the quadrotor and the arm**, so the learning objective is end-effector pose tracking rather than simply keeping the flying base at a position setpoint.

### Policy observations (33 dimensions)

| Observation | Dimensions | Description |
| --- | ---: | --- |
| Vehicle position relative to goal | 3 | Target position minus vehicle position |
| Vehicle orientation | 6 | Rotation in 6D representation |
| Vehicle linear and angular velocities | 6 | Translational velocity and body angular velocity |
| End-effector position error | 3 | Target position minus computed end-effector position |
| End-effector orientation error | 9 | Flattened relative rotation matrix |
| Manipulator joint positions and velocities | 6 | Three joint angles and three joint velocities |
| **Total** | **33** | |

End-effector position, orientation, and velocity are computed from the vehicle state and arm joint states using forward kinematics in the task implementation.

### Policy actions (10 dimensions)

| Action | Dimensions | Purpose |
| --- | ---: | --- |
| Desired vehicle acceleration | 3 | Translational motion reference |
| Desired body rates | 3 | Roll, pitch, and yaw angular-rate references |
| Desired yaw | 1 | Heading reference |
| Desired arm joint positions | 3 | Manipulator configuration references |
| **Total** | **10** | |

The normalized policy actions are scaled to controller and joint-reference ranges in [`position_setpoint_task_sim2real_px4_arm_config.py`](aerial_gym/config/task_config/position_setpoint_task_sim2real_px4_arm_config.py). The first seven components drive the vehicle controller; the last three specify arm-joint position targets.

### Reward design

The task's reward is centered on **end-effector position and orientation tracking**, with additional terms to encourage useful closed-loop behavior:

- Progress toward the end-effector target and reduced position error.
- End-effector orientation alignment and low end-effector velocity near the target.
- Vehicle uprightness and reduced yaw-rate motion.
- Smoothness and reasonable magnitude of commanded vehicle actions.

The task also provides episode termination and reset conditions. Reward terms and weights can be inspected in the task's [`compute_reward`](aerial_gym/task/position_setpoint_task_sim2real_px4_arm/position_setpoint_task_sim2real_px4_arm.py) implementation.

## Controller-in-the-Loop Training

One focus of this repository is **training with the inner-loop controller in the simulation**, rather than adding a flight controller only when running a trained policy in another environment. This lets policy actions influence the simulated vehicle through a disturbance-compensating control law, so the next observations and rewards reflect that closed-loop response.

The principal control path is:

```text
                  RL policy (10 actions)
                           |
            +--------------+--------------+
            |                             |
    Acceleration, body rates,          Arm joint
       and yaw references           position targets
            |                             |
    Disturbance-estimating          Joint PD control
       attitude controller                |
            |                             |
     Vehicle thrust/torque           Joint efforts
            |                             |
            +--------------+--------------+
                           |
                 Aerial Gym physics
                           |
               Updated vehicle + arm state
                           |
             End-effector observations/reward
                           |
                       RL policy
```

The active aerial-manipulator task selects [`il_control.py`](aerial_gym/control/controllers/il_control.py). It converts acceleration and yaw references into collective thrust and desired attitude, computes geometric attitude/rate errors, estimates lumped rotational disturbance using filtered torque and angular-acceleration feedback, and outputs a vehicle wrench to the simulator. The manipulator's position targets are tracked by a joint PD effort controller.

An alternative [`asmc.py`](aerial_gym/control/controllers/asmc.py) provides adaptive sliding-mode attitude control and is registered as `asmc_control`. These controllers implement the same underlying disturbance-estimating and adaptive sliding-mode control laws as their [C++ counterparts](https://github.com/1shit4/quadcopter_inner_loop_control), adapted for parallel simulation and controller-in-the-loop reinforcement learning in Aerial Gym.

## Training and Experiment Files

| Location | Purpose |
| --- | --- |
| [`aerial_gym/task/position_setpoint_task_sim2real_px4_arm/`](aerial_gym/task/position_setpoint_task_sim2real_px4_arm/) | Aerial-manipulator observations, actions, forward kinematics, rewards, and episode logic |
| [`aerial_gym/config/task_config/position_setpoint_task_sim2real_px4_arm_config.py`](aerial_gym/config/task_config/position_setpoint_task_sim2real_px4_arm_config.py) | Task, action scaling, observation dimensions, and controller selection |
| [`resources/robots/x500arm/`](resources/robots/x500arm/) and [`x500arm_config.py`](aerial_gym/config/robot_config/x500arm_config.py) | Robot model, three-joint configuration, and simulated physical properties |
| [`aerial_gym/control/controllers/il_control.py`](aerial_gym/control/controllers/il_control.py) | Disturbance-estimating vehicle attitude/torque controller |
| [`aerial_gym/control/controllers/asmc.py`](aerial_gym/control/controllers/asmc.py) | Adaptive sliding-mode controller variant |
| [`aerial_gym/rl_training/rl_games/`](aerial_gym/rl_training/rl_games/) | PPO configuration and RL-Games training runner |
| [`aerial_gym/task/position_setpoint_task_sim2real_px4/`](aerial_gym/task/position_setpoint_task_sim2real_px4/) | Related quadrotor-only task experiments |

The supplied [`ppo_aerial_quad.yaml`](aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml) selects the aerial-manipulator task and configures PPO training through RL-Games. After installing the dependencies described in the [upstream Aerial Gym setup guide](https://ntnu-arl.github.io/aerial_gym_simulator/2_getting_started/), the corresponding training entry point is:

```bash
cd aerial_gym/rl_training/rl_games
python runner.py --train \
  --file ppo_aerial_quad.yaml \
  --task position_setpoint_task_sim2real_px4_arm \
  --num_envs 512 --headless True --use_warp False
```

The environment count is adjustable for the available GPU. This is the repository's configured training path; successful execution depends on the required Isaac Gym, CUDA, PyTorch, and RL-Games installation.

## Related Repositories

- **[Aerial Manipulation — Project Overview](https://github.com/1shit4/aerial-manipulation-rl-control):** research objective and links to the full set of experimental implementations.
- **[RLquadcopter_aerial_gym](https://github.com/1shit4/RLquadcopter_aerial_gym):** separate ROS 2/PX4–Gazebo policy-inference and tracking experiments.
- **[quadcopter_inner_loop_control](https://github.com/1shit4/quadcopter_inner_loop_control):** related disturbance-compensating controllers implemented in C++ for PX4/Gazebo testing.
- **[arm_control_ws](https://github.com/1shit4/arm_control_ws):** aerial-manipulator modeling and arm-joint control in ROS 2/Gazebo.

## Upstream Framework

This is a research fork of **[NTNU Autonomous Robots Lab's Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator)**, not an independently developed simulator. Aerial Gym provides the underlying Isaac Gym environment, GPU-parallel simulation infrastructure, robot abstractions, and baseline control/training facilities. The original project's [documentation](https://ntnu-arl.github.io/aerial_gym_simulator/) and [BSD-3-Clause license](LICENSE) remain applicable.

*Research conducted collaboratively at the Robotics Research Center, IIIT Hyderabad.*
