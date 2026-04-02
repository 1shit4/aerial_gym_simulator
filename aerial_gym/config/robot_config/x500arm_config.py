import numpy as np

from aerial_gym import AERIAL_GYM_DIRECTORY

from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.lidar_config.base_lidar_config import (
    BaseLidarConfig,
)
from aerial_gym.config.sensor_config.lidar_config.osdome_64_config import OSDome_64_Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class X500ArmCfg:

    class init_config:
        # init_state tensor is of the format [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        min_init_state = [
            0,
            0,
            0.25,
            -np.pi / 6,
            -np.pi / 6,
            -np.pi,
            1.0,
            -0.5,
            -0.5,
            -0.5,
            -0.2,
            -0.2,
            -0.2,
        ]
        max_init_state = [
            1.0,
            1.0,
            1.0,
            np.pi / 6,
            np.pi / 6,
            np.pi,
            1.0,
            0.5,
            0.5,
            0.5,
            0.2,
            0.2,
            0.2,
        ]

        # min_init_state = [
        #     0,
        #     0,
        #     0.25,
        #     0.0,
        #     0.0,
        #     0.0,
        #     1.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0,
        # ]
        # max_init_state = [
        #     1.0,
        #     1.0,
        #     1.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     1.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0,
        # ]

    class sensor_config:
        enable_camera = False
        camera_config = BaseDepthCameraConfig

        enable_lidar = False
        lidar_config = BaseLidarConfig  # OSDome_64_Config

        enable_imu = False
        imu_config = BaseImuConfig

    class reconfiguration_config:
        num_dofs = 3 
        
        # Use "effort" if you want to use the Python PD controller we just wrote.
        # Use "position" if you want Nvidia PhysX to handle it (often stiffer/stable).
        dof_mode = "effort" 
        
        # Initialization ranges (Required by BaseReconfigurable)
        # 2 joints, so we need lists of length 2
        init_state_min = [
            [-3.04, -2.094, -3.14], # Position (Radians)
            [0.0, 0.0, 0.0], # Velocity (Rad/s)
        ]

        init_state_max = [
            [-0.1, 2.094, 3.14], # Position
            [0.0, 0.0, 0.0], # Velocity
        ]

        # Gains for the Python PD Controller: DONT TOUCH
        # stiffness = [12.0, 5.0, 2.0] 
        # damping = [0.5, 0.05, 0.05]

        # max_stiffness = [200.0, 120.0, 15.0]
        max_stiffness = [50.0, 30.0, 10.0]
        # max_stiffness = [8.0, 4.0, 2.0]
        min_stiffness = [8.0, 4.0, 2.0]

        damping = [0.5, 0.05, 0.05]    # Lower from 2.0   [2.0, 0.5] 
        ##intentionlly leaving delay to match gz deployment, stiffness can be increased firther to reachthe target quickly
        
        # Physical Limits (Optional, used if we add clamping logic later)
        max_effort = [100.0, 100.0, 100.0] 
        max_velocity = [1.0, 1.0, 1.0]

    class lee_rates_controller_config:
        # P-Gains for [Roll, Pitch, Yaw] rates
        # Tune these: Higher = snappier, Lower = smoother
        # K_angvel_tensor_max = [8.0, 8.0, 4.0]
        # K_angvel_tensor_min = [8.0, 8.0, 4.0]

        #========THESE R NOT EVEN BEING USED=========#

        K_angvel_tensor_max = [100.0, 100.0, 100.0]
        K_angvel_tensor_min = [100.0, 100.0, 100.0]
        
        # Unused gains (required by parent class)
        K_pos_tensor_min = [0.0, 0.0, 0.0]
        K_pos_tensor_max = [0.0, 0.0, 0.0]
        K_vel_tensor_min = [0.0, 0.0, 0.0]
        K_vel_tensor_max = [0.0, 0.0, 0.0]
        K_rot_tensor_min = [0.0, 0.0, 0.0]
        K_rot_tensor_max = [0.0, 0.0, 0.0]
        
        randomize_params = False
        max_yaw_rate = 10

    class disturbance:
        enable_disturbance = True
        prob_apply_disturbance = 0.01
        max_force_and_torque_disturbance = [2.5, 2.5, 1.0, 0.1, 0.1, 0.05]  # [fx, fy, fz, tx, ty, tz]

    class damping:
        linvel_linear_damping_coefficient = [0.1, 0.1, 0.1]  # along the body [x, y, z] axes
        linvel_quadratic_damping_coefficient = [0.3, 0.3, 0.5]  # along the body [x, y, z] axes
        angular_linear_damping_coefficient = [0.01, 0.01, 0.01]  # along the body [x, y, z] axes
        angular_quadratic_damping_coefficient = [0.01, 0.01, 0.01]  # along the body [x, y, z] axes

    class robot_asset:
        # UPDATED PATH for the new robot
        asset_folder = f"{AERIAL_GYM_DIRECTORY}/resources/robots/x500arm"
        file = "model.urdf"
        name = "x500arm"  # actor name
        base_link_name = "base_link"
        disable_gravity =  False
        collapse_fixed_joints = False #False  # merge bodies connected by fixed joints.
        fix_base_link = False  # fix the base of the robot
        collision_mask = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        density = 0.000001
        angular_damping = 0.02 #2.0 #differnet from aerodynamic drag, this is the internal damping of the robot's links, not related to velocity
        linear_damping = 0.02 #0.02
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.00001

        semantic_id = 0
        per_link_semantic = False

        min_state_ratio = [
            0.1,
            0.1,
            0.1,
            0,
            0,
            -np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        max_state_ratio = [
            0.9,
            0.9,
            0.9,
            0,
            0,
            np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]

        max_force_and_torque_disturbance = [
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
        ]  # [fx, fy, fz, tx, ty, tz]

        color = None
        semantic_masked_links = {}
        keep_in_env = True  # this does nothing for the robot

        min_position_ratio = None
        max_position_ratio = None

        min_euler_angles = [-np.pi, -np.pi, -np.pi]
        max_euler_angles = [np.pi, np.pi, np.pi]

        place_force_sensor = True  # set this to True if IMU is desired
        force_sensor_parent_link = "base_link"
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, qx, qy, qz, qw]

        use_collision_mesh_instead_of_visual = False  # does nothing for the robot

    class control_allocator_config:
        num_motors = 4
        force_application_level = "root_link" #"motor_link"  # "motor_link" or "root_link" decides to apply combined forces acting on the robot at the root link or at the individual motor links

        application_mask = [0, 1, 2, 3]
        
        # 2. Set Spin Directions [FR, BR, BL, FL]
        # 1 = CCW, -1 = CW. 
        # Standard X-Quad: Diagonals spin same way. Adjacent spin opposite.
        motor_directions = [1, -1, 1, -1]

        # 3. Allocation Matrix [Fx, Fy, Fz, Tx, Ty, Tz]
        # Cols: [Front_Right, Back_Right, Back_Left, Front_Left]
        allocation_matrix = [
            # Fx (None)
            [0.0, 0.0, 0.0, 0.0],
            # Fy (None)
            [0.0, 0.0, 0.0, 0.0],
            # Fz (Thrust - All Up)
            [1.0, 1.0, 1.0, 1.0],
            # Tx (Roll - Right is Neg, Left is Pos)
            [-0.174, -0.174, 0.174, 0.174],
            # Ty (Pitch - Front is Neg, Back is Pos)
            [-0.174, 0.174, 0.174, -0.174],
            # Tz (Yaw - CCW props Pos, CW props Neg)
            # Assuming torque coeff = 0.025
            [0.025, -0.025, 0.025, -0.025],
        ]

        # application_mask = [4, 1, 3, 2] # front right, back_left, front_left, back_right
        # motor_directions = [1, 1, -1, -1]

        # allocation_matrix = [
        #     [0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0],
        #     [1.0, 1.0, 1.0, 1.0],
        #     [-0.13, 0.13, 0.13, -0.13],
        #     [-0.13, 0.13, -0.13, 0.13],
        #     [-0.025, 0.025, -0.025, 0.025],
        # ]

        # allocation_matrix = [
        #     [0.0, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0, 0.0],
        #     [1.0, 1.0, 1.0, 1.0],
        #     [0.13, -0.13, 0.13, -0.13],
        #     [-0.13, -0.13, 0.13, 0.13],
        #     [0.025, -0.025, -0.025, 0.025],
        # ]

        class motor_model_config:
            use_rps = True
            motor_thrust_constant_min = 8.54858e-6 #0.00000926312
            motor_thrust_constant_max = 8.54858e-6 #0.00001826312
            motor_time_constant_increasing_min = 0.0125
            motor_time_constant_increasing_max = 0.0125
            motor_time_constant_decreasing_min = 0.025
            motor_time_constant_decreasing_max = 0.025
            max_thrust = 20.0
            min_thrust = 0.0
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.025
            use_discrete_approximation = False  # use discrete approximation for motor dynamics