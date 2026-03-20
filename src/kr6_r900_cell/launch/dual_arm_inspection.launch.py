import os
import xacro
from launch import LaunchDescription
from launch.actions import TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg    = get_package_share_directory('kr6_r900_cell')
    pkg_gz = get_package_share_directory('ros_gz_sim')

    # dual arm URDF
    xacro_file = os.path.join(pkg, 'urdf', 'kr6_dual_arm.urdf.xacro')
    robot_description_raw = xacro.process_file(xacro_file).toxml()

    srdf_file = os.path.join(pkg, 'srdf', 'kr6_dual_arm.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = f.read()

    kinematics = {
        'robot_description_kinematics': {
            'arm_1': {
                'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
                'kinematics_solver_search_resolution': 0.005,
                'kinematics_solver_timeout': 0.05,
            },
            'arm_2': {
                'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
                'kinematics_solver_search_resolution': 0.005,
                'kinematics_solver_timeout': 0.05,
            },
        }
    }

    planning = {
        'planning_plugin': 'ompl_interface/OMPLPlanner',
        'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints',
        'start_state_max_bounds_error': 0.1,
    }

    trajectory_execution = {
        'trajectory_execution': {
            'allowed_execution_duration_scaling': 5.0,
            'allowed_goal_duration_margin': 10.0,
            'allowed_start_tolerance': 0.1,
            'execution_duration_monitoring': False,
            'wait_for_trajectory_completion': False,
        },
        'moveit_controller_manager':
            'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        'moveit_simple_controller_manager': {
            'controller_names': ['arm_1_controller', 'arm_2_controller'],
            'arm_1_controller': {
                'action_ns': 'follow_joint_trajectory',
                'type': 'FollowJointTrajectory',
                'default': True,
                'joints': [
                    'arm_1_joint_1', 'arm_1_joint_2', 'arm_1_joint_3',
                    'arm_1_joint_4', 'arm_1_joint_5', 'arm_1_joint_6',
                ],
            },
            'arm_2_controller': {
                'action_ns': 'follow_joint_trajectory',
                'type': 'FollowJointTrajectory',
                'default': True,
                'joints': [
                    'arm_2_joint_1', 'arm_2_joint_2', 'arm_2_joint_3',
                    'arm_2_joint_4', 'arm_2_joint_5', 'arm_2_joint_6',
                ],
            },
        },
    }

    world_file = os.path.join(pkg, 'worlds', 'inspection_cell.sdf')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_raw,
            'use_sim_time': True,
        }],
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_gz, 'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={'gz_args': f'{world_file} -r -v1'}.items(),
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/inspection_cam/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/inspection_cam/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'kr6_dual_arm',
            '-x', '0.0', '-y', '0.0', '-z', '0.0',
        ],
        output='screen',
    )

    spawn_jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    spawn_arm1 = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_1_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    spawn_arm2 = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_2_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {'robot_description': robot_description_raw},
            {'robot_description_semantic': robot_description_semantic},
            {'use_sim_time': False},
            {'planning_plugin': 'ompl_interface/OMPLPlanner'},
            {'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints'},
            kinematics,
            planning,
            trajectory_execution,
        ],
    )

    rviz_config = os.path.join(pkg, 'config', 'moveit.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        parameters=[
            {'robot_description': robot_description_raw},
            {'robot_description_semantic': robot_description_semantic},
        ],
    )

    return LaunchDescription([
        robot_state_publisher,
        gz_sim,
        gz_bridge,
        TimerAction(period=3.0,  actions=[spawn_robot]),
        TimerAction(period=7.0,  actions=[spawn_jsb]),
        TimerAction(period=9.0,  actions=[spawn_arm1]),
        TimerAction(period=10.0, actions=[spawn_arm2]),
        TimerAction(period=12.0, actions=[move_group]),
        TimerAction(period=14.0, actions=[rviz]),
    ])
