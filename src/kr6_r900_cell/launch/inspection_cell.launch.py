import os
import xacro
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    pkg    = get_package_share_directory('kr6_r900_cell')
    pkg_gz = get_package_share_directory('ros_gz_sim')

    ompl_yaml = os.path.join(pkg, 'config', 'ompl_planning.yaml')
    import yaml
    with open(ompl_yaml, 'r') as f:
        ompl_config = yaml.safe_load(f)

    xacro_file        = os.path.join(pkg, 'urdf', 'kr6_r900_2.urdf.xacro')
    robot_description_raw = xacro.process_file(xacro_file).toxml()

    srdf_file = os.path.join(pkg, 'srdf', 'kr6_r900_2.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = f.read()

    kinematics = {
        'robot_description_kinematics': {
            'manipulator': {
                'kinematics_solver': 'kdl_kinematics_plugin/KDLKinematicsPlugin',
                'kinematics_solver_search_resolution': 0.005,
                'kinematics_solver_timeout': 0.05,
            }
        }
    }

    planning = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        },
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
            'controller_names': ['joint_trajectory_controller'],
            'joint_trajectory_controller': {
                'action_ns': 'follow_joint_trajectory',
                'type': 'FollowJointTrajectory',
                'default': True,
                'joints': [
                    'joint_1', 'joint_2', 'joint_3',
                    'joint_4', 'joint_5', 'joint_6',
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

    # bridge clock + camera image from Ignition to ROS2
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
            '-name', 'kr6_r900',
            '-x', '0.0', '-y', '0.0', '-z', '0.0',
        ],
        output='screen',
    )

    spawn_jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster',
                   '--controller-manager', '/controller_manager'],
        output='screen',
    )

    spawn_jtc = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller',
                   '--controller-manager', '/controller_manager'],
        output='screen',
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {'robot_description': robot_description_raw},
            {'robot_description_semantic': robot_description_semantic},
            {'use_sim_time': True},
            {'planning_plugin': 'ompl_interface/OMPLPlanner'},
            {'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints'},
            {'start_state_max_bounds_error': 0.1},
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
            {'use_sim_time': True},
        ],
    )

    return LaunchDescription([
        robot_state_publisher,
        gz_sim,
        gz_bridge,
        TimerAction(period=3.0,  actions=[spawn_robot]),
        TimerAction(period=7.0,  actions=[spawn_jsb]),
        TimerAction(period=9.0,  actions=[spawn_jtc]),
        TimerAction(period=11.0, actions=[move_group]),
        TimerAction(period=13.0, actions=[rviz]),
    ])
