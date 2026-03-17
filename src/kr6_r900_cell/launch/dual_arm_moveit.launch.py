import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():

    pkg = get_package_share_directory('kr6_r900_cell')

    # --- URDF ---
    xacro_file = os.path.join(pkg, 'urdf', 'kr6_dual_arm.urdf.xacro')
    robot_description_raw = xacro.process_file(xacro_file).toxml()

    # --- SRDF ---
    srdf_file = os.path.join(pkg, 'srdf', 'kr6_dual_arm.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = f.read()

    # --- kinematics: one solver per group ---
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
        'request_adapters': ' '.join([
            'default_planner_request_adapters/AddTimeOptimalParameterization',
            'default_planner_request_adapters/FixWorkspaceBounds',
            'default_planner_request_adapters/FixStartStateBounds',
            'default_planner_request_adapters/FixStartStateCollision',
            'default_planner_request_adapters/FixStartStatePathConstraints',
        ]),
        'start_state_max_bounds_error': 0.1,
    }

    trajectory_execution = {
        'trajectory_execution': {
            'allowed_execution_duration_scaling': 1.2,
            'allowed_goal_duration_margin': 0.5,
            'allowed_start_tolerance': 0.01,
        },
        'moveit_controller_manager':
            'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        'moveit_simple_controller_manager': {
            'controller_names': [
                'arm_1_controller',
                'arm_2_controller',
            ],
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

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw}],
    )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'robot_description': robot_description_raw}],
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {'robot_description': robot_description_raw},
            {'robot_description_semantic': robot_description_semantic},
            kinematics,
            planning,
            trajectory_execution,
            {'use_sim_time': False},
        ],
    )

    rviz_config = os.path.join(pkg, 'config', 'dual_arm.rviz')
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
        joint_state_publisher,
        move_group,
        rviz,
    ])
