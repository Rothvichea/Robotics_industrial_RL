import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro


def generate_launch_description():

    pkg = get_package_share_directory('kr6_r900_cell')

    xacro_file = os.path.join(pkg, 'urdf', 'kr6_r900_2.urdf.xacro')
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

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw}],
    )

    # joint_state_publisher with upright home position
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{
            'robot_description': robot_description_raw,
            'zeros.joint_1':  0.0,
            'zeros.joint_2': -1.5708,
            'zeros.joint_3':  1.5708,
            'zeros.joint_4':  0.0,
            'zeros.joint_5':  0.0,
            'zeros.joint_6':  0.0,
        }],
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
        joint_state_publisher,
        move_group,
        rviz,
    ])
