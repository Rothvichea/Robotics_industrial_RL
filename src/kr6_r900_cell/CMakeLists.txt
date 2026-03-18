cmake_minimum_required(VERSION 3.8)
project(kr6_r900_cell)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf srdf config launch meshes worlds materials
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/joint_goal_trajectory.py
    scripts/dual_arm_trajectory.py
    scripts/yolo_inspector_node.py
    scripts/pick_place_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
