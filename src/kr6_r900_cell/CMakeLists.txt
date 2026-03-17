cmake_minimum_required(VERSION 3.8)
project(kr6_r900_cell)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf srdf config launch meshes
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/joint_goal_trajectory.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
