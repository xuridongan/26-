cmake_minimum_required(VERSION 3.0.2)
project(saft_city_description)

find_package(catkin REQUIRED COMPONENTS
  joint_state_publisher
  robot_state_publisher
  roslaunch
  urdf
  xacro
)

catkin_package(
  CATKIN_DEPENDS joint_state_publisher robot_state_publisher roslaunch urdf xacro
)

# Install URDF/XACRO files
install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
  FILES_MATCHING PATTERN "*.xacro" PATTERN "*.urdf"
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml" PATTERN "*.rviz"
)

# Install meshes
install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
  FILES_MATCHING PATTERN "*.dae" PATTERN "*.stl"
)
