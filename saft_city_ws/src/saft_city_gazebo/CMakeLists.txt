cmake_minimum_required(VERSION 3.0.2)
project(saft_city_gazebo)

find_package(catkin REQUIRED COMPONENTS
  gazebo_plugins
  gazebo_ros
  gazebo_ros_control
  xacro
)

catkin_package(
  CATKIN_DEPENDS gazebo_plugins gazebo_ros gazebo_ros_control xacro
)

# Install world files
install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
  FILES_MATCHING PATTERN "*.world"
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

# Install model files
install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

# Install scripts (if any Python files exist)
# catkin_install_python(PROGRAMS
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
