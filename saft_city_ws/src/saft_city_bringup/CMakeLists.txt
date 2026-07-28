cmake_minimum_required(VERSION 3.0.2)
project(saft_city_bringup)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  move_base_msgs
  actionlib
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs move_base_msgs actionlib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  src/navigation_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
