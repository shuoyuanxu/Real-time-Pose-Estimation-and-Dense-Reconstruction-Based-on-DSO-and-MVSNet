cmake_minimum_required(VERSION 2.4.6)
project(unimvsnet)

find_package(catkin REQUIRED COMPONENTS 
 rospy 
 message_generation
 sensor_msgs 
 std_msgs 
 cv_bridge)
 
add_message_files(FILES
 dso_ros.msg)
 
generate_messages(DEPENDENCIES 
 sensor_msgs
 std_msgs
 )

set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O2 -g -std=c++0x -fno-omit-frame-pointer"
)

set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -lpthread")

catkin_package(CATKIN_DEPENDS 
 message_runtime)

