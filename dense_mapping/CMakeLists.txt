cmake_minimum_required(VERSION 2.4.6)
project(dense_mapping)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  cv_bridge
  unimvsnet
)

find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 

# SSE flags

set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O2 -g -std=c++0x -fno-omit-frame-pointer"
)

set(CMAKE_CXX_STANDARD 14)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include/dense_mapping
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)  

add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})
catkin_package(
  CATKIN_DEPENDS 
)

## Declare a C++ executable
add_executable(dense_mapping_node
src/dense_mapping.cpp
src/pointcloudmapping.cpp)

target_link_libraries(dense_mapping_node
    ${PCL_LIBRARY}
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES})


install(
  TARGETS dense_mapping_node 
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


