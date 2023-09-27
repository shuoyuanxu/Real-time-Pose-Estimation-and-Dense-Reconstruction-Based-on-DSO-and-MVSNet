cmake_minimum_required(VERSION 3.0.2)
project(dense_mapping)


find_package(catkin REQUIRED COMPONENTS
#  pcl_conversions
#  pcl_ros
  roscpp
  rospy
  sensor_msgs
  cv_bridge
  unimvsnet
)

find_package(OpenCV 4.2 REQUIRED)
message("\n\n---- FOUND OpenCV Headers at\n\"${OpenCV_INCLUDE_DIRS}\"")
message("---- FOUND OpenCV Libs at\n\"${OpenCV_LIBS}\"")

find_package(PCL 1.7 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include/dense_mapping
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
) 

add_definitions( ${PCL_DEFINITIONS} )
link_directories( ${PCL_LIBRARY_DIRS} )

catkin_package( CATKIN_DEPENDS
)

add_executable(dense_mapping_node
  src/dense_mapping.cpp
  src/pointcloudmapping.cpp
)
target_link_libraries(dense_mapping_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

install(
  TARGETS dense_mapping_node 
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
  
