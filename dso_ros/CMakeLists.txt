cmake_minimum_required(VERSION 2.4.6)
project(dso_ros)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  cv_bridge
)

add_message_files(
  FILES
  SE3Msg.msg
  SlidingWindowsMsg.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
)

set(DSO_PATH $ENV{DSO_PATH})
message("\n\n---- LOOKING FOR DSO at\n\"${DSO_PATH}\"")


find_package(Pangolin 0.2 REQUIRED)
message("\n\n---- FOUND Pangolin Headers at\n\"${Pangolin_INCLUDE_DIRS}\"")
message("---- FOUND Pangolin Libs at\n\"${Pangolin_LIBRARIES}\"")

find_package(OpenCV 3.3.4 REQUIRED)
message("\n\n---- FOUND OpenCV Headers at\n\"${OpenCV_INCLUDE_DIRS}\"")
message("---- FOUND OpenCV Libs at\n\"${OpenCV_LIBS}\"")

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 
find_library(DSO_LIBRARY dso ${DSO_PATH}/build/lib)

# SSE flags

set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O2 -g -std=c++0x -fno-omit-frame-pointer"
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${DSO_PATH}/src
  ${DSO_PATH}/thirdparty/Sophus
  ${OpenCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)  

catkin_package(
  CATKIN_DEPENDS message_runtime
)

## Declare a C++ executable
add_executable(dso_live src/main_xu.cpp)

target_link_libraries(dso_live
    ${DSO_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES})


install(
  TARGETS dso_live 
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


