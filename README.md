# Real time Pose Estimation and Dense Reconstruction Based on DSO and MVSNet
To improve my mathematical, machine learning and coding skills as an algorithm engineer, here is a passion project of mine (update weekly)   

System Description (to be updated)

Most existing monocular dense mapping algorithms often struggle to meet real-time requirements or only capable of producing low-quality reconstructions. To address these issues, This project aims to develop a novel real-time monocular dense mapping system with the following features:

* Loose Coupling Between Localisation and Mapping: This allows for high local accuracy in pose estimation and high-quality dense point cloud reconstruction.

* Pose Estimation: We use DSO (Direct Sparse Odometry) algorithm.

* Depth Estimation: We employ the state-of-the-art learning-based MVSNet technology to achieve real-time estimation of high-resolution depth maps.

* Map Construction: Photometric errors are used to effectively remove outliers. Additionally, we integrate adaptive voxel filtering to reduce the memory footprint of the dense point cloud map.

Why [DSO](https://github.com/JakobEngel/dso) + [MVSNET](https://github.com/prstrive/UniMVSNet)?: 

DSO (Direct Sparse Odometry) uses a sliding window approach for Bundle Adjustment (BA), which makes it an ideal input for MVSNet. In contrast to keyframe-based methods like ORB-SLAM, where choosing the right images for processing can be tricky, DSO's sliding window naturally provides a continuous and optimised set of images. This makes it easier and more effective to integrate with MVSNet for real-time, high-quality 3D reconstruction.

System: i5-12400f, Nvdia 3060Ti G6X, Asrock B660M-ITX

## Week1: Environment configuration
### ROS
fishros script (this is kinda cheating XD)

  ``` wget http://fishros.com/install -O fishros && . fishros ```
  
Installation check

  ``` dpkg -l | grep ros- ``` 
  
### DSO
1. Ubuntu 20.04 (DON't use Install third-party software, dont use 'Additional Driver' options in system 'Software & Updates' since it may cause black screen )
2. Blacklist nouveau:

``` 
sudo apt-get remove --purge '^nvidia-.*'
sudo nano /etc/modprobe.d/blacklist-nvidia-nouveau.conf
add: blacklist nouveau // options nouveau modeset=0
sudo update-initramfs -u
sudo reboot
```

3. Install the nv driver (535 for RTX 3060Ti and Ubuntu 20.04)

```
sudo apt install build-essential dkms
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt upgrade
sudo apt install nvidia-driver-535
```

4. Install Eigen3

```
sudo apt-get install libsuitesparse-dev libeigen3-dev libboost-all-dev
```
5. Install opencv (from source, 3.3.1):

```
mkdir ~/opencv_build && cd ~/opencv_build
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```    
cd to both opencv and opencv_contrib
        
```
git checkout 3.3.1
cd ~/opencv_build/opencv
mkdir build && cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    -D BUILD_EXAMPLES=ON ..
make -j4
sudo make install
pkg-config --modversion opencv (check)
```
    
Possible Error1: invalid conversion from ‘const char*’ to ‘char*’ [-fpermissive]
    in opencv/opencv/sources/modules/python/src2/cv2.cpp
Just replace
```char* str = PyString_AsString(obj);```

with

```char* str = (char*)PyString_AsString(obj);```

Possible Error2: ‘CODEC_FLAG_GLOBAL_HEADER’ was not declared in this scope         c->flags |= 
         CODEC_FLAG_GLOBAL_HEADER;
modify: /opt/opencv/opencv-3.2.0/modules/videoio/src/cap_ffmpeg_impl.hpp，add o top：
```
#define AV_CODEC_FLAG_GLOBAL_HEADER (1 << 22)
#define CODEC_FLAG_GLOBAL_HEADER AV_CODEC_FLAG_GLOBAL_HEADER
#define AVFMT_RAWPICTURE 0x0020
```
6. Install Pangolin (must be 0.6):
```
git clone https://github.com/stevenlovegrove/Pangolin
git checkout v0.6
sudo apt-get install libglew-dev
sudo apt-get install libboost-dev libboost-thread-dev libboost-filesystem-dev
sudo apt-get install ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev
sudo apt-get install libpng-dev
sudo apt-get install build-essential
cd Pangolin
mkdir build
cd build
cmake -DCPP11_NO_BOOST=1 ..
make -j
sudo make install
```
7. Install ziplib:
```
sudo apt-get install zlib1g-dev 
cd thirdparty #find dso/thirdparty 
tar -zxvf libzip-1.1.1.tar.gz 
cd libzip-1.1.1/ 
./configure 
make 
sudo make install 
sudo cp lib/zipconf.h /usr/local/include/zipconf.h
```
8. DSO
```
git clone https://github.com/JakobEngel/dso.git
cd dso 
mkdir build 
cd build 
cmake .. 
make
```
Possible Error3: /usr/local/include/sigslot/signal.hpp:109:79: error: ‘decay_t’ is not a member of ‘std’;              did you mean 'decay'?
```
replace
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O3 -g -std=c++0x -march=native"
   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
)
```
with
```
set(CMAKE_CXX_FLAGS "-std=c++11")
```
Possible Error4: dso/src/IOWrapper/OpenCV/ImageRW_OpenCV.cpp:37:35: error:                       
         CV_LOAD_IMAGE_GRAYSCALE’ was not declared in this scope
   CV_LOAD_IMAGE_[x] is from an older version of openCV and has been deprecated. 
Edit the file to have
```
#include <opencv2/imgcodecs.hpp>
```
Then replace all of the CV_LOAD_IMAGE_[x] with cv::IMREAD_[x], e.g. 

```CV_LOAD_IMAGE_GRAYSCALE -> cv::IMREAD_GRAYSCALE```

9. Testing 
Download the Dataset into the following directory then run in a terminal:

```./dso_dataset files=./sequence_12/images calib=./sequence_12/camera.txt gamma=./sequence_12/pcalib.txt vignette=./sequence_12/vignette.png preset=0 mode=0```

### UniMVSNet
1. CUDA 11.8 (12 not supported)
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
nano ~/.bashrc
```
Add to the last few lines
```
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LIBRARY_PATH=$LIBRARY_PAT:/usr/local/cuda/lib64
source ~/.bashrc
nvcc -V
```
2. CUDNN (https://developer.nvidia.com/rdp/cudnn-download)
```
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.9.3.28_1.0-1_amd64.deb 
sudo cp /var/cudnn-local-repo-ubuntu2004-8.9.3.28/cudnn-local-AAD7FE56-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8 libcudnn8-dev libcudnn8-samples
```
3. Pytorch 
```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118```
Stable (2.0.1) doesn't work because of this error, installed Preview (Nightly) instead
```ERROR: No matching distribution found for setuptools>=40.8.0```
Test
```
import torch
print(torch.rand(5, 3))
torch.cuda.is_available()
```

## Week2: A ROS API for DSO

The input required for MVSNet consists of the camera state and the images within a sliding window. To facilitate this, an API is needed to extract these data from DSO.

### Install [DSO_ROS](https://github.com/BluewhaleRobot/dso_ros)
```
export DSO_PATH=/home/shu/catkin_ws/src/dso
catkin_make
```
Testing
```
source catkin_ws/devel/setup.bash
roscore
rosbag play --pause MH_01_easy.bag
rosrun dso_ros dso_live image:=/cam0/image_raw calib='/home/shu/Database/MH01/cam0/camera.txt' mode=1
```
Use rostopic list to find the correct image source here
```
image:=/cam0/image_raw
```
### Steps to add a publisher to a exisiting CPP cmake project
The following sections will utilise the camera states, often referred to as `camToWorld`, as an example to illustrate how to develop such an API.

> **Note**: For the code related to handling images, please refer to the source code directly.

1. Define the  to-be-published message in dso_ros/msg/SE3Msg.msg
```
float64[16] camToWorld
```
2. Make a ros publisher in catkin_ws
0) include the message header
```
float64[16] camToWorld
```
1) define the publisher variable
```
#include "dso_ros/SE3Msg.h"
```
2) assigning values to the defined variable
```
dso_ros::SE3Msg SE3;
for(int i=0; i<4; i++)
    for(int j=0; j<4; j++)
        SE3.camToWorld[4*i+j] = fullSystem->get_curSE3().matrix()(i,j);
SE3Pub.publish(SE3);
```
3) initialise the ros node
```
ros::init(argc, argv, "dso_live");
ros::NodeHandle nh;
```
4) publish
```
SE3Pub = nh.advertise<dso_ros::SE3Msg>("curSE3", 10);
ros::spin();
```
5) Modify cmakelist.txt 
```
find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
  roscpp
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
```
6) Modify package.xml
```
<build_depend>cv_bridge</build_depend>
<build_depend>message_generation</build_depend>
  
<run_depend>cv_bridge</run_depend>
<run_depend>message_runtime</run_depend>
```
  
3. Implement a function to retrieve the values within the original cpp code
```
SE3 get_curSE3(); // in a .h file
// in the corresponding .cpp file
SE3 FullSystem::get_curSE3(){
    return allFrameHistory[allFrameHistory.size()-1]->camToWorld; // the last frame of state (newest)
}
```
4. Testing(4 terminals)
```
roscore
rosbag play --pause MH_01_easy.bag
rosrun dso_ros dso_live image:=/cam0/image_raw calib='/home/shu/Database/MH01/cam0/camera.txt' mode=1
rostopic echo /curSE3
```
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/f583aa74-778c-4050-a09a-2134dee3f1b9)

## Week3: Understanding PCL
### Testing (Already installed with ROS)
1. Simple Visulisation
 cmakelist.txt
```
cmake_minimum_required(VERSION 2.6)
project(pcl_test)

find_package(PCL 1.12 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_test pcl_test.cpp)

target_link_libraries (pcl_test ${PCL_LIBRARIES})

install(TARGETS pcl_test RUNTIME DESTINATION bin)
```
pcl_test.cpp
```
#include <iostream>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
 
 
int main(int argc, char **argv) {
    std::cout << "Test PCL !!!" << std::endl;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    uint8_t r(255), g(15), b(15);
    for (float z(-1.0); z <= 1.0; z += 0.05)
    {
      for (float angle(0.0); angle <= 360.0; angle += 5.0)
      {
        pcl::PointXYZRGB point;
        point.x = 0.5 * cosf (pcl::deg2rad(angle));
        point.y = sinf (pcl::deg2rad(angle));
        point.z = z;
        uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        point.rgb = *reinterpret_cast<float*>(&rgb);
        point_cloud_ptr->points.push_back (point);
      }
      if (z < 0.0)
      {
        r -= 12;
        g += 12;
      }
      else
      {
        g -= 12;
        b += 12;
      }
    }
    point_cloud_ptr->width = (int) point_cloud_ptr->points.size ();
    point_cloud_ptr->height = 1;
    
    pcl::visualization::CloudViewer viewer ("test");
    viewer.showCloud(point_cloud_ptr);
    while (!viewer.wasStopped()){ };
    return 0;
}
```
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/4768e8c0-8dbc-4ce5-af64-b91cc893eaf8)

2. SLAM 14, ch5
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/1275d8d9-ae96-4367-a9c0-a599ce6fc3fd)

### Understanding PCL Functionalities (pcl::PointCloud)
1. Type
```
pcl::PointCloud<pcl::PointXYZ> XYZ only
pcl::PointCloud<pcl::PointXYZRGB> XYZ with RGB
```
2. Create
```
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
```
  ```::Ptr: smart pointer (boost::shared_ptr)``` to the point cloud object. 
  ```new pcl::PointCloud<pcl::PointXYZ>()``` creates a new point cloud object
  ```cloud(new pcl::PointCloud<pcl::PointXYZ>)``` This initialises the smart pointer cloud with the newly created point cloud object.
4. Access
  1. By index
```
pcl::PointXYZ point = cloud->points[0]; // access the first point 
float x = point.x;
float y = point.y;
float z = point.z;
```
  2. Traverse 
```
for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud->begin(); it != cloud->end(); ++it) {
    pcl::PointXYZ point = *it;
}
```
4. Load
  ```pcl::io::loadPCDFile```
  ```pcl::io::loadPLYFile```
  ```pcl::io::loadOBJFile```
  e.g.
```
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::io::loadPCDFile("cloud.pcd", *cloud);
```
5. Save
  ```pcl::io::savePCDFile```
  ```pcl::io::savePLYFile```
  ```pcl::io::saveOBJFile```
  e.g.
```pcl::io::savePCDFile("output_cloud.pcd", *cloud);```
6. Coordinate Transformation
  1. Define transformation
```
Eigen::Affine3f transform = Eigen::Affine3f::Identity();
transform.translation() = translation;
transform.rotate(rotation);
```
  B. Perform transformation
```pcl::transformPointCloud(*cloud_original, *cloud_transformed, transform);```
