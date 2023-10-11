# Real time Pose Estimation and Dense Reconstruction Based on DSO and MVSNet
To improve my mathematical, machine learning and coding skills as an algorithm engineer, here is a passion project of mine  


## System Description
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/aaad7197-5708-4cea-9e4f-1e8f313d825d)
Most existing monocular dense mapping algorithms often struggle to meet real-time requirements or only capable of producing low-quality reconstructions. To address these issues, This project aims to develop a novel real-time monocular dense mapping system with the following features:

* Loose Coupling Between Localisation and Mapping: This allows for high local accuracy in pose estimation and high-quality dense point cloud reconstruction.

* Pose Estimation: We use DSO (Direct Sparse Odometry) algorithm.

* Depth Estimation: We employ the state-of-the-art learning-based MVSNet technology to achieve real-time estimation of high-resolution depth maps.

* Map Construction: Photometric errors are used to effectively remove outliers. Additionally, we integrate adaptive voxel filtering to reduce the memory footprint of the dense point cloud map.

Why [DSO](https://github.com/JakobEngel/dso) + [MVSNET](https://github.com/prstrive/UniMVSNet)?: 

DSO (Direct Sparse Odometry) uses a sliding window approach for Bundle Adjustment (BA), which makes it an ideal input for MVSNet. In contrast to keyframe-based methods like ORB-SLAM, where choosing the right images for processing can be tricky, DSO's sliding window naturally provides a continuous and optimised set of images. This makes it easier and more effective to integrate with MVSNet for real-time, high-quality 3D reconstruction.

System: i5-12400f, Nvdia 3060Ti G6X, Asrock B660M-ITX

## 1. Environment configuration
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

## 2. A ROS Node for DSO

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

## 3. Understanding PCL
PCL will be used to process the output of MVSNet to obtain an accurate 3d reconstruction. Therefore, understanding the basics of PCL is required for the next stage of development 
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

2. [SLAM 14](https://github.com/gaoxiang12/slambook), ch5
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


## 4. Understanding [UniMVSNet](https://github.com/prstrive/UniMVSNet)
### Plane sweep algorithm in multi-view stereo
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/13181d0c-7dc5-4efc-a965-b6315d9168fd)
- How it works:
  1. Select a Reference Image: Choose one image from the set as the reference image.
  2. Create a Depth Map: Initialise a depth value for each pixel in the reference image.
  3. Plane Sweeping: For each of these depth values, find corresponding pixels in the other images and project them back onto the reference image. This step is often simplified through epipolar geometry.
  4. Compute Cost: For each depth value, calculate a cost metric to measure how similar the projected pixels are to the pixels in the reference image. Similarity metrics like SSD (Sum of Squared Differences) or NCC (Normalized Cross-Correlation) can be used.
  5. Select Best Depth: For each pixel in the reference image, select the depth that minimizes the cost (or maximizes the similarity).
  6. Generate 3D Model: Use the best depth values to back-project and generate a 3D model of the scene.
- Inputs:
  1. Multiple images (at least two), usually taken from different viewpoints.
  2. Camera intrinsics and extrinsics for projecting and back-projecting pixels to 3D points.
- Outputs:
  1. Depth map of the reference image.

### MVSNet
MVSNet is a direct extension of the plane sweep algorithm with:
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/70099e7b-3cce-4f4d-a976-74008d941cc9)
  1. An added deep feature extraction module
  2. Using learnt features to formulate cost  (construct a cost volume then regularise it to a probability volume to reduce the effect of noise)
  3. Soft argmin to generate depth map using P as weight over all hypothesised depth
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/3750f9ae-5183-4355-b4e2-08cb6a3fe354)
  4. Loss (refined depth considers the context)
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/b5af00db-d022-4463-88e0-6578d084ce51)


### UniMVSNet
The regression nature of the original MVSNet  requires the learning of learn a complex combination of weights which is non-trivial and tends to over-fit. The classification of  R-MVSNet finds the best depth hypothesis which ensures the robustness of MVS yet incapable predict depth directly. UniMVSNet combines the regression and classification approach in MVSNet and its variants. 
1. Classification: 
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/aa6e27ce-c169-4b38-bd5c-26abca68b6a8)

2. Regression: 
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/6e7d4a09-64d8-40f8-8e41-6cd063131efa)

3. Unification: classify to find the most likely hypothesis then regress to find its weight (done simultaneously as follows)
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/4b1fcd90-e793-4351-9faa-5983428fe9d2)

![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/d9e0214c-6a4e-487c-b1b0-4b737c4bd0f7)

![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/46a09c45-5656-413d-b9ad-3cfb94555007)

### Running UniMVSNet
1. Download Testing data DTU testing data and unzip it.
2. Download the trained model unimvsnet_dtu
3. Fusibile installation:  Download fusible,  cmake. , make
4. Point generation. To recreate the results from our paper, you need to specify the datapath to <your dtu_testing path>, outdir to <your output save path>, resume to <your model path>, and fusibile_exe_path to <your fusibile path>/fusibile in shell file ./script/dtu_test.sh first and then run:
```bash ./scripts/dtu_test.sh```
5. Result
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/acab31dc-4e53-4cfa-8f4c-3f022fb8854b)

![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/bdd6b295-365e-45ca-8062-4e0da56fc78f)

A simple script for visualisation pfm
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/06e77bf0-e358-4809-b18f-eb8e39bfe484)


## 5. A ROS node for UniMVSNet
### Understanding how UniMVSNet Runs
- Input and Output
  1. Input: Batch of images, camera intrinsic and extrinsic
  2. Output: Depth Map and Confidence Map
- How UniMVSNet runs
  1. The main function is called model.py
  2. Model uses args to initialise itself, then uses model.main() to run the testing
  3. Use images, camera extrinsics, and initial depth to run the network
  ``` outputs = self.network(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"]) ```
  4. Finally processing the 'output' to get depth and conf map 

- For every time step, we need to run UniMVSNet using the input of 
  1. Images in the current DSO sliding window 
  2. Camera extrinsic  from DSO
  3. Camera intrinsic

#### Therefore, we need to modify main.py to subscribe to the ros node we developed in DSO then use such data on the self.network. We do it in the following order:
1. Defining a ros msg for data transfer
2. Modifying the model.py so that it can take ros subed data as input
3. Create a launch script (following the style of main.py) to run the ros node of UniMVSNet and publish the depth map and confidence map.

### Steps 
#### Make a cmake project for msg used by UniMVSNet
1. DepthMsg.msg
```
sensor_msgs/Image image
float64[16] camToWorld
float64[4] Intrinsics
sensor_msgs/Image depth
sensor_msgs/Image confidence
```
2. CMakeLists.txt and package.xml
#### Make a Ros Node
1. Adding a member function in model.py to use images from ros
```
def test_ros(self, imgs, proj_matrices_ms,depth_min,depth_max):
  self.network.eval()
  
  depth_interval = (depth_max - depth_min) / self.args.numdepth
  depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
  
  sample = {"imgs": torch.from_numpy(np.expand_dims(imgs,axis=0)),
            "proj_matrices": proj_matrices_ms,
            "depth_values": torch.from_numpy(np.expand_dims(depth_values,axis=0))}
  #print(sample)
  sample_cuda = tocuda(sample)
  start_time = time.time()
  # get output
  outputs = self.network(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
  end_time = time.time()
  outputs = tensor2numpy(outputs)
  del sample_cuda
  imgs = sample["imgs"].numpy()
  print('Time:{} Res:{}'.format(end_time - start_time, imgs[0].shape))
  torch.cuda.empty_cache()
  return outputs
```
Adding 'test_ros' as a member function because we need to use 'self.network' to run the model. We simplify the data processing of 'output', which will be done in 'unimvsnet_node.py'
2. Adding a ros node to UniMVSNet (added to a launch script)
3. Testing
```
source catkin_ws/devel/setup.bash
rosrun unimvsnet unimvsnet_node.py
```
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/80b97cc3-f54a-4124-a033-6047cf349c98)

## 6. A ROS node for merging point cloud
### Input and Output
#### Input
1. Images
2. Depth Map
3. Confidence Map
4. Camera Intrinsics and Extrinsics
#### Output
Global Point Cloud
### Steps
#### Receiving input through ROS
```
imgSub = nh.subscribe("depth_info", depthInfoQueueSize, &vidCb);
```
#### Create a thread for visulisation
```
viewerThread = std::make_shared<std::thread>( std::bind(&PointCloudMapping::update_globalMap, this ) ); //mutex is required to prevent accessing conflict by different threads
```
#### Create another thread for point cloud merging (see update_globalMap() in pointcloudmapping.cpp)
1. Creating a local point cloud using input from UniMVSNet
```
PointCloudT::Ptr p = generatePointCloud( intrinsics[i], extrinsics[i], colorImgs[i], depthImgs[i], confidenceImgs[i]);
```
2. Adding the generated point cloud to the globalMap
```
*globalMap += *p;
```
3. Performs voxel filtering
```
voxel.filter( *tmp );
```
4. Visulisation
```
viewer->updatePointCloud(globalMap, "globalMap");
```
#### Publishing output through ROS
```
pointCloudpub = nh.advertise<sensor_msgs::PointCloud2> ("global_map", 1);
```
#### Running the whole project (5 terminals or use the launch script in dense_mapping)
```
source catkin_ws/devel/setup.bash
roscore
rosbag play --pause MH_01_easy.bag
rosrun dso_ros dso_live image:=/cam0/image_raw calib='/home/shu/Database/MH01/cam0/camera.txt' mode=1
rosrun unimvsnet unimvsnet_node.py
rosrun dense_mapping dense_mapping_node
```
#### Results
![image](https://github.com/shuoyuanxu/Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet/assets/21218812/724ebfed-5435-43c1-baf1-457edf93c273)
Video
https://www.youtube.com/watch?v=DwrOYyBTUiY

## Methods to improve the performance of this project
### Introducing other sensors (IMU)
### Using better perfromed MVSNet
### Truncated Signed Distance Function (TSDF) for global map

