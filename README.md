# Real-time-Pose-Estimation-and-Dense-Reconstruction-Based-on-DSO-and-MVSNet
To improve my mathematical, machine learning and coding skills as an algorithm engineer, here is a passion project of mine (update weekly)   

System Description (to be updated)

Most existing monocular dense mapping algorithms often struggle to meet real-time requirements or only capable of producing low-quality reconstructions. To address these issues, This project aims to develop a novel real-time monocular dense mapping system with the following features:

* Loose Coupling Between Localisation and Mapping: This allows for high local accuracy in pose estimation and high-quality dense point cloud reconstruction.

* Pose Estimation: We use DSO (Direct Sparse Odometry) algorithm.

* Depth Estimation: We employ the state-of-the-art learning-based MVSNet technology to achieve real-time estimation of high-resolution depth maps.

* Map Construction: Photometric errors are used to effectively remove outliers. Additionally, we integrate adaptive voxel filtering to reduce the memory footprint of the dense point cloud map.

Why DSO + MVSNET?: 

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
