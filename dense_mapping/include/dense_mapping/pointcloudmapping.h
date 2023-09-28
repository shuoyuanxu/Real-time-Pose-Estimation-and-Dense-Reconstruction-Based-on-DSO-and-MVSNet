#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include <pcl/common/transforms.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <condition_variable>
#include <thread>
#include <mutex>

using namespace std;

class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;

    PointCloudMapping(double resolution_, float prob_threshold_);
    // Constructor, prob_threshold_ is for outlier removal (0.7~0.8)
    PointCloudT::Ptr get_globalMap();
    // Setting resolution for Voxel Filtering (to remove redundency)
    void set_resolution(double resolution);
    //insert a frame of information to the buffer list
    void insertKeyFrame(cv::Mat&  intrinsic, cv::Mat& extrinsic, cv::Mat& color, cv::Mat& depth, cv::Mat& confidence);
    void shutdown();
    // Getting an element from the buffer list
    void update_globalMap();

protected:
    //generate local pointcloud using 1 frame of info, called by update_globalMap()
    PointCloudT::Ptr generatePointCloud(cv::Mat&  intrinsic, cv::Mat& extrinsic, cv::Mat& color, cv::Mat& depth, cv::Mat& confidence);
    // globalmap variable
    PointCloudT::Ptr globalMap;
    std::shared_ptr<std::thread> viewerThread;

    bool shutDownFlag=false;
    std::mutex shutDownMutex;
    // Flag, if a new keyframe is inserted
    std::condition_variable  keyFrameUpdated;
    // mutal exclusion, ensuring shared data are not accesed by multiple threads simultaneously
    std::mutex               keyFrameUpdateMutex;

    // data to generate point clouds
    std::vector<cv::Mat> intrinsics;
    std::vector<cv::Mat> extrinsics;
    std::vector<cv::Mat> colorImgs;
    std::vector<cv::Mat> depthImgs;
    std::vector<cv::Mat> confidenceImgs;
    std::mutex keyframeMutex;
    uint16_t lastKeyframeSize = 0;
    float prob_threshold = 0.5;

    double resolution = 0.04;
    pcl::VoxelGrid<PointT> voxel;
    
};

#endif // POINTCLOUDMAPPING_H
