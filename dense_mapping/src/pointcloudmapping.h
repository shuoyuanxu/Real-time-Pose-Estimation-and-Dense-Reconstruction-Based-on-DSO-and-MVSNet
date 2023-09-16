#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include <pcl/common/transforms.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
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

    PointCloudT::Ptr get_globalMap();
    //设置体素滤波的分辨率
    void set_resolution(double resolution);
    //插入一帧数据
    void insertKeyFrame(cv::Mat&  intrinsic, cv::Mat& extrinsic, \
                        cv::Mat& color, cv::Mat& depth, cv::Mat& confidence);
    //更新全局地图与可视化线程
    void update_globalMap();
    void shutdown();
protected:
    //使用一帧数据生成局部点云地图
    PointCloudT::Ptr generatePointCloud(cv::Mat&  intrinsic, cv::Mat& extrinsic,\
                                        cv::Mat& color, cv::Mat& depth, cv::Mat& confidence);

    PointCloudT::Ptr globalMap;
    std::shared_ptr<std::thread> viewerThread;

    bool shutDownFlag=false;
    std::mutex shutDownMutex;

    std::condition_variable  keyFrameUpdated;
    std::mutex               keyFrameUpdateMutex;

    // data to generate point clouds
    std::vector<cv::Mat> intrinsics;
    std::vector<cv::Mat> extrinsics;
    std::vector<cv::Mat> colorImgs;
    std::vector<cv::Mat> depthImgs;
    std::vector<cv::Mat> confidenceImgs;
    std::mutex keyframeMutex;
    uint16_t lastKeyframeSize = 0;
    float prob_threshold = 0.75;

    double resolution = 0.04;
    pcl::VoxelGrid<PointT> voxel;
    
};

#endif // POINTCLOUDMAPPING_H

