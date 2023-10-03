#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <time.h>

#include "cv_bridge/cv_bridge.h"
#include "unimvsnet/DepthMsg.h"
#include "pointcloudmapping.h"

ros::Subscriber imgSub;
ros::Publisher pointCloudpub;
PointCloudMapping* pointcloud_mapping;
time_t start_time=0,end_time=0; 

void vidCb(const unimvsnet::DepthMsg DepthMsg)
{
	//Disregards the first 5 frames
	static int count = 0;
	count++;
	if(count < 5) return;
	// Convert data to cv matrix format
	cv_bridge::CvImagePtr img = cv_bridge::toCvCopy(DepthMsg.image, sensor_msgs::image_encodings::BGR8);
	assert(img->image.type() == CV_8UC3);
	assert(img->image.channels() == 3);
	
	cv_bridge::CvImagePtr depth = cv_bridge::toCvCopy(DepthMsg.depth, sensor_msgs::image_encodings::TYPE_32FC1);
	assert(depth->image.type() == CV_32F);
	assert(depth->image.channels() == 1);
	
	cv_bridge::CvImagePtr confidence = cv_bridge::toCvCopy(DepthMsg.confidence, sensor_msgs::image_encodings::TYPE_32FC1);
	assert(confidence->image.type() == CV_32F);
	assert(confidence->image.channels() == 1);

	cv::Mat intrinsic = (cv::Mat_<double>(3,3) << DepthMsg.Intrinsics[0], 0, DepthMsg.Intrinsics[2], \
													0, DepthMsg.Intrinsics[1], DepthMsg.Intrinsics[3], \
													0, 0, 1);
	cv::Mat extrinsic = (cv::Mat_<double>(4,4) << DepthMsg.camToWorld[0], DepthMsg.camToWorld[1], DepthMsg.camToWorld[2], DepthMsg.camToWorld[3],\
												  DepthMsg.camToWorld[4], DepthMsg.camToWorld[5], DepthMsg.camToWorld[6], DepthMsg.camToWorld[7],\
												  DepthMsg.camToWorld[8], DepthMsg.camToWorld[9], DepthMsg.camToWorld[10], DepthMsg.camToWorld[11],\
												  DepthMsg.camToWorld[12], DepthMsg.camToWorld[13], DepthMsg.camToWorld[14], DepthMsg.camToWorld[15]);
	
	pointcloud_mapping->insertKeyFrame(intrinsic, extrinsic, img->image, depth->image, confidence->image);

	//Publishing global map	
	
	if(start_time==0) start_time = clock();
	end_time=clock();
	//500 ms
	if( (end_time-start_time)/1000 > 500 ){
	//if(1){
		sensor_msgs::PointCloud2 output_msg;
		pcl::toROSMsg( *(pointcloud_mapping->get_globalMap()), output_msg);
    	output_msg.header.frame_id = "map";
    	pointCloudpub.publish(output_msg);
		start_time = end_time;
	}
	
}


int main( int argc, char** argv )
{
	ros::init(argc, argv, "dense_mapping");
    ros::NodeHandle nh;
	double resolution,prob_threshold;
	int depthInfoQueueSize;
	
	nh.param<double>("resolution", resolution, 0.01);
	nh.param<double>("prob_threshold", prob_threshold, 0.94);
	nh.param<int>("depthInfoQueueSize", depthInfoQueueSize, 10000);

	pointcloud_mapping = new PointCloudMapping(resolution, prob_threshold);
    imgSub = nh.subscribe("depth_info", depthInfoQueueSize, &vidCb);
	
	pointCloudpub = nh.advertise<sensor_msgs::PointCloud2> ("global_map", 1);

    ros::spin();

	return 0;
}


