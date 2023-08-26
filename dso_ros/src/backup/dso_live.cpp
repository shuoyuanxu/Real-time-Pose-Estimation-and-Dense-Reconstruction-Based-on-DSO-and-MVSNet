/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/





#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fstream>

//opencv&Eigen
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>S
#include <Eigen/Dense>

//DSO headers
#include "util/settings.h"
#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

//ROS headers
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include "cv_bridge/cv_bridge.h"

//Defined msg type
#include "dso_ros/SlidingWindowsMsg.h"
#include "dso_ros/SE3Msg.h"

//Path of certain values
std::string calib = "";
std::string vignetteFile = "";
std::string gammaFile = "";
bool useSampleOutput=false;

using namespace dso;

//Parsing command-line arguments, not used in this version
void parseArgument(char* arg)
{
	int option;
	char buf[1000];

	if(1==sscanf(arg,"sampleoutput=%d",&option))
	{
		if(option==1)
		{
			useSampleOutput = true;
			printf("USING SAMPLE OUTPUT WRAPPER!\n");
		}
		return;
	}

	if(1==sscanf(arg,"quiet=%d",&option))
	{
		if(option==1)
		{
			setting_debugout_runquiet = true;
			printf("QUIET MODE, I'll shut up!\n");
		}
		return;
	}


	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}
	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignetteFile = buf;
		printf("loading vignette from %s!\n", vignetteFile.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaFile = buf;
		printf("loading gammaCalib from %s!\n", gammaFile.c_str());
		return;
	}

	printf("could not parse argument \"%s\"!!\n", arg);
}


FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
ros::Subscriber imgSub;
ros::Publisher slidingWindowsPub;
ros::Publisher SE3Pub;
int frameID = 0;

void imgCb(const sensor_msgs::ImageConstPtr img)
{
	// ROS img to cv img
	// grayscale img
	static int count = 0;
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
	// make sure grayscale and 1 channel
	assert(cv_ptr->image.type() == CV_8U);
	assert(cv_ptr->image.channels() == 1);
	// bgr img
	cv_bridge::CvImagePtr cvbgr_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
	// make sure bgr and 3 channel
	assert(cvbgr_ptr->image.type() == CV_8UC3);
	assert(cvbgr_ptr->image.channels() == 3);

	//auto reset when lost
	if(setting_fullResetRequested || fullSystem->isLost)
	{
		std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
		delete fullSystem;
		for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
		fullSystem = new FullSystem();
		fullSystem->linearizeOperation=false;
		fullSystem->outputWrapper = wraps;
	    if(undistorter->photometricUndist != 0)
	    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
		setting_fullResetRequested=false;
		frameID = 0;
		count = 0;
	}

	// cv img to dso img
	MinimalImageB minImg((int)cv_ptr->image.cols, (int)cv_ptr->image.rows,(unsigned char*)cv_ptr->image.data);
	// undistort img to ImageAndExposure type
	ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1,0, 1.0f);

	//bgr img undistort
	cv::Mat img_bgr;
	auto cameraparams = undistorter->getOriginalParameter();
	cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << cameraparams[0], 0, cameraparams[2], \
														0, cameraparams[1], cameraparams[3], \
														0, 0, 1);
	// <<: OpenCV defined symbol, inserting elements into the Mat 1by1
	cv::Mat distCoeffs = (cv::Mat_<double>(4,1) << cameraparams[4], cameraparams[5], cameraparams[6], cameraparams[7]);
	cv::Mat newCameraMatrix;
	cv::eigen2cv(undistorter->getK(), newCameraMatrix);
	cv::undistort(cvbgr_ptr->image,	img_bgr, cameraMatrix, distCoeffs, newCameraMatrix);

	//running VO
	fullSystem->addActiveFrame(undistImg, img_bgr, frameID);
	frameID++;
	delete undistImg;

	//pub marged Frame topic
	dso_ros::SE3Msg SE3;
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			SE3.camToWorld[4*i+j] = fullSystem->get_curSE3().matrix()(i,j);
	SE3Pub.publish(SE3);

	if(fullSystem->slidingWindows_Frames.size() != 0){
		std::cout << "fullSystem->slidingWindows_Frames.size():" << fullSystem->slidingWindows_Frames.size() << std::endl;
		// sequence of sliding window frames
		auto sw = fullSystem->slidingWindows_Frames.front();
		// poping each frames of the sequence
		fullSystem->slidingWindows_Frames.pop_front();
		
		for(int n=0; n<setting_maxFrames; n++){
			dso_ros::SlidingWindowsMsg msg;
			for(int i=0; i<4; i++)
				for(int j=0; j<4; j++)
					msg.camToWorld[4*i+j] = sw.at(n).camToWorld.matrix()(i,j);
			msg.image = *(cv_bridge::CvImage(std_msgs::Header(), "bgr8", sw.at(n).image).toImageMsg());
			msg.Intrinsics[0] = newCameraMatrix.at<double>(0,0);//fx
			msg.Intrinsics[1] = newCameraMatrix.at<double>(1,1);//fy
			msg.Intrinsics[2] = newCameraMatrix.at<double>(0,2);//cx
			msg.Intrinsics[3] = newCameraMatrix.at<double>(1,2);//cy

			msg.msg_id = count;
			msg.window_size = setting_maxFrames;
			slidingWindowsPub.publish(msg);
		}
		count++;
	}
	
}


int main( int argc, char** argv )
{
	// Initialise ros node
	ros::init(argc, argv, "dso_live");
	ros::NodeHandle nh;

	// parse the input parameters
	for(int i=1; i<argc;i++) parseArgument(argv[i]);

	// system parameter
	nh.param<float>("setting_desiredImmatureDensity", setting_desiredImmatureDensity, 800);
	nh.param<float>("setting_desiredPointDensity", setting_desiredPointDensity, 1000);
	nh.param<int>("setting_minFrames", setting_minFrames, 5);
	nh.param<int>("setting_maxFrames", setting_maxFrames, 7);
	nh.param<int>("setting_maxOptIterations", setting_maxOptIterations, 10);
	nh.param<int>("setting_minOptIterations", setting_minOptIterations, 5);
	nh.param<bool>("setting_logStuff", setting_logStuff, false);
	nh.param<float>("setting_kfGlobalWeight", setting_kfGlobalWeight, 1);
	nh.param<bool>("disableDisplay_DSO", disableAllDisplay, false);

	printf("MODE WITH CALIBRATION, but without exposure times!\n");
	setting_photometricCalibration = 2;
	setting_affineOptModeA = 0;
	setting_affineOptModeB = 0;

	// Defining and initialising of undistortion class
    undistorter = Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile);

    setGlobalCalib(
            (int)undistorter->getSize()[0],
            (int)undistorter->getSize()[1],
            undistorter->getK().cast<float>());

	// create new dso system (FullSystem)
    fullSystem = new FullSystem();
    fullSystem->linearizeOperation=false;

	
    if(!disableAllDisplay)
	    fullSystem->outputWrapper.push_back(new IOWrap::PangolinDSOViewer(
	    		 (int)undistorter->getSize()[0],
	    		 (int)undistorter->getSize()[1]));
	

    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());


    if(undistorter->photometricUndist != 0)
    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
    
    
	std::string img_topic;
	int slidingWindowsQueueSize;
	// Image input path
	nh.param<std::string>("img_topic", img_topic, "/usb_cam/image_raw");
	nh.param<int>("slidingWindowsQueueSize", slidingWindowsQueueSize, 10000);

	// Subscribe to image topic as input, calling &vidCb whenever a topic comes in
	imgSub = nh.subscribe(img_topic, 5, &imgCb);
	// Publish SlidingWindowsMsg and SE3Msg message to topic "SlidingWindows" and "curSE3"
	slidingWindowsPub = nh.advertise<dso_ros::SlidingWindowsMsg>("SlidingWindows", slidingWindowsQueueSize);
	SE3Pub = nh.advertise<dso_ros::SE3Msg>("curSE3", 10);
	// Loop until the ros node is closed, normally used when there are multiple callback functions in a node
    ros::spin();

	
    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
        delete ow;
    }
	
    delete undistorter;
    delete fullSystem;

	return 0;
}


