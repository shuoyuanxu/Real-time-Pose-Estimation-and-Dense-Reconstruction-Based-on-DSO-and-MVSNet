#include "pointcloudmapping.h"

// Constructor
PointCloudMapping::PointCloudMapping(double resolution_, float prob_threshold_)
{
    set_resolution(resolution_);
    this->prob_threshold = prob_threshold_;
    // Shared pointer to initialise globalmap
    globalMap = boost::make_shared< PointCloudT >();
    // Shared pointer to initialise visulisation thread, bind the pointer 'this' to update_globalMap
    viewerThread = std::make_shared<std::thread>( std::bind(&PointCloudMapping::update_globalMap, this ) );
}

void PointCloudMapping::set_resolution(double resolution_){
    resolution = resolution_;
    voxel.setLeafSize( resolution_, resolution_, resolution_);
}

// Destructor
void PointCloudMapping::shutdown()
{
    {   // Ensure mutex works within '{}'
        std::unique_lock<std::mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

PointCloudMapping::PointCloudT::Ptr PointCloudMapping::get_globalMap()
{
    return globalMap;
}

// Insert the data of a frame
void PointCloudMapping::insertKeyFrame(cv::Mat&  intrinsic, cv::Mat& extrinsic, cv::Mat& color, cv::Mat& depth, cv::Mat& confidence)
{
    std::unique_lock<std::mutex> lck(keyframeMutex); // get the mutex

    intrinsics.push_back( intrinsic.clone() );
    extrinsics.push_back( extrinsic.clone() );
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );
    confidenceImgs.push_back( confidence.clone() );

    keyFrameUpdated.notify_one(); // once pushback is done, notify the waitting thread
}

// generate point cloud
pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::generatePointCloud(cv::Mat& intrinsics, cv::Mat& extrinsics, cv::Mat& color, cv::Mat& depth, cv::Mat& confidence)
{   // temporary point cloud
    PointCloudT::Ptr tmp_pc( new PointCloudT() );

    // point cloud is null ptr
    double fx = intrinsics.at<double>(0,0);
    double fy = intrinsics.at<double>(1,1);
    double cx = intrinsics.at<double>(0,2);
    double cy = intrinsics.at<double>(1,2);
    
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {   // Using probability to remove outliers
            if(confidence.ptr<float>(m)[n] < prob_threshold){
                continue;
            }

            float d = depth.ptr<float>(m)[n];
            if( d < 0 || d > 15 ) continue;
            PointT p;
            p.z = d;
            p.x = ( n - cx) * p.z / fx;
            p.y = ( m - cy) * p.z / fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

            tmp_pc->points.push_back(p);
        }
    }
    // Assign extrinsics to Eigen Matrix
    static Eigen::Matrix4f T_delta;

    Eigen::Matrix4f T;
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            T(i,j) = extrinsics.at<double>(i,j);
    // Define a output point cloud
    PointCloudT::Ptr output( new PointCloudT() );

    PointCloudT::Ptr cloud(new PointCloudT);
    pcl::transformPointCloud( *tmp_pc, *cloud, T.matrix());
    cloud->is_dense = false;
    //Adaptive Voxel Filtering
    double resolution_tmp = cv::mean(depth).val[0]/ ((fx<fy?fx:fy) * 20);
    set_resolution( resolution_tmp < resolution ? resolution_tmp : resolution);
    
    cout << "generate point cloud for kf size=" << cloud->points.size() << endl;
    return cloud;
}

// Update global map. running on a independent thread
void PointCloudMapping::update_globalMap()
{   
    // Using shared pointer to generate a visulisation object
    //pcl::visualization::CloudViewer viewer("viewer");
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("viewer"));
    viewer->setBackgroundColor(0,0,0);
    // Basic handlings when using PCL visulisation
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(globalMap);
    viewer->addPointCloud<pcl::PointXYZRGBA> (globalMap, rgb, "globalMap");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "globalMap");

    while(1)
    {
        {
            std::unique_lock<std::mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            std::unique_lock<std::mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            // only running when waited variable arrived
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }

        // keyframe is updated
        size_t N=0;
        {
            std::unique_lock<std::mutex> lck( keyframeMutex );
            N = depthImgs.size();
        }
        // only loop through the newly add keyframes
        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {
            PointCloudT::Ptr p = generatePointCloud( intrinsics[i], extrinsics[i], colorImgs[i], depthImgs[i], confidenceImgs[i]);
            // Adding the generated point cloud to the globalMap
            *globalMap += *p;
        }
        
        PointCloudT::Ptr tmp(new PointCloudT());
        voxel.setInputCloud( globalMap );
        voxel.filter( *tmp );
        // swap the filtered point cloud back to global map
        globalMap->swap( *tmp );

        
        viewer->updatePointCloud(globalMap, "globalMap");
        viewer->spinOnce(0.3);
        //viewer.showCloud( globalMap );

        cout << "show global map, size=" << globalMap->points.size() << endl;
        
        lastKeyframeSize = N;
    }
    
}

