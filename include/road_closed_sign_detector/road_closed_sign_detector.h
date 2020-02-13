#ifndef __ROAD_CLOSED_SIGN_DETECTOR_H
#define __ROAD_CLOSED_SIGN_DETECTOR_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Core>

class ClosedSignDetector{
    private:
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;
        ros::Subscriber velodyne_sub;
        ros::Publisher cloud_pub;
        ros::Publisher flag_pub;

        std_msgs::Bool closed_flag;
        bool USE_RANSAC, USE_NORMAL, USE_CURVATURE;
        double LEAF_SIZE;
        double TOLERANCE;
        int MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE, PLANE_CLUSTER_SIZE;
        double MAX_WIDTH, MAX_HEIGHT, MAX_DEPTH;
        double MIN_WIDTH, MIN_HEIGHT, MIN_DEPTH;
        double SAMPLE_RATIO;
        double ANGLE_THRESHOLD, CENTROID_THRESHOLD;
        double LOWER_NORMAL_THRESHOLD, UPPER_NORMAL_THRESHOLD;
        double DIST_ERROR_THRESHOLD;
        double PLANE_DIST_ERROR_THRESHOLD;
        double INTENSITY_RATIO;
        double CURVATURE_THRESHOLD;
    public:
        ClosedSignDetector();
        void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr&);
        Eigen::Vector4f calc_centroid(pcl::PointCloud<pcl::PointXYZI>::Ptr&);
        Eigen::Vector3f calc_cluster_size(pcl::PointCloud<pcl::PointXYZI>::Ptr&);
        double calc_dist(std::vector<Eigen::Vector4f>&);
        double calc_intensity_ratio(pcl::PointCloud<pcl::PointXYZI>::Ptr&, double);
        void clustering(pcl::PointCloud<pcl::PointXYZI>::Ptr&, std::vector<pcl::PointIndices>&);
        void downsampling(pcl::PointCloud<pcl::PointXYZI>::Ptr&, pcl::PointCloud<pcl::PointXYZI>::Ptr&);
        void normal_estimation(pcl::PointCloud<pcl::PointXYZI>::Ptr&, pcl::PointCloud<pcl::Normal>::Ptr&);
        bool is_valid_cluster_size(pcl::PointCloud<pcl::PointXYZI>::Ptr&, Eigen::Vector4f, Eigen::Vector3f);
        void plane_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr&, pcl::PointCloud<pcl::PointXYZI>::Ptr&);
        void process(pcl::PointCloud<pcl::PointXYZI>::Ptr&, pcl::PointCloud<pcl::PointXYZI>::Ptr&);
};

#endif //ROAD_CLOSED_SIGN_DETECTOR_H
