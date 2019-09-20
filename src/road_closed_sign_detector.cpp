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
        double LEAF_SIZE;
        double TOLERANCE;
        int MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE;
        double MAX_WIDTH, MAX_HEIGHT, MAX_DEPTH;
        double MIN_WIDTH, MIN_HEIGHT, MIN_DEPTH;
        double SAMPLE_RATIO;
        double ANGLE_THRESHOLD, CENTROID_THRESHOLD;
        double LOWER_NORMAL_THRESHOLD, UPPER_NORMAL_THRESHOLD;
        double DIST_ERROR_THRESHOLD;
        double PLANE_DIST_ERROR_THRESHOLD;
        double INTENSITY_RATIO;
    public:
        ClosedSignDetector();
        void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr&);
        Eigen::Vector4f calc_centroid(pcl::PointCloud<pcl::PointXYZI>::Ptr);
        Eigen::Vector3f calc_cluster_size(pcl::PointCloud<pcl::PointXYZI>::Ptr);
        double calc_dist(std::vector<Eigen::Vector4f>&);
        void clustering(pcl::PointCloud<pcl::PointXYZI>::Ptr, std::vector<pcl::PointIndices>&);
        void downsampling(pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr);
        void normal_estimation(pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::Normal>::Ptr);
        double pi_2_pi(double);
        bool pickup_cluster(pcl::PointCloud<pcl::PointXYZI>::Ptr, Eigen::Vector4f, Eigen::Vector3f);
        void plane_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr ,pcl::PointCloud<pcl::PointXYZI>::Ptr);
        void process(pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr);
        double square(double);
};

ClosedSignDetector::ClosedSignDetector()
    : private_nh("~")
{
    velodyne_sub = nh.subscribe("/velodyne_obstacles", 1, &ClosedSignDetector::velodyne_callback, this);
    cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud/closed_sign_cloud", 1);
    flag_pub = nh.advertise<std_msgs::Bool>("/recognition/closed_sign", 1);

    private_nh.param("LEAF_SIZE", LEAF_SIZE, 0.08);
    private_nh.param("TOLERANCE", TOLERANCE, 0.15);
    private_nh.param("MIN_CLUSTER_SIZE", MIN_CLUSTER_SIZE, 20);
    private_nh.param("MAX_CLUSTER_SIZE", MAX_CLUSTER_SIZE, 1200);
    private_nh.param("MAX_WIDTH", MAX_WIDTH, 0.5);
    private_nh.param("MAX_HEIGHT", MAX_HEIGHT, 0.7);
    private_nh.param("MAX_DEPTH", MAX_DEPTH, 0.45);
    private_nh.param("MIN_WIDTH", MIN_WIDTH, 0.2);
    private_nh.param("MIN_HEIGHT", MIN_HEIGHT, 0.6);
    private_nh.param("MIN_DEPTH", MIN_DEPTH, 0.1);
    private_nh.param("SAMPLE_RATIO", SAMPLE_RATIO, 0.5);
    private_nh.param("ANGLE_THRESHOLD", ANGLE_THRESHOLD, 0.5);
    private_nh.param("CENTROID_THRESHOLD", CENTROID_THRESHOLD, -1.0);
    private_nh.param("LOWER_NORMAL_THRESHOLD", LOWER_NORMAL_THRESHOLD, 0.20);
    private_nh.param("UPPER_NORMAL_THRESHOLD", UPPER_NORMAL_THRESHOLD, 0.30);
    private_nh.param("INTENSITY_RATIO", INTENSITY_RATIO, 60.0);
    private_nh.param("PLANE_DIST_ERROR_THRESHOLD", PLANE_DIST_ERROR_THRESHOLD, 0.005);
    private_nh.param("DIST_ERROR_THRESHOLD", DIST_ERROR_THRESHOLD, 0.5);
}

double ClosedSignDetector::square(double a)
{
    return a*a;
}

Eigen::Vector4f ClosedSignDetector::calc_centroid(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cluster, centroid);
    
    return centroid;
}

Eigen::Vector3f ClosedSignDetector::calc_cluster_size(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster)
{

    Eigen::Vector3f min_p;
    min_p[0] = cluster->points[0].x;
    min_p[1] = cluster->points[0].y;
    min_p[2] = cluster->points[0].z;

    Eigen::Vector3f max_p;
    max_p[0] = cluster->points[0].x;
    max_p[1] = cluster->points[0].y;
    max_p[2] = cluster->points[0].z;

    for(size_t i=0; i<cluster->points.size(); ++i){
        if(cluster->points[i].x < min_p[0]) min_p[0] = cluster->points[i].x;
        if(cluster->points[i].y < min_p[1]) min_p[1] = cluster->points[i].y;
        if(cluster->points[i].z < min_p[2]) min_p[2] = cluster->points[i].z;

        if(cluster->points[i].x > max_p[0]) max_p[0] = cluster->points[i].x;
        if(cluster->points[i].y > max_p[1]) max_p[1] = cluster->points[i].y;
        if(cluster->points[i].z > max_p[2]) max_p[2] = cluster->points[i].z;
    }

    float depth = max_p[0] - min_p[0];
    float width = max_p[1] - min_p[1];
    float height = max_p[2] - min_p[2];

    Eigen::Vector3f cluster_size = {width, height, depth};

    return cluster_size;
}

double ClosedSignDetector::calc_dist(std::vector<Eigen::Vector4f>& centroids)
{
    return sqrt(square(centroids[0][0] - centroids[1][0]) + square(centroids[0][1] - centroids[1][1]));
}

void ClosedSignDetector::clustering(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in, std::vector<pcl::PointIndices>& cluster_indices)
{
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud_in);
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(TOLERANCE);
    ec.setMinClusterSize(MIN_CLUSTER_SIZE);
    ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_in);
    ec.extract(cluster_indices);
} 

void ClosedSignDetector::downsampling(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr ds_cloud)
{
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(cloud_in);
    vg.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    vg.filter(*ds_cloud);
}

void ClosedSignDetector::normal_estimation(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud_in);
    pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
    ne.setInputCloud(cloud_in);
    ne.setKSearch(5);
    ne.setSearchMethod(tree);
    ne.compute(*normals);
}

double ClosedSignDetector::pi_2_pi(double angle)
{
    return atan2(sin(angle), cos(angle));
}

bool ClosedSignDetector::pickup_cluster(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster, Eigen::Vector4f centroid, Eigen::Vector3f size)
{
    double width = size[0];
    double height = size[1];
    double depth = size[2];

    if(MIN_WIDTH < width && width < MAX_WIDTH){
        if(MIN_HEIGHT < height && height < MAX_HEIGHT){
            if(MIN_DEPTH < depth && depth < MAX_DEPTH){
                double angle = atan2(centroid[1], centroid[0]);
                angle = pi_2_pi(angle);
                if(fabs(angle) < ANGLE_THRESHOLD && centroid[2] < CENTROID_THRESHOLD){

                    //std::cout << "width  :" << width << "  height  :" << height << "  depth  :" << depth << std::endl;
                    //std::cout << "centroid_z :" << centroid[2] << std::endl;
                    //std::cout << "cluster_angle :" << angle << std::endl;

                    //std::cout << "size : " << cluster->points.size() << std::endl;
                    return true;

                }
            }
        }
    }
    return false;
}

void ClosedSignDetector::plane_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr cluster, pcl::PointCloud<pcl::PointXYZI>::Ptr plane_cluster)
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZI> segmentation;
    segmentation.setInputCloud(cluster);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(PLANE_DIST_ERROR_THRESHOLD);
    segmentation.setOptimizeCoefficients(true);

    pcl::PointIndices inlierIndices;
    segmentation.segment(inlierIndices, *coefficients);

    if(inlierIndices.indices.size())
        pcl::copyPointCloud<pcl::PointXYZI>(*cluster, inlierIndices, *plane_cluster);
}

void ClosedSignDetector::process(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr stop_sign_cluster)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr ds_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    downsampling(cloud_in, ds_cloud);

    double avg_intensity = 0.0;
    std::vector<float> tmp_point_z;
    tmp_point_z.resize(ds_cloud->points.size());
    for(size_t i=0; i<ds_cloud->points.size(); ++i){
        tmp_point_z[i] = ds_cloud->points[i].z;
        ds_cloud->points[i].z = 0.0;
        avg_intensity += ds_cloud->points[i].intensity;
    }

    avg_intensity /= ds_cloud->points.size();
    
    std::vector<pcl::PointIndices> cluster_indices;
    clustering(ds_cloud, cluster_indices);
    
    for(size_t i=0; i<ds_cloud->points.size(); ++i)
        ds_cloud->points[i].z = tmp_point_z[i];

    //pcl::PointCloud<pcl::PointXYZI>::Ptr points_all (new pcl::PointCloud<pcl::PointXYZI>);
    std::vector<Eigen::Vector4f> centroids;
    for(std::vector<pcl::PointIndices>::const_iterator it=cluster_indices.begin(); it != cluster_indices.end(); ++it){
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);
        
        for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(ds_cloud->points[*pit]);

        Eigen::Vector3f cluster_size = calc_cluster_size(cloud_cluster);
        Eigen::Vector4f cluster_centroid = calc_centroid(cloud_cluster);

        if(pickup_cluster(cloud_cluster, cluster_centroid, cluster_size)){
            pcl::PointCloud<pcl::PointXYZI>::Ptr plane_cluster(new pcl::PointCloud<pcl::PointXYZI>);
            plane_filter(cloud_cluster, plane_cluster);
            if(plane_cluster->points.size() < 10)
                continue;

            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            normal_estimation(plane_cluster, normals);
            double avg_normal_x = 0;
            double avg_normal_y = 0;
            double avg_normal_z = 0;
            int point_num = normals->points.size();
            for(int i=0; i<point_num; ++i){
                avg_normal_x += normals->points[i].normal_x;
                avg_normal_y += normals->points[i].normal_y;
                avg_normal_z += normals->points[i].normal_z;
                //std::cout << "n_x: " << normals->points[i].normal_x << " n_y: " << normals->points[i].normal_y << " n_z :" << normals->points[i].normal_z << std::endl;
            }
            avg_normal_x /= point_num;
            avg_normal_y /= point_num;
            avg_normal_z /= point_num;

            if(avg_normal_z > UPPER_NORMAL_THRESHOLD || avg_normal_z < LOWER_NORMAL_THRESHOLD)
                continue;

            //std::cout << "p_width  :" << plane_size[0] << "  p_height  :" << plane_size[1] << "  p_depth  :" << plane_size[2] << std::endl;
            //std::cout << "default size :" << cloud_cluster->points.size() << " p size :" << plane_cluster->points.size() << std::endl;
            //std::cout << "avg_x: " << avg_normal_x << " avg_y: " << avg_normal_y << " avg_z: " << avg_normal_z << std::endl;
            //std::cout << "===============================================" << std::endl;
            
            std::sort(plane_cluster->points.begin(), plane_cluster->points.end(), [](const pcl::PointXYZI &a, const pcl::PointXYZI &b) {
                return a.intensity > b.intensity;
            });

            double sample_avg_intensity = 0.0;
            int sample_point_num = (int)(plane_cluster->points.size() * SAMPLE_RATIO + 1);
            for(size_t i=0; i<sample_point_num; ++i){
                sample_avg_intensity += plane_cluster->points[i].intensity;
            }
            sample_avg_intensity /= sample_point_num;

            double intensity_ratio = sample_avg_intensity / avg_intensity;
            if(intensity_ratio  < INTENSITY_RATIO)
                continue;

            stop_sign_cluster->points.insert(stop_sign_cluster->points.begin(), plane_cluster->points.begin(), plane_cluster->points.end());
            centroids.push_back(cluster_centroid);
            std::cout << "-----------------------------------------------------------" << std::endl;
            std::cout << "width  :" << cluster_size[0] << "  height  :" << cluster_size[1] << "  depth  :" << cluster_size[2] << std::endl;
            std::cout << "centroid :" << cluster_centroid[2] << std::endl;
            std::cout << "avg_x: " << avg_normal_x << " avg_y: " << avg_normal_y << " avg_z: " << avg_normal_z << std::endl;
            std::cout << "point num :" << point_num << " sample num :" << sample_point_num << " intensity_ratio :" << intensity_ratio << std::endl;
            std::cout << "avg_intensity :" << avg_intensity << " sample_avg_intensity :" << sample_avg_intensity << std::endl;
            for(auto point : plane_cluster->points)
                std::cout << "intensity :" << point.intensity << std::endl;
        }
        //points_all->points.insert(points_all->points.begin(), cloud_cluster->points.begin(), cloud_cluster->points.end());
    }
    if(centroids.size() == 2){
        double cluster_dist = calc_dist(centroids);
        if(cluster_dist > (2.0 -DIST_ERROR_THRESHOLD) && cluster_dist < (5.0 + DIST_ERROR_THRESHOLD))
            closed_flag.data = true;
    }
/*
    sensor_msgs::PointCloud2 cloud_ros;
    pcl::toROSMsg(*points_all, cloud_ros);
    cloud_ros.header.frame_id = "velodyne";
    cloud_ros.header.stamp = ros::Time::now();
    cloud_pub.publish(cloud_ros);
*/   

}

void ClosedSignDetector::velodyne_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr stop_sign_cluster (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cloud_in);
    closed_flag.data = false;

    if(cloud_in->points.size())
        process(cloud_in, stop_sign_cluster);

    sensor_msgs::PointCloud2 cloud_ros;
    pcl::toROSMsg(*stop_sign_cluster, cloud_ros);
    cloud_ros.header.frame_id = msg->header.frame_id;
    cloud_ros.header.stamp = ros::Time::now();
    cloud_pub.publish(cloud_ros);
    flag_pub.publish(closed_flag);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "closed_sign_detector");
    
    ClosedSignDetector detector;
    ros::spin();

    return 0;
}
