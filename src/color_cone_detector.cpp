#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <pcl/kdtree/kdtree.h> 
#include <pcl/search/kdtree.h> 
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/pca.h>
#include <Eigen/Core>

class ColorConeDetector{
    private:
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;

        ros::Subscriber velodyne_sub;
        ros::Publisher cloud_pub;
        ros::Publisher flag_pub;

        double LEAF_SIZE;
        double TOLERANCE;
        int MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE;
        double MAX_WIDTH, MAX_HEIGHT, MAX_DEPTH;
        double MIN_WIDTH, MIN_HEIGHT, MIN_DEPTH;
        double ANGLE_THRESHOLD, CENTROID_THRESHOLD;
        double DIST2, DIST3, DIST4;
        double DIST_ERROR_THRESHOLD;
    public:
        ColorConeDetector();
        double pi_2_pi(double angle);
        double square(double a);
        void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
        Eigen::Vector4f calc_centroid(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster);
        Eigen::Vector3f calc_cluster_size(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster);
        void clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cone_cluster);

        Eigen::Matrix3f pca(pcl::PointCloud<pcl::PointXYZ>::Ptr clouds);
        bool judge_stop_sign(std::vector<Eigen::Vector4f> centroids);
        bool pickup_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, Eigen::Vector4f centroid, Eigen::Vector3f size);

};

ColorConeDetector::ColorConeDetector()
    : private_nh("~")
{
    velodyne_sub = nh.subscribe("/velodyne_obstacles", 1, &ColorConeDetector::velodyne_callback, this);
    cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud/color_cone", 1);
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
    private_nh.param("ANGLE_THRESHOLD", ANGLE_THRESHOLD, 0.5);
    private_nh.param("CENTROID_THRESHOLD", CENTROID_THRESHOLD, -1.0);
    private_nh.param("DIST2", DIST2, 1.9);
    private_nh.param("DIST3", DIST3, 1.4);
    private_nh.param("DIST4", DIST4, 1.0);
    private_nh.param("DIST_ERROR_THRESHOLD", DIST_ERROR_THRESHOLD, 0.3);
}

double ColorConeDetector::pi_2_pi(double angle)
{
    return atan2(sin(angle), cos(angle));
}

double ColorConeDetector::square(double a)
{
    return a*a;
}

Eigen::Vector4f ColorConeDetector::calc_centroid(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cluster, centroid);
    
    return centroid;
}
Eigen::Vector3f ColorConeDetector::calc_cluster_size(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
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

void ColorConeDetector::clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cone_cluster)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud_in);
    vg.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    vg.filter(*ds_cloud);

    std::vector<float> tmp_point_z;
    tmp_point_z.resize(ds_cloud->points.size());
    for(size_t i=0; i<ds_cloud->points.size(); ++i){
        tmp_point_z[i] = ds_cloud->points[i].z;
        ds_cloud->points[i].z = 0.0;
    }
    
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(ds_cloud);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(TOLERANCE);
    ec.setMinClusterSize(MIN_CLUSTER_SIZE);
    ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
    ec.setSearchMethod(tree);
    ec.setInputCloud(ds_cloud);
    ec.extract(cluster_indices);
    
    for(size_t i=0; i<ds_cloud->points.size(); ++i)
        ds_cloud->points[i].z = tmp_point_z[i];

    std::vector<Eigen::Vector4f> centroids;
    for(std::vector<pcl::PointIndices>::const_iterator it=cluster_indices.begin(); it != cluster_indices.end(); ++it){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        
        for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            cloud_cluster->points.push_back(ds_cloud->points[*pit]);
        
        Eigen::Vector3f cluster_size = calc_cluster_size(cloud_cluster);
        Eigen::Vector4f cluster_centroid = calc_centroid(cloud_cluster);

        if(pickup_cluster(cloud_cluster, cluster_centroid, cluster_size)){
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation;
            normalEstimation.setInputCloud(cloud_cluster);
            normalEstimation.setKSearch(10);
            normalEstimation.setSearchMethod(tree);
            normalEstimation.compute(*normals);
            double avg_normal_x = 0;
            double avg_normal_y = 0;
            double avg_normal_z = 0;
            int n_size = normals->points.size();
            for(int i=0; i<n_size; ++i){
                avg_normal_x += normals->points[i].normal_x;
                avg_normal_y += normals->points[i].normal_y;
                avg_normal_z += normals->points[i].normal_z;
                //std::cout << "n_x: " << normals->points[i].normal_x << " n_y: " << normals->points[i].normal_y << " n_z :" << normals->points[i].normal_z << std::endl;

            }
            avg_normal_x /= n_size;
            avg_normal_y /= n_size;
            avg_normal_z /= n_size;
            //std::cout << "avg_x: " << avg_normal_x << " avg_y: " << avg_normal_y << " avg_z: " << avg_normal_z << std::endl;

            if(avg_normal_z > 0.1){
                cone_cluster->points.insert(cone_cluster->points.begin(), cloud_cluster->points.begin(), cloud_cluster->points.end());
                centroids.push_back(cluster_centroid);
            }
        }
    }

    if(judge_stop_sign(centroids)){
        std::cout << "-----closed point-----" << std::endl;
        std_msgs::Bool flag;
        flag.data = true;
        flag_pub.publish(flag);
    }
}

Eigen::Matrix3f ColorConeDetector::pca(pcl::PointCloud<pcl::PointXYZ>::Ptr clouds)
{
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(clouds);
    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
    Eigen::Vector3f eigen_values = pca.getEigenValues(); 
    //std::cout << "eigen_values\n" << eigen_values << std::endl;
    //std::cout << "eigen_vector\n" << eigen_vectors << std::endl;

    return eigen_vectors;
}

bool ColorConeDetector::pickup_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cluster, Eigen::Vector4f centroid, Eigen::Vector3f size)
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
                    //Eigen::Matrix3f eigen_vectors = pca(cluster);
                    return true;

                }
            }
        }
    }
    return false;
}

bool ColorConeDetector::judge_stop_sign(std::vector<Eigen::Vector4f> centroids)
{
    std::vector<double> target_dist = {DIST2, DIST3, DIST4}; 
    std::vector<double> distances;
    int color_cone_num = centroids.size();
    if(1 < color_cone_num && color_cone_num < 4){
        //std::cout << "color_cone_num :" << color_cone_num << std::endl;
        for(int i=0; i<color_cone_num; ++i){
            for(int j=i+1; j<color_cone_num; ++j){
                double dist = sqrt(square(centroids[i][0] - centroids[j][0]) + square(centroids[i][1] - centroids[j][1]));
                //std::cout << "dist" << i*(color_cone_num-1) + j - (i+1)<< " :" << dist << std::endl;
                distances.push_back(dist);
            }
        }
        std::sort(distances.begin(), distances.end());
        for(int i=0; i<color_cone_num-1; ++i){
            //std::cout << "dist_error :" << distances[i] - target_dist[color_cone_num-2] << std::endl;
            if(fabs(distances[i] - target_dist[color_cone_num-2]) > DIST_ERROR_THRESHOLD)
                return false;
        }

        return true;
    }

    return false;
}

void ColorConeDetector::velodyne_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cone_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud_in);

    if(0 < cloud_in->points.size())
        clustering(cloud_in, cone_cluster);

    sensor_msgs::PointCloud2 cloud_ros;
    pcl::toROSMsg(*cone_cluster, cloud_ros);
    cloud_ros.header.frame_id = msg->header.frame_id;
    cloud_ros.header.stamp = ros::Time::now();
    cloud_pub.publish(cloud_ros);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "road_closed_sign_detector");
    
    ColorConeDetector detector;
    ros::spin();

    return 0;
}
