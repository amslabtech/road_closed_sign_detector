#include "road_closed_sign_detector/road_closed_sign_detector.h"

ClosedSignDetector::ClosedSignDetector()
    : private_nh("~")
{
    velodyne_sub = nh.subscribe("/velodyne_obstacles", 1, &ClosedSignDetector::velodyne_callback, this);
    cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/cloud/closed_sign_cloud", 1);
    flag_pub = nh.advertise<std_msgs::Bool>("/recognition/closed_sign", 1);

    private_nh.param("USE_RANSAC", USE_RANSAC, true);
    private_nh.param("USE_NORMAL", USE_NORMAL, true);
    private_nh.param("USE_CURVATURE", USE_CURVATURE, true);
    private_nh.param("LEAF_SIZE", LEAF_SIZE, 0.08);
    private_nh.param("TOLERANCE", TOLERANCE, 0.15);
    private_nh.param("MIN_CLUSTER_SIZE", MIN_CLUSTER_SIZE, 20);
    private_nh.param("MAX_CLUSTER_SIZE", MAX_CLUSTER_SIZE, 1200);
    private_nh.param("PLANE_CLUSTER_SIZE", PLANE_CLUSTER_SIZE, 15);
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
    private_nh.param("CURVATURE_THRESHOLD", CURVATURE_THRESHOLD, 0.02);
    private_nh.param("PLANE_DIST_ERROR_THRESHOLD", PLANE_DIST_ERROR_THRESHOLD, 0.005);
    private_nh.param("DIST_ERROR_THRESHOLD", DIST_ERROR_THRESHOLD, 0.5);
}

inline double Pow(double x){
    return x*x;
}

Eigen::Vector4f ClosedSignDetector::calc_centroid(pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cluster, centroid);
    return centroid;
}

Eigen::Vector3f ClosedSignDetector::calc_cluster_size(pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster)
{
    Eigen::Vector3f min_p = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector3f max_p = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
    for(size_t i=0; i<cluster->points.size(); ++i){
        min_p[0] = std::min(min_p[0], cluster->points[i].x);
        min_p[1] = std::min(min_p[1], cluster->points[i].y);
        min_p[2] = std::min(min_p[2], cluster->points[i].z);
        max_p[0] = std::max(max_p[0], cluster->points[i].x);
        max_p[1] = std::max(max_p[1], cluster->points[i].y);
        max_p[2] = std::max(max_p[2], cluster->points[i].z);
    }
    float depth = max_p[0] - min_p[0];
    float width = max_p[1] - min_p[1];
    float height = max_p[2] - min_p[2];
    Eigen::Vector3f cluster_size = {width, height, depth};
    return cluster_size;
}

double ClosedSignDetector::calc_dist(std::vector<Eigen::Vector4f>& centroids)
{
    return sqrt(Pow(centroids[0][0] - centroids[1][0]) + Pow(centroids[0][1] - centroids[1][1]));
}

double ClosedSignDetector::calc_intensity_ratio(pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_points, double avg_intensity)
{
    double sample_avg_intensity = 0.0;
    int sample_point_num = cluster_points->points.size() * SAMPLE_RATIO + 1;
    for(int i=0; i<sample_point_num; ++i){
        sample_avg_intensity += cluster_points->points[i].intensity;
    }
    sample_avg_intensity /= sample_point_num;
    return sample_avg_intensity / avg_intensity;
}

void ClosedSignDetector::clustering(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in, std::vector<pcl::PointIndices>& cluster_indices)
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

void ClosedSignDetector::downsampling(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& ds_cloud)
{
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(cloud_in);
    vg.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    vg.filter(*ds_cloud);
}

void ClosedSignDetector::normal_estimation(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in, pcl::PointCloud<pcl::Normal>::Ptr& normals)
{
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud_in);
    pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
    ne.setInputCloud(cloud_in);
    ne.setKSearch(5);
    ne.setSearchMethod(tree);
    ne.compute(*normals);
}

bool ClosedSignDetector::is_valid_cluster_size(pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster, Eigen::Vector4f centroid, Eigen::Vector3f size)
{
    double width = size[0];
    double height = size[1];
    double depth = size[2];
    std::cout << "size : " << cluster->points.size() << std::endl;
    if(MIN_WIDTH < width && width < MAX_WIDTH){
        if(MIN_HEIGHT < height && height < MAX_HEIGHT){
            if(MIN_DEPTH < depth && depth < MAX_DEPTH){
                double angle = atan2(centroid[1], centroid[0]);
                if(fabs(angle) < ANGLE_THRESHOLD && centroid[2] < CENTROID_THRESHOLD){
                    //std::cout << "width  :" << width << "  height  :" << height << "  depth  :" << depth << std::endl;
                    //std::cout << "centroid_z :" << centroid[2] << std::endl;
                    //std::cout << "cluster_angle :" << angle << std::endl;
                    //std::cout << "size : " << cluster->points.size() << std::endl;
                    std::cout << "true" << std::endl;
                    return true;
                }
            }
        }
    }
    std::cout << "false" << std::endl;
    return false;
}

void ClosedSignDetector::plane_filter(pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster, pcl::PointCloud<pcl::PointXYZI>::Ptr& plane_cluster)
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
    if(inlierIndices.indices.size()){
        pcl::copyPointCloud<pcl::PointXYZI>(*cluster, inlierIndices, *plane_cluster);
    }
}

void ClosedSignDetector::process(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& stop_sign_cluster)
{
    double start = ros::Time::now().toSec();

    pcl::PointCloud<pcl::PointXYZI>::Ptr ds_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    downsampling(cloud_in, ds_cloud);
    double avg_intensity = 0.0;
    std::vector<float> tmp_point_z;
    size_t ds_point_num = ds_cloud->points.size();
    for(size_t i=0; i<ds_point_num; ++i){
        tmp_point_z.emplace_back(ds_cloud->points[i].z);
        ds_cloud->points[i].z = 0.0;
        avg_intensity += ds_cloud->points[i].intensity;
    }
    avg_intensity /= ds_point_num;
    std::vector<pcl::PointIndices> cluster_indices;
    clustering(ds_cloud, cluster_indices);
    for(size_t i=0; i<ds_cloud->points.size(); ++i){
        ds_cloud->points[i].z = tmp_point_z[i];
    }
    std::vector<Eigen::Vector4f> centroids;
    for(const auto& cidx : cluster_indices){
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZI>);
        for(const auto& pidx : cidx.indices){
            cloud_cluster->points.emplace_back(ds_cloud->points[pidx]);
        }
        Eigen::Vector3f cluster_size = calc_cluster_size(cloud_cluster);
        Eigen::Vector4f cluster_centroid = calc_centroid(cloud_cluster);
        if(is_valid_cluster_size(cloud_cluster, cluster_centroid, cluster_size)){
            pcl::PointCloud<pcl::PointXYZI>::Ptr plane_cluster(new pcl::PointCloud<pcl::PointXYZI>);
            if(USE_RANSAC){
                plane_filter(cloud_cluster, plane_cluster);
            }else{
                plane_cluster = cloud_cluster;
            }
            size_t plane_point_num = plane_cluster->points.size();
            std::cout << "ransac point num :" << plane_point_num << std::endl;
            if(plane_point_num < PLANE_CLUSTER_SIZE){
                continue;
            }
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            normal_estimation(plane_cluster, normals);
            Eigen::Vector3f avg_normal = Eigen::Vector3f::Constant(0.0);
            double curvature = 0;
            for(size_t i=0; i<plane_point_num; ++i){
                avg_normal[0] += normals->points[i].normal_x;
                avg_normal[1] += normals->points[i].normal_y;
                avg_normal[2] += normals->points[i].normal_z;
                curvature += normals->points[i].curvature;
            }
            avg_normal /= plane_point_num;
            curvature /= plane_point_num;
            std::cout << "avg curvature :" << curvature << std::endl;
            std::cout << "anx :" << avg_normal[0] << " any :" << avg_normal[1]<< " anz :" << avg_normal[2]<< std::endl;
            if(USE_NORMAL){
                if(avg_normal[2] > UPPER_NORMAL_THRESHOLD || avg_normal[2] < LOWER_NORMAL_THRESHOLD){
                    continue;
                }
            }
            if(USE_CURVATURE){
                if(curvature > CURVATURE_THRESHOLD){
                    continue;
                }
            }
            std::sort(plane_cluster->points.begin(), plane_cluster->points.end(), [](const pcl::PointXYZI &a, const pcl::PointXYZI &b) {
                return a.intensity > b.intensity;
            });
            double intensity_ratio = calc_intensity_ratio(plane_cluster, avg_intensity);
            if(intensity_ratio < INTENSITY_RATIO){
                continue;
            }
            *stop_sign_cluster += *plane_cluster;
            centroids.emplace_back(cluster_centroid);
            // std::cout << "-----------------------------------------------------------" << std::endl;
            // std::cout << "width  :" << cluster_size[0] << "  height  :" << cluster_size[1] << "  depth  :" << cluster_size[2] << std::endl;
            // std::cout << "centroid :" << cluster_centroid[2] << std::endl;
            // std::cout << "avg_x: " << avg_normal[0] << " avg_y: " << avg_normal[1] << " avg_z: " << avg_normal[2] << std::endl;
            // for(auto point : plane_cluster->points)
            //     std::cout << "intensity :" << point.intensity << std::endl;
        }
    }
    if(centroids.size() == 2){
        double cluster_dist = calc_dist(centroids);
        std::cout << "\033[31mcluster dist :" << cluster_dist << "\033[m" << std::endl;
        if(cluster_dist > (2.0 -DIST_ERROR_THRESHOLD) && cluster_dist < (5.0 + DIST_ERROR_THRESHOLD)){
            closed_flag.data = true;
            flag_pub.publish(closed_flag);
            std::cout << "\033[33m----------------------- closed sign ----------------------- \033[m" << std::endl;
        }
    }
    // std::cout << ros::Time::now().toSec() - start << "[s]" << std::endl;
}

void ClosedSignDetector::velodyne_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr stop_sign_cluster (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cloud_in);
    if(cloud_in->points.size()){
            process(cloud_in, stop_sign_cluster);
    }
    sensor_msgs::PointCloud2 cloud_ros;
    pcl::toROSMsg(*stop_sign_cluster, cloud_ros);
    cloud_ros.header.frame_id = msg->header.frame_id;
    cloud_ros.header.stamp = ros::Time::now();
    cloud_pub.publish(cloud_ros);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "closed_sign_detector");
    ClosedSignDetector detector;
    ros::spin();

    return 0;
}
