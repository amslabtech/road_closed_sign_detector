#include "road_closed_sign_detector/road_closed_sign_detector.h"

template <typename T>
ClosedSignDetector<T>::ClosedSignDetector()
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
    private_nh.param("LOWER_INTENSITY_THRESHOLD", LOWER_INTENSITY_THRESHOLD, 20);
    private_nh.param("UPPER_INTENSITY_THRESHOLD", UPPER_INTENSITY_THRESHOLD, 255);
    private_nh.param("INTENSITY_RATIO", INTENSITY_RATIO, 60.0);
    private_nh.param("CURVATURE_THRESHOLD", CURVATURE_THRESHOLD, 0.02);
    private_nh.param("PLANE_DIST_ERROR_THRESHOLD", PLANE_DIST_ERROR_THRESHOLD, 0.005);
    private_nh.param("DIST_ERROR_THRESHOLD", DIST_ERROR_THRESHOLD, 0.5);
}

template <typename T>
inline double Pow(T x){
    return x*x;
}

template <typename T>
Eigen::Vector4f ClosedSignDetector<T>::calc_centroid(typename pcl::PointCloud<T>::Ptr& cluster)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cluster, centroid);
    return centroid;
}

template <typename T>
Eigen::Vector3f ClosedSignDetector<T>::calc_cluster_size(typename pcl::PointCloud<T>::Ptr& cluster)
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

template <typename T>
double ClosedSignDetector<T>::calc_dist(std::vector<Eigen::Vector4f>& centroids)
{
    return sqrt(Pow(centroids[0][0] - centroids[1][0]) + Pow(centroids[0][1] - centroids[1][1]));
}

template <typename T>
double ClosedSignDetector<T>::calc_intensity_ratio(typename pcl::PointCloud<T>::Ptr& cluster_points, double avg_intensity)
{
    double sample_avg_intensity = 0.0;
    int sample_point_num = cluster_points->points.size() * SAMPLE_RATIO + 1;
    for(int i=0; i<sample_point_num; ++i){
        sample_avg_intensity += cluster_points->points[i].intensity;
    }
    sample_avg_intensity /= sample_point_num;
    return sample_avg_intensity / avg_intensity;
}

template <typename T>
void ClosedSignDetector<T>::clustering(typename pcl::PointCloud<T>::Ptr& cloud_in, std::vector<pcl::PointIndices>& cluster_indices)
{
    typename pcl::search::KdTree<T>::Ptr tree (new pcl::search::KdTree<T>);
    tree->setInputCloud(cloud_in);
    pcl::EuclideanClusterExtraction<T> ec;
    ec.setClusterTolerance(TOLERANCE);
    ec.setMinClusterSize(MIN_CLUSTER_SIZE);
    ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_in);
    ec.extract(cluster_indices);
}

template <typename T>
void ClosedSignDetector<T>::downsampling(typename pcl::PointCloud<T>::Ptr& cloud_in, typename pcl::PointCloud<T>::Ptr& ds_cloud)
{
    pcl::VoxelGrid<T> vg;
    vg.setInputCloud(cloud_in);
    vg.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
    vg.filter(*ds_cloud);
}

template <typename T>
void ClosedSignDetector<T>::filter_intensity(typename pcl::PointCloud<T>::Ptr& cloud_in, typename pcl::PointCloud<T>::Ptr& cloud_out)
{
    pcl::PassThrough<T> pass;
    pass.setInputCloud(cloud_in);
    pass.setFilterFieldName("intensity");
    pass.setFilterLimits(LOWER_INTENSITY_THRESHOLD, UPPER_INTENSITY_THRESHOLD);
    pass.filter(*cloud_out);
}

template <typename T>
void ClosedSignDetector<T>::filter_plane(typename pcl::PointCloud<T>::Ptr& cluster, typename pcl::PointCloud<T>::Ptr& plane_cluster)
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::SACSegmentation<T> segmentation;
    segmentation.setInputCloud(cluster);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setDistanceThreshold(PLANE_DIST_ERROR_THRESHOLD);
    segmentation.setOptimizeCoefficients(true);
    pcl::PointIndices inlierIndices;
    segmentation.segment(inlierIndices, *coefficients);
    if(inlierIndices.indices.size()){
        pcl::copyPointCloud<T>(*cluster, inlierIndices, *plane_cluster);
    }
}

template<>
void ClosedSignDetector<pcl::PointXYZINormal>::normal_estimation(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud_in, pcl::PointCloud<pcl::Normal>::Ptr& normals)
{
    pcl::copyPointCloud(*cloud_in, *normals);
}

template <typename T>
void ClosedSignDetector<T>::normal_estimation(typename pcl::PointCloud<T>::Ptr& cloud_in, pcl::PointCloud<pcl::Normal>::Ptr& normals)
{
    typename pcl::search::KdTree<T>::Ptr tree (new pcl::search::KdTree<T>);
    tree->setInputCloud(cloud_in);
    pcl::NormalEstimationOMP<T, pcl::Normal> ne;
    ne.setInputCloud(cloud_in);
    ne.setKSearch(5);
    ne.setSearchMethod(tree);
    ne.compute(*normals);
}

template <typename T>
bool ClosedSignDetector<T>::is_valid_cluster_size(typename pcl::PointCloud<T>::Ptr& cluster, Eigen::Vector4f centroid, Eigen::Vector3f size)
{
    double width = size[0];
    double height = size[1];
    double depth = size[2];
    std::cout << "width :" << width;
    std::cout << "\theight :" << height;
    std::cout << "\tdepth :" << depth << std::endl;
    std::cout << "centroid x :" << centroid[0];
    std::cout << "\tcentroid y :" << centroid[1];
    std::cout << "\tcentroid_z :" << centroid[2] << std::endl;
    if(MIN_WIDTH < width && width < MAX_WIDTH){
        if(MIN_HEIGHT < height && height < MAX_HEIGHT){
            if(MIN_DEPTH < depth && depth < MAX_DEPTH){
                double angle = atan2(centroid[1], centroid[0]);
                if(fabs(angle) < ANGLE_THRESHOLD && centroid[2] < CENTROID_THRESHOLD){
                    // std::cout << "width :" << width << " height :" << height << " depth :" << depth << std::endl;
                    // std::cout << "centroid_z :" << centroid[2] << std::endl;
                    // std::cout << "cluster_angle :" << angle << std::endl;
                    // std::cout << "size : " << cluster->points.size() << std::endl;
                    return true;
                }
            }
        }
    }
    return false;
}

template <typename T>
void ClosedSignDetector<T>::process(typename pcl::PointCloud<T>::Ptr& cloud_in, typename pcl::PointCloud<T>::Ptr& stop_sign_cluster)
{
    double start = ros::Time::now().toSec();

    typename pcl::PointCloud<T>::Ptr ds_cloud (new pcl::PointCloud<T>);
    downsampling(cloud_in, ds_cloud);
    double avg_intensity = 0.0;
    size_t ds_point_num = ds_cloud->points.size();
    for(size_t i=0; i<ds_point_num; ++i){
        avg_intensity += ds_cloud->points[i].intensity;
    }
    avg_intensity /= ds_point_num;
    typename pcl::PointCloud<T>::Ptr intensity_cloud (new pcl::PointCloud<T>);
    filter_intensity(ds_cloud, intensity_cloud);
    std::vector<float> tmp_point_z;
    for(size_t i=0; i<intensity_cloud->points.size(); ++i){
        tmp_point_z.emplace_back(intensity_cloud->points[i].z);
        intensity_cloud->points[i].z = 0.0;
    }
    std::vector<pcl::PointIndices> cluster_indices;
    clustering(intensity_cloud, cluster_indices);
    for(size_t i=0; i<intensity_cloud->points.size(); ++i){
        intensity_cloud->points[i].z = tmp_point_z[i];
    }
    typename pcl::PointCloud<T>::Ptr points_all (new pcl::PointCloud<T>);
    std::vector<Eigen::Vector4f> centroids;
    for(const auto& cidx : cluster_indices){
        typename pcl::PointCloud<T>::Ptr cloud_cluster (new pcl::PointCloud<T>);
        for(const auto& pidx : cidx.indices){
            cloud_cluster->points.emplace_back(intensity_cloud->points[pidx]);
        }
        Eigen::Vector3f cluster_size = calc_cluster_size(cloud_cluster);
        Eigen::Vector4f cluster_centroid = calc_centroid(cloud_cluster);
        std::cout << "-----------------------------------" << std::endl;
        if(is_valid_cluster_size(cloud_cluster, cluster_centroid, cluster_size)){
            typename pcl::PointCloud<T>::Ptr plane_cluster(new pcl::PointCloud<T>);
            if(USE_RANSAC){
                filter_plane(cloud_cluster, plane_cluster);
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
            double curvature = 0.0;
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
            std::sort(plane_cluster->points.begin(), plane_cluster->points.end(), [](const T &a, const T &b) {
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

template <typename T>
void ClosedSignDetector<T>::velodyne_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    typename pcl::PointCloud<T>::Ptr cloud_in(new pcl::PointCloud<T>);
    typename pcl::PointCloud<T>::Ptr stop_sign_cluster (new pcl::PointCloud<T>);
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
    ClosedSignDetector<pcl::PointXYZINormal> detector;
    ros::spin();

    return 0;
}
