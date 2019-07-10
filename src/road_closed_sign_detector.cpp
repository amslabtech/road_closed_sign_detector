#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/kdtree/kdtree.h> 
#include <pcl/search/kdtree.h> 
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include<pcl/filters/voxel_grid.h>
#include<Eigen/Core>

struct Cluster{
	float x;
	float y;
	float z;
	float width;
	float height;
	float depth;
	Eigen::Vector3f min_p;
	Eigen::Vector3f max_p;
};

struct Clusters{
	Cluster data;
	pcl::PointCloud<pcl::PointXYZ> centroid;
	pcl::PointCloud<pcl::PointXYZ> points;
};

class Color_cone_detector{
	private:
		ros::NodeHandle nh;
		ros::NodeHandle private_nh;

		ros::Subscriber hokuyo_sub;
		ros::Subscriber velodyne_sub;
		ros::Publisher pub;

		sensor_msgs::PointCloud2 cloud_input;
		std::vector<sensor_msgs::PointCloud2> cloud_clusters_ros_;

		double LEAF_SIZE;
		double TOLERANCE;
		int MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE;
		double MAX_WIDTH, MAX_HEIGHT, MAX_DEPTH;
		double MIN_WIDTH, MIN_HEIGHT, MIN_DEPTH;

	public:
		Color_cone_detector();
		void hokuyo_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
		void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
		void getClusterInfo(pcl::PointCloud<pcl::PointXYZ> cloud, Cluster& cluster);
		void clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<Clusters> &cluster_array, std::vector<pcl::PointIndices> &cluster_indices);
		void pickup_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr clouds, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centroid, std::vector<Clusters> &cluster_array);
		void extractCluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_pcl_ptr, std::vector<pcl::PointIndices> &cluster_indices);
};

Color_cone_detector::Color_cone_detector()
	: private_nh("~")
{
	hokuyo_sub = nh.subscribe("/hokuyo_obstacles", 1, &Color_cone_detector::hokuyo_callback, this);
	velodyne_sub = nh.subscribe("/velodyne_obstacles", 1, &Color_cone_detector::velodyne_callback, this);
	pub = nh.advertise<sensor_msgs::PointCloud2>("cluster_test",1);

	private_nh.param("LEAF_SIZE", LEAF_SIZE, 0.08);
	private_nh.param("TOLERANCE", TOLERANCE, 0.15);
	private_nh.param("MIN_CLUSTER_SIZE", MIN_CLUSTER_SIZE, 20);
	private_nh.param("MAX_CLUSTER_SIZE", MAX_CLUSTER_SIZE, 1200);
	private_nh.param("MAX_WIDTH", MAX_WIDTH, 0.4);
	private_nh.param("MAX_HEIGHT", MAX_HEIGHT, 0.8);
	private_nh.param("MAX_DEPTH", MAX_DEPTH, 0.4);
	private_nh.param("MIN_WIDTH", MIN_WIDTH, 0.2);
	private_nh.param("MIN_HEIGHT", MIN_HEIGHT, 0.4);
	private_nh.param("MIN_DEPTH", MIN_DEPTH, 0.1);
}

void Color_cone_detector::getClusterInfo(pcl::PointCloud<pcl::PointXYZ> cloud, Cluster& cluster)
{
	Eigen::Vector3f centroid;
	centroid[0] = cloud.points[0].x;
	centroid[1] = cloud.points[0].y;
	centroid[2] = cloud.points[0].z;

	Eigen::Vector3f min_p;
	min_p[0] = cloud.points[0].x;
	min_p[1] = cloud.points[0].y;
	min_p[2] = cloud.points[0].z;

	Eigen::Vector3f max_p;
	max_p[0] = cloud.points[0].x;
	max_p[1] = cloud.points[0].y;
	max_p[2] = cloud.points[0].z;

	for(int i=0; i<cloud.points.size(); ++i){
		centroid[0] += cloud.points[i].x;
		centroid[1] += cloud.points[i].y;
		centroid[2] += cloud.points[i].z;

		if(cloud.points[i].x < min_p[0]) min_p[0] = cloud.points[i].x;
		if(cloud.points[i].y < min_p[1]) min_p[1] = cloud.points[i].y;
		if(cloud.points[i].z < min_p[2]) min_p[2] = cloud.points[i].z;

		if(cloud.points[i].x > max_p[0]) max_p[0] = cloud.points[i].x;
		if(cloud.points[i].y > max_p[1]) max_p[1] = cloud.points[i].y;
		if(cloud.points[i].z > max_p[2]) max_p[2] = cloud.points[i].z;
	}

	cluster.x = centroid[0]/cloud.points.size();
	cluster.y = centroid[1]/cloud.points.size();
	cluster.z = centroid[2]/cloud.points.size();
	cluster.depth = max_p[0] - min_p[0];
	cluster.width = max_p[1] - min_p[1];
	cluster.height = max_p[2] - min_p[2];
	cluster.min_p = min_p;
	cluster.max_p = max_p;

}
void Color_cone_detector::clustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, std::vector<Clusters> &cluster_array, std::vector<pcl::PointIndices> &cluster_indices)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr ds_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud(cloud_in);
	vg.setLeafSize(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE);
	vg.filter(*ds_cloud);

	std::vector<float> tmp_point_z;
	tmp_point_z.resize(ds_cloud->points.size());
	for(int i=0; i<ds_cloud->points.size(); ++i){
		tmp_point_z[i] = ds_cloud->points[i].z;
		ds_cloud->points[i].z = 0.0;
	}
	
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(ds_cloud);
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(TOLERANCE);
	ec.setMinClusterSize(MIN_CLUSTER_SIZE);
	ec.setMaxClusterSize(MAX_CLUSTER_SIZE);
	ec.setSearchMethod(tree);
	ec.setInputCloud(ds_cloud);
	ec.extract(cluster_indices);
	
	for(int i=0; i<ds_cloud->points.size(); ++i)
		ds_cloud->points[i].z = tmp_point_z[i];

	for(int i=0; i<cluster_indices.size(); ++i){
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
		cloud_cluster->points.resize(cluster_indices[i].indices.size());
		
		for(int j=0; j < cluster_indices[i].indices.size(); ++j){
			int point_id = cluster_indices[i].indices[j];
			cloud_cluster->points[j] = ds_cloud->points[point_id];
		}
		Cluster data;
		getClusterInfo(*cloud_cluster, data);

		pcl::PointXYZ center;
		center.x = data.x;
		center.y = data.y;
		center.z = data.z;

		Clusters cluster;
		cluster.data = data;
		cluster.centroid.points.push_back(center);

		for(int i=0; i<cloud_cluster->points.size(); ++i)
			cluster.points.points.push_back(cloud_cluster->points[i]);

		cluster_array.push_back(cluster);
	}
}

void Color_cone_detector::extractCluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_pcl_ptr, std::vector<pcl::PointIndices> &cluster_indices)
{
	for(std::vector<pcl::PointIndices>::iterator it = cluster_indices.begin(); it!= cluster_indices.end(); ++it){
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_pcl_ptr (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices(*it));
		extract.setInputCloud(cloud_filtered_pcl_ptr->makeShared());
		extract.setIndices(inliers);
		extract.setNegative(false);
		extract.filter(*cloud_cluster_pcl_ptr);
		std::cout << cloud_cluster_pcl_ptr->points.size() << std::endl;

		sensor_msgs::PointCloud2 cloud_cluster_ros;
		pcl::toROSMsg(*cloud_cluster_pcl_ptr, cloud_cluster_ros);
		cloud_cluster_ros.header.frame_id = "velodyne";

		pub.publish(cloud_cluster_ros);
		
	}
}

void Color_cone_detector::pickup_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr clouds, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centroid, std::vector<Clusters> &cluster_array)
{
	for(int i=0; i<cluster_array.size(); ++i){
		Clusters cluster = cluster_array[i];
		Cluster data = cluster.data;
		if(MIN_WIDTH < data.width && data.width < MAX_WIDTH){
			if(MIN_HEIGHT < data.height && data.height < MAX_HEIGHT){
				if(MIN_DEPTH < data.depth && data.depth < MAX_DEPTH){
					std::cout << "width  :" << data.width << "  height  :" << data.height << "  depth  :" << data.depth << std::endl;
					for(int j=0; j<cluster.points.points.size(); ++j)
						clouds->points.push_back(cluster.points.points[j]);

					cloud_centroid->points.push_back(cluster.centroid.points[0]);
				}
			}
		}
	}
}

void Color_cone_detector::velodyne_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr clouds(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centroid(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::fromROSMsg(*msg, *cloud);

	std::vector<pcl::PointIndices> cluster_indices;

	std::vector<Clusters> cluster_array;

	if(0 < cloud->points.size())
		clustering(cloud, cluster_array, cluster_indices);

	
	pickup_cluster(clouds, cloud_centroid, cluster_array);

	sensor_msgs::PointCloud2 cloud_ros;
	pcl::toROSMsg(*clouds, cloud_ros);
	cloud_ros.header.frame_id = msg->header.frame_id;
	cloud_ros.header.stamp = ros::Time::now();
	
	//std::cout << "---publish---" << std::endl;
	pub.publish(cloud_ros);

}
void Color_cone_detector::hokuyo_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
/*	cloud_input = *msg;
	std::cout << "---subscribe---" << std::endl;
	operate();
*/
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "road_closed_sign_detector");
	
	Color_cone_detector detector;
	ros::spin();

	return 0;
}
