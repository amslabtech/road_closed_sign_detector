#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/kdtree/kdtree.h> 
#include <pcl/search/kdtree.h> 
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>

class Color_cone_detector{
	private:
		ros::NodeHandle nh;
		tf::TransformListener tflistener;
		ros::Subscriber sub;
		ros::Publisher pub;
		sensor_msgs::PointCloud2 cloud_input;
		std::vector<sensor_msgs::PointCloud2> cloud_clusters_ros_;

	public:
		Color_cone_detector();
		void hokuyo_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
		void operate();
		void clusterKdTree(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_ptr, std::vector<pcl::PointIndices> &cluster_indices);
		void extractCluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_pcl_ptr, std::vector<pcl::PointIndices> &cluster_indices);
};

Color_cone_detector::Color_cone_detector()
{
	vub = nh.subscribe("/hokuyo_obstacles", 1, &Color_cone_detector::hokuyo_callback, this);
	pub = nh.advertise<sensor_msgs::PointCloud2>("cluster_test",1);
}

void Color_cone_detector::operate()
{

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input_pcl_ptr (new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<pcl::PointIndices> cluster_indices;

	pcl::fromROSMsg(cloud_input, *cloud_input_pcl_ptr);
	clusterKdTree(cloud_input_pcl_ptr, cluster_indices);
	extractCluster(cloud_input_pcl_ptr, cluster_indices);
}

void Color_cone_detector::clusterKdTree(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_ptr, std::vector<pcl::PointIndices> &cluster_indices)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_filtered_ptr);
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(0.1);
	ec.setMinClusterSize(5);
	ec.setMaxClusterSize(60);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered_ptr);
	ec.extract(cluster_indices);
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

		sensor_msgs::PointCloud2 cloud_cluster_ros;
		pcl::toROSMsg(*cloud_cluster_pcl_ptr, cloud_cluster_ros);
		cloud_cluster_ros.header.frame_id = "base_link";

		pub.publish(cloud_cluster_ros);
	}

}

void Color_cone_detector::hokuyo_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	cloud_input = *msg;
	std::cout << "---subscribe---" << std::endl;
	operate();
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "road_closed_sign_detector");
	
	Color_cone_detector detector;
	ros::spin();

	return 0;
}