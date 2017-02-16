#include <ros/ros.h>
#include <ros/package.h>

#include <dynamic_reconfigure/server.h>

#include <actionlib/server/simple_action_server.h>
#include <icp_alignment_server/PointcloudAlignmentAction.h>
#include <geometry_msgs/PoseStamped.h>

#include <string>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <float.h>
#include <vector>
#include <algorithm>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/point_types_conversion.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <omp.h>
#include <pthread.h>


using namespace Eigen;
using namespace std;

typedef struct Cube {
    VectorXf t0, half_edge_length;
    int depth;
} Cube;

typedef struct QueueElement {
    Cube *cube;
    struct QueueElement *next;
} QueueElement;

typedef struct PriorityQueue {
    QueueElement *head;
} PriorityQueue;

class PointcloudAlignmentAction
{
private:

protected:
    ros::NodeHandle nh_;
    actionlib::SimpleActionServer<icp_alignment_server::PointcloudAlignmentAction> as_;
    std::string action_name_;
    icp_alignment_server::PointcloudAlignmentFeedback feedback_;
    icp_alignment_server::PointcloudAlignmentResult result_;

public:

    float DISTANCE_THRESHOLD, MIN_OVERLAPPING_PERCENTAGE, EVALUATION_THRESHOLD,
          MAX_TIME, ICP_EPS, ICP_EPS2, MAX_NUMERICAL_ERROR, LAMBDA, MU;
    int NUMBER_SUBCLOUDS, SIZE_SOURCE, SIZE_TARGET, REFINEMENT_ICP_SOURCE_SIZE, REFINEMENT_ICP_TARGET_SIZE, MAX_DEPTH, MAX_ICP_IT, MAX_ICP_EVALUATIONS, NUMBER_ROTATION_SAMPLES;

    pcl::KdTreeFLANN<pcl::PointXYZ> targetKdTree;

    PointcloudAlignmentAction(std::string name) :
        as_(nh_, name, boost::bind(&PointcloudAlignmentAction::executeCB, this, _1), false),
        action_name_(name) {

        initializeParameters();

        as_.start();
    }

    ~PointcloudAlignmentAction(void) {}

        void executeCB(const icp_alignment_server::PointcloudAlignmentGoalConstPtr &goal) {

        MatrixXf scancloud = convertPointcloud(goal->scancloud);
        MatrixXf pointmap = convertPointcloud(goal->pointmap);
        VectorXf validRotationAxis = convertVector(goal->valid_rotation_axis);

        MatrixXf R(3,3);
        VectorXf t(3);

        float error = global_pointcloud_alignment(scancloud, pointmap, R, t, validRotationAxis);

        // send result to client
        geometry_msgs::Quaternion orientation;
        geometry_msgs::Point position;

        if (rotationIsValid(R) == false || R(0,0) + R(1,1) + R(2,2) == 0) {
            ROS_ERROR("Something went wrong! Computed rotation matrix is not valid!");
            R = MatrixXf::Identity(3,3);
            t = VectorXf(3);
            t << 0,0,0;
        }

        position.x = t(0);
        position.y = t(1);
        position.z = t(2);

        orientation.w = sqrt(1. + R(0,0) + R(1,1) + R(2,2)) / 2.;
        orientation.x = (R(2,1) - R(1,2)) / (4.*orientation.w);
        orientation.y = (R(0,2) - R(2,0)) / (4.*orientation.w);
        orientation.z = (R(1,0) - R(0,1)) / (4.*orientation.w);

        geometry_msgs::PoseStamped result;

        result.pose.orientation = orientation;
        result.pose.position = position;

        result_.estimated_pose = result;

        ROS_INFO("%s: Succeeded", action_name_.c_str());
        as_.setSucceeded(result_);
    }

    // the global search procedure
    float global_pointcloud_alignment(MatrixXf &source_pointcloud, MatrixXf &target_pointcloud, MatrixXf &R, VectorXf &t, VectorXf const &validRotationAxis) {

        // set up parameters, kdTree, ...
        struct timeval start;
        gettimeofday(&start, NULL);

        int queueLength;
        Cube **Q = initPriorityQueue(queueLength, target_pointcloud);

        MatrixXf R_init = copyMatrix(R);

        int itCt = 0;
        float cur_err = FLT_MAX;
        float cur_percentage = FLT_MIN;

        MatrixXf *source_subclouds = subsample_source_cloud(source_pointcloud, SIZE_SOURCE);

        MatrixXf target_subcloud = random_filter(target_pointcloud, SIZE_TARGET);
        createTargetKdTree(target_subcloud);

        float best_quality = FLT_MAX;

        MatrixXf *rotation_samples = new MatrixXf[NUMBER_ROTATION_SAMPLES];
        for (int i = 0; i < NUMBER_ROTATION_SAMPLES; i++) {
            float angle = i * (M_PI/((float) NUMBER_ROTATION_SAMPLES));
            VectorXf r = copyVector(validRotationAxis);
            r(0) = (r(0)/r.norm())*angle;
            r(1) = (r(1)/r.norm())*angle;
            r(2) = (r(2)/r.norm())*angle;

            rotation_samples[i] = getAARot(r);
        }


        // process priority queue
        int i = 0;
        #pragma omp parallel for shared(cur_err, R, i, cur_percentage, itCt, best_quality)
        for (i = 0; i < queueLength; i++) {

            if (i >= MAX_ICP_EVALUATIONS) {
                continue;
            }

            for (int j = 0; j < NUMBER_ROTATION_SAMPLES; j++) {
                MatrixXf R_i = copyMatrix(rotation_samples[j]);
                VectorXf t_i = copyVector(Q[i]->t0);

                local_pointcloud_alignment(source_subclouds, target_subcloud, R_i, t_i);

                float percentage = calc_overlapping_percentage(source_subclouds[0], target_subcloud, R_i, t_i);
                float ppe = per_point_error(source_subclouds[0], target_subcloud , R_i, t_i);
                VectorXf axis_i = getAxisOfRotation(R_i);
                float rotation_error = min((validRotationAxis-axis_i).norm(), (validRotationAxis+axis_i).norm()); // TODO: auch invertieren?

                float quality = eval_quality(percentage, ppe, rotation_error);

                if (quality < best_quality) {
                    cur_err = ppe;
                    cur_percentage = percentage;

                    best_quality = quality;

                    R = R_i;
                    t = t_i;

                    sendFeedback(cur_percentage, cur_err);
                }

                itCt++;
            }
        }

        ROS_INFO("Executed %d icp iterations, per-point error: %f, aligned percentage: %f.", itCt+1, cur_err, cur_percentage);
        return cur_err;
    }

    // the modified ICP algorithm
    float local_pointcloud_alignment(MatrixXf *source_subclouds, MatrixXf const &target_pointcloud, MatrixXf &R, VectorXf &t) {

        // create variables and stuff..
        int source_size = source_subclouds[0].cols(); // all subclouds have the same size

        float err_old;
        int itCt = 0;
        int source_pos = 0; // denotes the currently used subsample of the source cloud

        MatrixXf correspondences(3, source_size);
        VectorXf distances(source_size);
        MatrixXf source_proj(3, source_size);
        MatrixXf source_cloud;
        MatrixXf source_trimmed, correspondences_trimmed;
        MatrixXf R_old(3,3);
        VectorXf t_old(3);

        // start ICP iteration
        while(itCt < MAX_ICP_IT) {

            source_cloud = source_subclouds[source_pos % NUMBER_SUBCLOUDS]; // update subsample of the source cloud
            itCt++;

            R_old = R;
            t_old = t;

            apply_transformation(source_cloud, source_proj, R, t);

            // E-Step: assign all projected source points their correspondence
            if (find_correspondences(source_proj, target_pointcloud, correspondences, distances) == false) {
                return FLT_MAX;
            }

            // discard all source-correspondence-pairs which distance is too high
            if (trim_pointcloud(source_cloud, correspondences, distances, source_trimmed, correspondences_trimmed) == false) {
                return FLT_MAX;
            }

            // M-Step: minimize distance between source and correspondence points
            if (find_transformation(source_trimmed, correspondences_trimmed, R, t) == false) {
                return FLT_MAX;
            }

            // if the parameters did not change, use different subsample of the source pointcloud
            // if all subsamples have been tried without any effect, quit the loop
            if ((R-R_old).norm() + (t-t_old).norm() < ICP_EPS) {
                if (source_pos == 0) {
                    err_old = calc_error(source_cloud, target_pointcloud, R, t);
                } else if (source_pos % NUMBER_SUBCLOUDS == 0) {
                    if ((R-R_old).norm() + (t-t_old).norm() < ICP_EPS2) {
                        break;
                    } else {
                        err_old = calc_error(source_cloud, target_pointcloud, R, t);;
                    }
                }
                source_pos++;
            }
        }

        return calc_error(source_subclouds[0], target_pointcloud, R, t);
    }

    // quality function
    float eval_quality(float percentage, float ppe, float rotation_error) {
        return log(1.-percentage) + LAMBDA*log(ppe) + MU*log(rotation_error);
    }

    // calculates the percentage of points which distance to their nearest neighbor is smaller than the evaluation threshold
    float calc_overlapping_percentage(MatrixXf const &source_cloud, MatrixXf const &target_cloud, MatrixXf const &R, MatrixXf const &t) {
        return (((float) pointsLowerThanThreshold(source_cloud, target_cloud, R, t)) / ((float) source_cloud.cols()));
    }

    // calculates the number of points which distance to their nearest neighbor is smaller than the evaluation threshold
    int pointsLowerThanThreshold(MatrixXf const &source_cloud, MatrixXf const &target_pointcloud, MatrixXf const &R, VectorXf const& t) {
        MatrixXf source_proj(source_cloud.rows(), source_cloud.cols());
        apply_transformation(source_cloud, source_proj, R, t);
        MatrixXf correspondences(source_cloud.rows(), source_cloud.cols());
        VectorXf distances(source_cloud.cols());
        int number = 0;

        if (find_correspondences(source_proj, target_pointcloud , correspondences, distances) == false) {
            return INT_MAX;
        }

        for (int i = 0; i < source_cloud.cols(); i++) {
            if (distances(i) < EVALUATION_THRESHOLD) {
                number++;
            }
        }

        return number;
    }

    // discards all source-correspondence pairs which distance is higher than the distance threshold
    bool trim_pointcloud(MatrixXf const &pointcloud, MatrixXf const &correspondences, VectorXf const &distances, MatrixXf &pointcloud_trimmed, MatrixXf &correspondences_trimmed) {

        int min_valid_points = (int) (MIN_OVERLAPPING_PERCENTAGE*((float) pointcloud.cols()));
        int number_inliers = 0;

        for (int i = 0; i < distances.rows(); i++) {
            if (distances(i) < DISTANCE_THRESHOLD) {
                number_inliers++;
            }
        }

        if (number_inliers < min_valid_points) {
            number_inliers = min_valid_points;
        }

        pointcloud_trimmed = MatrixXf(3,number_inliers);
        correspondences_trimmed = MatrixXf(3, number_inliers);

        VectorXf distances_sorted = distances;

        sort(distances_sorted);

        float threshold = distances_sorted(number_inliers-1);

        int pos = 0;
        for (int i = 0; i < correspondences.cols(); i++) {
            if (distances(i) <= threshold && pos < number_inliers) {
                pointcloud_trimmed(0,pos) = pointcloud(0, i);
                pointcloud_trimmed(1,pos) = pointcloud(1, i);
                pointcloud_trimmed(2,pos) = pointcloud(2, i);

                correspondences_trimmed(0,pos) = correspondences(0, i);
                correspondences_trimmed(1,pos) = correspondences(1, i);
                correspondences_trimmed(2,pos) = correspondences(2, i);

                pos++;
            }
        }

        return true;
    }

    // assigns all points from the source cloud their nearest neighbor from the target cloud
    bool find_correspondences(MatrixXf const &source_pointcloud, MatrixXf const &target_pointcloud , MatrixXf &correspondences, VectorXf &distances) {
        pcl::PointXYZ searchPoint;

        for (int i = 0; i < source_pointcloud.cols(); i++) {
            searchPoint.x = source_pointcloud(0,i);
            searchPoint.y = source_pointcloud(1,i);
            searchPoint.z = source_pointcloud(2,i);

            if ((isnormal(searchPoint.x) == 0 && searchPoint.x != 0) ||
                (isnormal(searchPoint.y) == 0 && searchPoint.y != 0) ||
                (isnormal(searchPoint.z) == 0 && searchPoint.z != 0)) {
                return false;
            }

            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);
            if (targetKdTree.nearestKSearch (searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                correspondences(0,i) = target_pointcloud(0,pointIdxNKNSearch[0]);
                correspondences(1,i) = target_pointcloud(1,pointIdxNKNSearch[0]);
                correspondences(2,i) = target_pointcloud(2,pointIdxNKNSearch[0]);

                distances(i) = sqrt(pointNKNSquaredDistance[0]);
            }
        }

        return true;
    }

    // minimizes the sum of squared distances between the source-correspondence pairs by optimizing the transformation parameters
    bool find_transformation(MatrixXf const &pointcloud, MatrixXf const &correspondences, MatrixXf &R, VectorXf &t) {
        VectorXf mean1 = pointcloud.array().rowwise().mean();
        VectorXf mean2 = correspondences.array().rowwise().mean();

        MatrixXf pointcloud_norm = pointcloud.array().colwise() - mean1.array();
        MatrixXf correspondences_norm = correspondences.array().colwise() - mean2.array();

        MatrixXf W(3,3);
        W(0,0) = (pointcloud_norm.block(0,0,1,pointcloud.cols()) * correspondences_norm.block(0,0,1,correspondences.cols()).transpose())(0);
        W(0,1) = (pointcloud_norm.block(0,0,1,pointcloud.cols()) * correspondences_norm.block(1,0,1,correspondences.cols()).transpose())(0);
        W(0,2) = (pointcloud_norm.block(0,0,1,pointcloud.cols()) * correspondences_norm.block(2,0,1,correspondences.cols()).transpose())(0);

        W(1,0) = (pointcloud_norm.block(1,0,1,pointcloud.cols()) * correspondences_norm.block(0,0,1,correspondences.cols()).transpose())(0);
        W(1,1) = (pointcloud_norm.block(1,0,1,pointcloud.cols()) * correspondences_norm.block(1,0,1,correspondences.cols()).transpose())(0);
        W(1,2) = (pointcloud_norm.block(1,0,1,pointcloud.cols()) * correspondences_norm.block(2,0,1,correspondences.cols()).transpose())(0);

        W(2,0) = (pointcloud_norm.block(2,0,1,pointcloud.cols()) * correspondences_norm.block(0,0,1,correspondences.cols()).transpose())(0);
        W(2,1) = (pointcloud_norm.block(2,0,1,pointcloud.cols()) * correspondences_norm.block(1,0,1,correspondences.cols()).transpose())(0);
        W(2,2) = (pointcloud_norm.block(2,0,1,pointcloud.cols()) * correspondences_norm.block(2,0,1,correspondences.cols()).transpose())(0);

        JacobiSVD<MatrixXf> svd(W, ComputeThinU | ComputeThinV);

        MatrixXf U = -svd.matrixU();
        MatrixXf V = -svd.matrixV();

        R = U*V.transpose();
        R = R.inverse();

        if (R.determinant() < 0) {
            MatrixXf V = svd.matrixV();
            V(0,2) = -V(0,2);
            V(1,2) = -V(1,2);
            V(2,2) = -V(2,2);
            R = U*V.transpose();
            R = R.inverse();
        }

        t = mean2 - R*mean1;

        return parametersValid(R,t);
    }

    // applies the current transformation T(R,t,s) to the given pointcloud
    void apply_transformation(MatrixXf const &pointcloud, MatrixXf &pointcloud_proj, MatrixXf const &R, VectorXf const &t) {
        pointcloud_proj = R*pointcloud;
        pointcloud_proj = pointcloud_proj.array().colwise() + t.array();
    }

    // calculates the sum-of-least-squares of all inliers and divides it by the number of inliers
    float per_point_error(MatrixXf const &source_pointcloud, MatrixXf const &target_pointcloud, MatrixXf const &R, VectorXf const &t) {
        MatrixXf source_proj(3, source_pointcloud.cols());
        MatrixXf correspondences(3, source_pointcloud.cols());
        VectorXf distances(source_pointcloud.cols());

        apply_transformation(source_pointcloud, source_proj, R, t);
        if (find_correspondences(source_proj, target_pointcloud, correspondences, distances) == false) {
            return FLT_MAX;
        }

        MatrixXf diff = source_proj - correspondences;

        float err = 0;
        int n_p = 0;

        for (int i = 0; i < distances.rows(); i++) {
            if (distances(i) < EVALUATION_THRESHOLD) {
                err += diff(0,i)*diff(0,i) + diff(1,i)*diff(1,i) + diff(2,i)*diff(2,i);
                n_p++;
            }
        }

        if (n_p == 0) {
            return FLT_MAX;
        }

        return sqrt(err) / ((float) n_p);
    }


    // calculates the sum of least squares error between the source points and their nearest neighbors in the target cloud
    float calc_error(MatrixXf const &source_pointcloud, MatrixXf const &target_pointcloud, MatrixXf const &R, VectorXf const &t) {

        MatrixXf source_proj(3, source_pointcloud.cols());
        MatrixXf correspondences(3, source_pointcloud.cols());
        VectorXf distances(source_pointcloud.cols());
        MatrixXf source_proj_trimmed, correspondences_trimmed;

        apply_transformation(source_pointcloud, source_proj, R, t);
        if (find_correspondences(source_proj, target_pointcloud, correspondences, distances) == false) {
            return FLT_MAX;
        }
        trim_pointcloud(source_proj, correspondences, distances, source_proj_trimmed, correspondences_trimmed);


        MatrixXf diff = source_proj_trimmed - correspondences_trimmed;

        float err = 0;
        for (int i = 0; i < diff.cols(); i++) {

            err += diff(0,i)*diff(0,i) + diff(1,i)*diff(1,i) + diff(2,i)*diff(2,i);
        }

        return sqrt(err);
    }

    // sorts the given vector in increasing order
    void sort(VectorXf &v) {
      std::sort(v.data(), v.data()+v.size());
    }

    // converts the Eigen matrix to a vector
    VectorXf matrixToVector(MatrixXf m) {
        m.transposeInPlace();
        VectorXf v(Map<VectorXf>(m.data(), m.cols()*m.rows()));
        return v;
    }

    // returns the passed time since start
    float getPassedTime(struct timeval start) {
        struct timeval end;
        gettimeofday(&end, NULL);

        return (float) (((1.0/1000)*((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))/1000.);
    }

    // returns the Angle-Axis rotation matrix for the given vector
    MatrixXf getAARot(VectorXf &r) {
        MatrixXf R = MatrixXf::Identity(3,3);

        if (r.norm() == 0) {
            return R;
        }

        MatrixXf r_x(3,3);
        r_x << 0, -r(2), r(1),
            r(2), 0, -r(0),
            -r(1), r(0), 0;


        R += (r_x*sin(r.norm()))/(r.norm());
        R += (r_x*r_x*(1-cos(r.norm())))/(r.norm()*r.norm());

        return R;
    }

    // creates a new cube with given half_edge length, r0 and depth
    Cube* createCube(VectorXf t0, VectorXf half_edge_length, int depth) {
        Cube *C = new Cube;
        C->t0 = t0;
        C->half_edge_length = half_edge_length;
        C->depth = depth;

        return C;
    }

    // compares two cubes and states which of the two cubes has to be searched first (lower depth is better)
    bool betterThan(Cube *cube1, Cube *cube2) {
        if (cube1->depth < cube2->depth) {
            return true;
        } else if (cube1->depth == cube2->depth) {
            return true;
        } else {
            return false;
        }
    }

    // updates the kdTree with the given point cloud
    void createTargetKdTree(MatrixXf &pointcloud) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr target_pcl_pointcloud (new pcl::PointCloud<pcl::PointXYZ>);;

        // Fill in the cloud data
        target_pcl_pointcloud->width    = pointcloud.cols();
        target_pcl_pointcloud->height   = 1;
        target_pcl_pointcloud->is_dense = false;
        target_pcl_pointcloud->points.resize(target_pcl_pointcloud->width * target_pcl_pointcloud->height);

        for (int i = 0; i < pointcloud.cols(); i++) {
            target_pcl_pointcloud->points[i].x = pointcloud(0, i);
            target_pcl_pointcloud->points[i].y = pointcloud(1, i);
            target_pcl_pointcloud->points[i].z = pointcloud(2, i);
        }

        targetKdTree.setInputCloud(target_pcl_pointcloud);
    }

    // returns a subcloud of the given point cloud with number_points in it by a random sampling procedure
    MatrixXf random_filter(MatrixXf &pointcloud, int number_points) {
        if (pointcloud.cols() <= number_points) {
            return pointcloud;
        }

        vector<int> indices;
        for (int i = 0; i < pointcloud.cols(); i++) {
            indices.push_back(i);
        }
        random_shuffle(indices.begin(), indices.end());

        MatrixXf filtered_pointcloud(pointcloud.rows(), number_points);

        for (int i = 0; i < number_points; i++) {
            filtered_pointcloud(0,i) = pointcloud(0, indices[i]);
            filtered_pointcloud(1,i) = pointcloud(1, indices[i]);
            filtered_pointcloud(2,i) = pointcloud(2, indices[i]);
        }

        return filtered_pointcloud;
    }

    // preprocesses the source pointcloud: convertes it to a Eigen matrix and calculates the maximum radius between points in the source cloud
    MatrixXf convertPointcloud(sensor_msgs::PointCloud2 source_msg) {
        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pointcloud_source (new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(source_msg , *pointcloud_source);

        MatrixXf source_pointcloud = MatrixXf(3,pointcloud_source->size());

        for (int i = 0; i < pointcloud_source->size(); i++) {
            source_pointcloud(0,i) = pointcloud_source->at(i).x;
            source_pointcloud(1,i) = pointcloud_source->at(i).y;
            source_pointcloud(2,i) = pointcloud_source->at(i).z;
        }

        return source_pointcloud;
    }

    VectorXf convertVector(geometry_msgs::Point r) {
        VectorXf axis(3);
        axis << r.x, r.y, r.z;

        return axis;
    }

    // returns the chosen number of subsamples of the source cloud
    MatrixXf *subsample_source_cloud(MatrixXf source_pointcloud, float size_source) {
        MatrixXf *source_subclouds = new MatrixXf[NUMBER_SUBCLOUDS];

        for (int i = 0; i < NUMBER_SUBCLOUDS; i++) {
            source_subclouds[i] = random_filter(source_pointcloud, size_source);
        }
        return source_subclouds;
    }

    // sends feedback to the client
    void sendFeedback(float percentage, float err) {
        feedback_.aligned_percentage = percentage;
        feedback_.normalized_error = err;
        as_.publishFeedback(feedback_);
    }

    // states if the given rotation matrix is a valid one
    bool rotationIsValid(MatrixXf R) {
        if (abs(R.determinant()-1) > MAX_NUMERICAL_ERROR || (R*R.transpose() - MatrixXf::Identity(3,3)).norm() > MAX_NUMERICAL_ERROR) {
            return false;
        }
        return true;
    }

    // validates all transformation parameters
    bool parametersValid(MatrixXf R, VectorXf t) {
        for (int i = 0; i < R.rows(); i++) {
            for (int j = 0; j < R.cols(); j++) {
                if (isnormal(R(i,j)) == 0 && R(i,j) != 0) {
                    return false;
                }
            }
        }

        for (int i = 0; i < t.cols(); i++) {
            if (isnormal(t(i)) == 0 && t(i) != 0) {
                return false;
            }
        }

        return true;
    }

    // copy an Eigen matrix into a new data structure
    MatrixXf copyMatrix(MatrixXf const &m) {
        MatrixXf m_new(m.rows(), m.cols());

        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                m_new(i,j) = m(i,j);
            }
        }

        return m_new;
    }

    // copy an Eigen vector into a new data structure
    VectorXf copyVector(VectorXf const &v) {
        VectorXf v_new(v.rows());

        for (int i = 0; i < v.rows(); i++) {
            v_new(i) = v(i);
        }

        return v_new;
    }

    // returns a rotation matrix that rotates around axis 0==x, 1==y or 2 == z by an angle of pi
    MatrixXf getSymmetryRotation(int axis) {
        MatrixXf R(3,3);

        if (axis == 0) {    // x-axis
            R <<  1, 0, 0,
                  0, -1, 0,
                  0, 0, -1;
        } else if (axis == 1) { // y-axis
            R <<  -1, 0, 0,
                  0, 1, 0,
                  0, 0, -1;
        } else {    // z-axis
            R <<  -1, 0, 0,
                  0, -1, 0,
                  0, 0, 1;
        }

        return R;
    }

    VectorXf getAxisOfRotation(MatrixXf const &R) {
        VectorXf axis(3);

        float theta = acos(((R(0,0)+R(1,1)+R(2,2))-1.0f)/2.0f);

        axis(0) = (R(3,2)-R(2,3))/(2.0f*sin(theta));
        axis(1) = (R(1,3)-R(3,1))/(2.0f*sin(theta));
        axis(2) = (R(2,1)-R(1,2))/(2.0f*sin(theta));

        return axis;
    }

    //************* functions for management of the priority queue and the octree ***********

    // splits the given cube of the octree into eight subcubes
    Cube **splitCube(Cube *cube) {
        Cube **subcubes = new Cube*[8];
        VectorXf offset(3);
        float signs[2] = {-1.,+1.};

        int position = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    offset << cube->half_edge_length(0)/2.0f*signs[i],cube->half_edge_length(1)/2.0f*signs[j],cube->half_edge_length(2)/2.0f*signs[k];
                    VectorXf hel = copyVector(cube->half_edge_length / 2.0f);
                    subcubes[position++] = createCube(cube->t0 + offset, hel, cube->depth + 1);

                }
            }
        }

        return subcubes;
    }

    // returns the number of cubes in the priority queue with the given depth
    int depthNumber(PriorityQueue *Q, int depth) {
        int ct = 0;
        QueueElement *tmp = Q->head;
        while (tmp != NULL) {
            if (tmp->cube->depth == depth) {
                ct++;
            }
            tmp = tmp->next;
        }
        return ct;
    }

    // permutes the numbers between start and end
    int *getPermutation(int start, int end) {
        int length = end-start+1;

        vector<int> indices;
        for (int i = 0; i < length; i++) {
            indices.push_back(start+i);
        }
        random_shuffle(indices.begin(), indices.end());

        int *permutation = new int[length];
        for (int i = 0; i < length; i++) {
            permutation[i] = indices[i];
        }

        return permutation;
    }

    // initalizes a new priority queue with all cubes up to MAX_DEPTH and returns its length
    Cube **initPriorityQueue(int &queueLength, MatrixXf const &pointmap) {
        PriorityQueue *Q = createQueue();

        VectorXf t0_init(3), half_edge_length(3);
        t0_init<<0,0,0; // TODO:
        half_edge_length << 0,0,0;

        for (int i = 0; i < pointmap.cols(); i++) {
            t0_init(0) += pointmap(0,i);
            t0_init(1) += pointmap(1,i);
            t0_init(2) += pointmap(2,i);

        }
        t0_init(0) /= (float) pointmap.cols();
        t0_init(1) /= (float) pointmap.cols();
        t0_init(2) /= (float) pointmap.cols();

        float xMin = FLT_MAX, yMin = FLT_MAX, zMin = FLT_MAX;
        float xMax = FLT_MIN, yMax = FLT_MIN, zMax = FLT_MIN;

        for (int i = 0; i < pointmap.cols(); i++) {
            float x = pointmap(0,i) - t0_init(0);
            float y = pointmap(1,i) - t0_init(1);
            float z = pointmap(2,i) - t0_init(2);

            if (x < xMin) {
                xMin = x;
            } if (y < yMin) {
                yMin = y;
            } if (z < zMin) {
                zMin = z;
            } if (x > xMax) {
                xMax = x;
            } if (y > yMax) {
                yMax = y;
            } if (zMax > z) {
                zMax = z;
            }
        }
        half_edge_length(0) = (xMax-xMin)/2.0f;
        half_edge_length(1) = (yMax-yMin)/2.0f;
        half_edge_length(2) = (zMax-zMin)/2.0f;

        Cube *C_init = createCube(t0_init, half_edge_length, 0);

        fillPriorityQueue(Q, C_init, 0, MAX_DEPTH);
        int nCubes = length(Q);
        Cube **priorityQueue = new Cube *[nCubes];
        int depth = 0;
        int startPos = 0;
        while (depth <= MAX_DEPTH) {
            int ct = depthNumber(Q, depth++);
            int *offset = getPermutation(0, ct-1);

            for (int i = 0; i < ct; i++) {
                Cube *tmp = extractFirstElement(Q);
                priorityQueue[startPos+offset[i]] = tmp;
            }
            startPos += ct;
        }

        queueLength = nCubes;

        return priorityQueue;
    }

    // fills given priority queue with all cubes of the current depth
    void fillPriorityQueue(PriorityQueue *Q, Cube *cube, int curDepth, int MAX_DEPTH) {

        if (curDepth < MAX_DEPTH) {
            Cube **subcubes = splitCube(cube);

            for (int i = 0; i < 8; i++) {
                fillPriorityQueue(Q, subcubes[i], curDepth + 1, MAX_DEPTH);
            }
        }
    }

    // allocate queue memory
    PriorityQueue *createQueue() {
        PriorityQueue *queue = new PriorityQueue;
        queue->head = NULL;
        return queue;
    }

    // insert a new cube into the priority quene at
    void insert(PriorityQueue *queue, Cube *cube) {
        if (queue == NULL) {
            perror("queue == NULL");
            return;
        }
        QueueElement *newElement = new QueueElement;
        newElement->cube = cube;

        if (queue->head == NULL) {
            queue->head = newElement;
            queue->head->next = NULL;
            return;

        } else {
            if (betterThan(cube, queue->head->cube) == true) {
                newElement->next = queue->head;
                queue->head = newElement;
                return;
            }
            QueueElement *tmp = queue->head;
            while (tmp->next != 0) {
                if (betterThan(cube, tmp->next->cube) == true) {
                    newElement->next = tmp->next;
                    tmp->next = newElement;
                    return;;
                }

                tmp = tmp->next;
            }
            tmp->next = newElement;
            newElement->next = NULL;
        }
    }

    // deletes the priority queue
    void deleteQueue(PriorityQueue *queue) {
        if (queue == NULL)
            return;
        if (queue->head == NULL)
            free(queue);
        QueueElement *temp = queue->head, *next = NULL;
        while (temp != NULL) {
            next = temp->next;
            delete(temp);
            temp = next;
        }
        delete(queue);
    }

    // returns the length of the priority queue
    int length(PriorityQueue *queue) {
        if (queue == NULL || queue->head == NULL)
            return 0;

        int counter = 0;
        QueueElement *temp = queue->head;
        while (temp != NULL) {
            counter++;
            temp = temp->next;
        }
        return counter;
    }

    // extracts the first element from the prioirty queue and removes it from the queue
    Cube *extractFirstElement(PriorityQueue *queue) {
        if (queue == NULL || queue->head == NULL)
            return NULL;

        QueueElement *element = queue->head;
        Cube *cube = element->cube;
        queue->head = queue->head->next;
        delete(element);

        return cube;
    }

    //************* parameter initialization functions ****************************

    void initializeParameters() {
        DISTANCE_THRESHOLD = getFloatParameter("distance_threshold");
        MIN_OVERLAPPING_PERCENTAGE = getFloatParameter("min_overlapping_percentage");
        NUMBER_SUBCLOUDS = getIntegerParameter("number_subclouds");
        SIZE_SOURCE = getIntegerParameter("size_source");
        SIZE_TARGET = getIntegerParameter("size_target");
        EVALUATION_THRESHOLD = getFloatParameter("evaluation_threshold");
        MAX_DEPTH = getIntegerParameter("max_depth");
        ICP_EPS = getFloatParameter("icp_eps");
        MAX_ICP_IT = getIntegerParameter("max_icp_it");
        ICP_EPS2 = getFloatParameter("icp_eps2");
        MAX_NUMERICAL_ERROR = getFloatParameter("max_numerical_error");
        MAX_ICP_EVALUATIONS = getIntegerParameter("max_icp_evaluations");
        LAMBDA = getFloatParameter("lambda");
        MU = getFloatParameter("mu");
        NUMBER_ROTATION_SAMPLES = getIntegerParameter("number_rotation_samples");
    }

    float getFloatParameter(string parameter_name) {
        string key;
        if (nh_.searchParam(parameter_name, key) == true) {
          double val;
          ros::param::get(key, val);

          ROS_INFO("%s: %f", parameter_name.c_str(), val);

          return (float) val;
        } else {
            ROS_ERROR("parameter %s not found", parameter_name.c_str());
            return 0;
        }
    }

    int getIntegerParameter(string parameter_name) {

        string key;
        if (nh_.searchParam(parameter_name, key) == true) {
          int val;
          nh_.getParam(key, val);

          ROS_INFO("%s: %d", parameter_name.c_str(), val);

          return val;
        } else {
            ROS_ERROR("parameter %s not found", parameter_name.c_str());

            return 0;
        }
    }
    // ***********************************************************

};


int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_alignment");

    PointcloudAlignmentAction pointcloud_alignment(ros::this_node::getName());
    ros::spin();

    return 0;
}
