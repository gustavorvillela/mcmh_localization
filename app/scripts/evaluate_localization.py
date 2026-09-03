#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64
import os
from tf.transformations import euler_from_quaternion

class Evaluator:
    def __init__(self):
        self.est_topic = rospy.get_param("~est_topic", "/estimated_pose")
        self.gt_topic = rospy.get_param("~gt_topic", "/gazebo/model_states")
        self.mh_topic = rospy.get_param("~mh_topic", "/mh_rate")
        self.robot_name = rospy.get_param("~robot_name", "turtlebot3_waffle")
        self.Neff = rospy.get_param("~effective_sample_size", "/effective_sample_size")

        result_param = rospy.get_param("~result_name", "eval")
        default_results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../results")
        )

        if os.path.isabs(result_param):
            results_dir = os.path.dirname(result_param)
            result_name = os.path.basename(result_param)
        else:
            results_dir = default_results_dir
            result_name = result_param

        result_name = os.path.basename(result_name).replace(".txt", "")
        os.makedirs(results_dir, exist_ok=True)

        self.poses_file = os.path.join(results_dir, f"poses_{result_name}.txt")
        self.neff_file = os.path.join(results_dir, f"neff_{result_name}.txt")

        self.gt_pose = None
        self.mh_rate = None
        self.eval_start_time = None
        
        self.Neff_history = []

        # Store poses
        self.pose_history = []

        rospy.Subscriber(self.est_topic, PoseWithCovarianceStamped, self.estimated_callback)
        rospy.Subscriber(self.gt_topic, ModelStates, self.gt_callback)
        rospy.Subscriber(self.mh_topic, Float64, self.mh_callback)
        rospy.Subscriber(self.Neff, Float64, self.neff_callback)

    def get_yaw_from_pose(self, pose):
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        _, _, yaw = euler_from_quaternion(quat)
        return yaw

    def estimated_callback(self, msg):
        if self.gt_pose is None:
            print("Waiting for ground truth pose...")
            print(f"Robot name: {self.robot_name}")
            return

        # Use ROS timestamp (better than wall time)
        timestamp = msg.header.stamp.to_sec()

        est_pose = msg.pose.pose
        est_x = est_pose.position.x
        est_y = est_pose.position.y
        est_yaw = self.get_yaw_from_pose(est_pose)

        gt_x = self.gt_pose.position.x
        gt_y = self.gt_pose.position.y
        gt_yaw = self.get_yaw_from_pose(self.gt_pose)

        mh_rate = self.mh_rate if self.mh_rate is not None else 0.0
        #print(f"Time: {timestamp:.2f}, Est: ({est_x:.2f}, {est_y:.2f}, {est_yaw:.2f}), "
        #      f"GT: ({gt_x:.2f}, {gt_y:.2f}, {gt_yaw:.2f}), MH Rate: {mh_rate:.4f}")

        self.pose_history.append((
            timestamp,
            est_x, est_y, est_yaw,
            gt_x, gt_y, gt_yaw, mh_rate
        ))

    def gt_callback(self, msg):
        if self.robot_name not in msg.name:
            return
        idx = msg.name.index(self.robot_name)
        self.gt_pose = msg.pose[idx]

    def mh_callback(self, msg):
        self.mh_rate = msg.data

    # Action: Save the value in suscriber into a list
    # I/ Float64: msg
    # I/O/ Self@Evaluator: self
    # Necessity: A self that contain a List: Neff_history
    #           and a valid Float64 message
    # Produce: Append in self.Neff_history the value contained in msg
    def neff_callback (self, msg) :
        #print(f"[Test] : Neff={msg.data}")
        self.Neff_history.append(msg.data)

    def run(self):
        rospy.loginfo("Recording poses only...")
        rospy.spin()

    def save_results(self):
        if not self.pose_history:
            rospy.logwarn("No pose data recorded.")
            return

        with open(self.poses_file, "w") as f:
            f.write("time,est_x,est_y,est_yaw,gt_x,gt_y,gt_yaw,mh_rate\n")
            for data in self.pose_history :
                f.write(
                    f"{data[0]:.6f},{data[1]:.4f},{data[2]:.4f},{data[3]:.6f},"
                    f"{data[4]:.4f},{data[5]:.4f},{data[6]:.6f},{data[7]:.6f}\n"
                )

        with open(self.neff_file, "w") as f:
            f.write("Neff\n")
            for data in self.Neff_history:
                f.write(
                    f"{data:.4f}\n"
                )
        
        rospy.loginfo(f"Data saved to: {self.poses_file}")

if __name__ == "__main__":
    rospy.init_node("evaluate_localization")
    evaluator = Evaluator()

    rospy.on_shutdown(evaluator.save_results)

    evaluator.run()
