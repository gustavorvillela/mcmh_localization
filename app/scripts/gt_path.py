#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class GroundTruthPath:
    def __init__(self):
        rospy.init_node("ground_truth_path")

        # Parameters
        self.robot_name = rospy.get_param("~robot_name", "turtlebot3_waffle")
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.max_path_length = rospy.get_param("~max_path_length", 2000)

        # Publisher
        self.path_pub = rospy.Publisher("/ground_truth/path", Path, queue_size=10)

        # Path message
        self.path = Path()
        self.path.header.frame_id = self.frame_id
        self.last_update_time = rospy.Time(0)
        self.publish_period = rospy.Duration(0.1)  # 10 Hz

        # Subscriber
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback)

        rospy.loginfo(f"[GT PATH] Tracking robot: {self.robot_name}")


    def callback(self, msg):
        if self.robot_name not in msg.name:
            return

        i = msg.name.index(self.robot_name)
         
        pose = msg.pose[i]
        now = rospy.Time.now()

        now = rospy.Time.now()

        if (now - self.last_update_time) < self.publish_period:
            return

        self.last_update_time = now

        # Build PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = now
        pose_stamped.header.frame_id = self.frame_id

        pose_stamped.pose.position.x = pose.position.x + 0.7
        pose_stamped.pose.position.y = pose.position.y
        pose_stamped.pose.position.z = pose.position.z

        # Append to path
        self.path.header.stamp = now
        

        self.path.poses.append(pose_stamped)

        # Limit path length
        if len(self.path.poses) > self.max_path_length:
            self.path.poses.pop(0)

        # Publish
        self.path_pub.publish(self.path)


if __name__ == "__main__":
    try:
        node = GroundTruthPath()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass