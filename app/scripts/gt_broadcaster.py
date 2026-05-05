#!/usr/bin/env python3
import rospy
from gazebo_msgs.msg import ModelStates
import tf2_ros
from geometry_msgs.msg import TransformStamped


class GroundTruthTF:
    def __init__(self):
        rospy.init_node("ground_truth_tf")

        # Parameters (so you can override via launch file)
        self.robot_name = rospy.get_param("~robot_name", "turtlebot3_waffle")
        self.parent_frame = rospy.get_param("~parent_frame", "map")
        self.child_frame = rospy.get_param("~child_frame", "base_link_gt")
        self.last_stamp = rospy.Time(0)

        # TF broadcaster
        self.br = tf2_ros.TransformBroadcaster()

        # Subscriber
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callback)

        rospy.loginfo(f"[GT TF] Tracking robot: {self.robot_name}")

    def callback(self, msg):
        if self.robot_name not in msg.name:
            print(f"[GT TF] Robot '{self.robot_name}' not found in ModelStates. Available models: {msg.name}")
            return

        i = msg.name.index(self.robot_name)
        pose = msg.pose[i]

        now = rospy.Time.now()

        if now == self.last_stamp:
            return

        self.last_stamp = now
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        t.transform.translation.x = pose.position.x + 0.7 
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z

        t.transform.rotation = pose.orientation

        self.br.sendTransform(t)


if __name__ == "__main__":
    try:
        node = GroundTruthTF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass