#!/usr/bin/env python3
import rospy
import numpy as np
from numpy.random import choice
import tf.transformations as tft
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt


class PoseBroadcaster:

    def __init__(self):

        rospy.init_node('pose_broadcaster')

        #Subscriber
        rospy.Subscriber('/mcmh_estimated_pose', PoseStamped, self.pose_callback)

        # TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.spin()

    def pose_callback(self,pose):

        odom_to_base = self.get_odom_to_base()
        trans, rot = self.compute_map_to_odom_tf(pose,odom_to_base)
        self.broadcast_transform(trans,rot)

    def get_odom_to_base(self):
        try:
            return self.tf_buffer.lookup_transform("odom", "base_footprint", rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return None
        
    def compute_map_to_odom_tf(self,estimated_pose, odom_to_base):

        # 1. Obter T_map_base (pose estimada)
        T_map_base = tft.quaternion_matrix([
            estimated_pose.pose.orientation.x,
            estimated_pose.pose.orientation.y,
            estimated_pose.pose.orientation.z,
            estimated_pose.pose.orientation.w
        ])
        T_map_base[0:3, 3] = [
            estimated_pose.pose.position.x,
            estimated_pose.pose.position.y,
            0.0
        ]

        # 2. Obter T_odom_base (da odometria)
        T_odom_base = tft.quaternion_matrix([
            odom_to_base.transform.rotation.x,
            odom_to_base.transform.rotation.y,
            odom_to_base.transform.rotation.z,
            odom_to_base.transform.rotation.w
        ])
        T_odom_base[0:3, 3] = [
            odom_to_base.transform.translation.x,
            odom_to_base.transform.translation.y,
            0.0
        ]

        # T_map_odom = T_map_base * inv(T_odom_base)
        T_map_odom = np.dot(T_map_base, np.linalg.inv(T_odom_base))
        trans = tft.translation_from_matrix(T_map_odom)
        rot = tft.quaternion_from_matrix(T_map_odom)

        return trans, rot
    
    def broadcast_transform(self,trans, rot):

        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = 0.0
        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]


        self.tf_broadcaster.sendTransform(t)

    

    
        

if __name__ == '__main__':
    try:
        PoseBroadcaster()
    except rospy.ROSInterruptException:
        pass
    
