#!/usr/bin/env python3
import rospy
import numpy as np
from numpy.random import choice
import tf.transformations as tft
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt


class PoseBroadcaster:

    def __init__(self):

        rospy.init_node('pose_broadcaster')

        #Subscriber
        rospy.Subscriber('/mcmh_estimated_pose', PoseWithCovarianceStamped, self.pose_callback)

        # TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.last_tf_stamp = rospy.Time(0)
        
        rospy.spin()

    def pose_callback(self,pose):

        stamp = pose.header.stamp
        if stamp == rospy.Time(0):
            stamp = rospy.Time.now()

        odom_to_base = self.get_odom_to_base(stamp)
        if odom_to_base is None:
            rospy.logwarn_throttle(5.0, "Skipping map->odom broadcast: odom->base_footprint transform unavailable")
            return

        trans, rot = self.compute_map_to_odom_tf(pose.pose,odom_to_base)
        self.broadcast_transform(trans,rot,stamp)

    def get_odom_to_base(self, stamp):
        try:
            return self.tf_buffer.lookup_transform("odom", "base_footprint", stamp, rospy.Duration(0.05))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            try:
                return self.tf_buffer.lookup_transform("odom", "base_footprint", rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
                return None
        
    def compute_map_to_odom_tf(self, estimated_pose, odom_to_base):
        # 1. T_map_base (from estimated pose)
        q_map_base = np.array([
            estimated_pose.pose.orientation.x,
            estimated_pose.pose.orientation.y,
            estimated_pose.pose.orientation.z,
            estimated_pose.pose.orientation.w
        ])
        q_map_base /= np.linalg.norm(q_map_base)

        T_map_base = tft.quaternion_matrix(q_map_base)
        T_map_base[0:3, 3] = [
            estimated_pose.pose.position.x,
            estimated_pose.pose.position.y,
            0.0
        ]

        # 2. T_odom_base (from odom)
        q_odom_base = np.array([
            odom_to_base.transform.rotation.x,
            odom_to_base.transform.rotation.y,
            odom_to_base.transform.rotation.z,
            odom_to_base.transform.rotation.w
        ])
        q_odom_base /= np.linalg.norm(q_odom_base)

        T_odom_base = tft.quaternion_matrix(q_odom_base)
        T_odom_base[0:3, 3] = [
            odom_to_base.transform.translation.x,
            odom_to_base.transform.translation.y,
            0.0
        ]

        # 3. Compute T_map_odom
        T_map_odom = np.dot(T_map_base, np.linalg.inv(T_odom_base))
        trans = tft.translation_from_matrix(T_map_odom)
        rot = tft.quaternion_from_matrix(T_map_odom)

        # Normalize quaternion and force w positive (to avoid flipping)
        rot /= np.linalg.norm(rot)
        if rot[3] < 0:
            rot = -rot

        return trans, rot
    
    def broadcast_transform(self,trans, rot, stamp):

        if stamp <= self.last_tf_stamp:
            rospy.logdebug("Skipping map->odom broadcast with non-increasing stamp %.6f", stamp.to_sec())
            return

        self.last_tf_stamp = stamp

        t = TransformStamped()
        t.header.stamp = stamp
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
    
