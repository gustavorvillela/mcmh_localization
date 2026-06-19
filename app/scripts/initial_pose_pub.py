#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

class InitialPosePublisher:
    def __init__(self):
        rospy.init_node('initial_pose_publisher')
        
        # Configurable parameters
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)  # Hz
        self.initial_x = rospy.get_param('~x', -2.0)
        self.initial_y = rospy.get_param('~y', -0.5)
        self.initial_yaw = rospy.get_param('~yaw', 0.0)  # radianos
        self.covariance = rospy.get_param('~covariance', [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                         0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0685])
        
        self.publisher = rospy.Publisher('/initial_pose', PoseWithCovarianceStamped, queue_size=10,latch=True)
        
        rospy.loginfo(f"Initializing initial pose publisher at ({self.initial_x}, {self.initial_y}, {np.degrees(self.initial_yaw):.1f}°)")
        
    def publish_initial_pose(self):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        
        # Setting the position
        msg.pose.pose.position.x = self.initial_x
        msg.pose.pose.position.y = self.initial_y
        msg.pose.pose.position.z = 0.0
        
        # Converting yaw to quaternion
        cy = np.cos(self.initial_yaw * 0.5)
        sy = np.sin(self.initial_yaw * 0.5)
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = sy
        msg.pose.pose.orientation.w = cy
        
        # Setting covariance
        msg.pose.covariance = self.covariance
        
        self.publisher.publish(msg)
        
    def run(self):
        #rate = rospy.Rate(self.publish_rate)
        #while not rospy.is_shutdown():
        #    self.publish_initial_pose()
        #    rate.sleep()

        # Publica apenas uma vez
        self.publish_initial_pose()
        rospy.loginfo(f"Initial pose published!")
        # Keep the node alive so that the latched topic remains available
        rospy.spin()        

if __name__ == '__main__':
    try:
        publisher = InitialPosePublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass