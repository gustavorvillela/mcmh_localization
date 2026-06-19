#!/usr/bin/env python3

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class ParticleMarkerPublisher:
    def __init__(self):
        rospy.init_node('particle_generator', anonymous=True)
        
        # Parameters
        self.num_particles = rospy.get_param('~num_particles', 1000)
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)
        self.particle_scale = rospy.get_param('~particle_scale', 0.1)  # Particle size
        self.particle_color = rospy.get_param('~particle_color', [1.0, 0.0, 0.0, 1.0])  # RGBA
        
        # Map variables
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        
        # Publishers
        self.marker_pub = rospy.Publisher('/particle_markers', MarkerArray, queue_size=10)
        
        # Timer
        rospy.Timer(rospy.Duration(1.0/self.publish_rate), self.publish_markers)
        
    def map_callback(self, msg):
        """Callback to receive map data"""
        self.map_data = msg.data
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        rospy.loginfo("Map received. Resolution: %f, Dimensions: %d x %d", 
                     self.map_resolution, self.map_width, self.map_height)
    
    def is_valid_position(self, x, y):
        """Check if the position is in the free space of the map"""
        if not self.map_data:
            return False
            
        # Convert world coordinates to map coordinates
        map_x = int((x - self.map_origin.x) / self.map_resolution)
        map_y = int((y - self.map_origin.y) / self.map_resolution)
        
        # Check if it is within map limits
        if map_x < 0 or map_x >= self.map_width or map_y < 0 or map_y >= self.map_height:
            return False
            
        # Check if it is free space (0 in occupancy grid)
        index = map_y * self.map_width + map_x
        return self.map_data[index] == 0
    
    def generate_particle_markers(self):
        """Generate markers for random particles within the free space of the map"""
        marker_array = MarkerArray()
        
        if not self.map_data:
            rospy.logwarn("Map not yet received. Unable to generate particles.")
            return marker_array
            
        # Create a single marker for all particles (more efficient than one marker per particle)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "particles"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Marker configuration
        marker.scale.x = self.particle_scale  # Point size in x
        marker.scale.y = self.particle_scale  # Point size in y
        marker.color.r = self.particle_color[0]
        marker.color.g = self.particle_color[1]
        marker.color.b = self.particle_color[2]
        marker.color.a = self.particle_color[3]
        
        count = 0
        while count < self.num_particles:
            # Generate random coordinates
            x = np.random.uniform(self.map_origin.x, 
                                self.map_origin.x + self.map_width * self.map_resolution)
            y = np.random.uniform(self.map_origin.y, 
                                self.map_origin.y + self.map_height * self.map_resolution)
            
            if self.is_valid_position(x, y):
                point = Point()
                point.x = x
                point.y = y
                point.z = 0  # On the 2D plane
                marker.points.append(point)
                count += 1
        
        marker_array.markers.append(marker)
        
        # Add orientation markers (optional)
        if True:  # Change to True if you want arrows showing orientation
            for i in range(len(marker.points)):
                arrow_marker = Marker()
                arrow_marker.header = marker.header
                arrow_marker.ns = "particle_arrows"
                arrow_marker.id = i + 1
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                
                # Random position and orientation
                yaw = np.random.uniform(0, 2*np.pi)
                quat = quaternion_from_euler(0, 0, yaw)
                
                arrow_marker.pose.position = marker.points[i]
                arrow_marker.pose.orientation.x = quat[0]
                arrow_marker.pose.orientation.y = quat[1]
                arrow_marker.pose.orientation.z = quat[2]
                arrow_marker.pose.orientation.w = quat[3]
                
                arrow_marker.scale.x = self.particle_scale * 2  # Length
                arrow_marker.scale.y = self.particle_scale * 0.5  # Width
                arrow_marker.scale.z = self.particle_scale * 0.5  # Height
                arrow_marker.color.r = 0.0
                arrow_marker.color.g = 1.0
                arrow_marker.color.b = 0.0
                arrow_marker.color.a = 1.0
                
                marker_array.markers.append(arrow_marker)
        
        return marker_array
    
    def publish_markers(self, event):
        """Publish particle markers"""
        marker_array = self.generate_particle_markers()
        self.marker_pub.publish(marker_array)
        rospy.loginfo("Published %d particle markers", self.num_particles)

if __name__ == '__main__':
    try:
        pmp = ParticleMarkerPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass