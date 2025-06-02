#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf
import rospkg
import cv2
from scipy.ndimage import distance_transform_edt
import yaml

class MCMHLocalizer:
    def __init__(self):
        rospy.init_node('mcmh_localizer')

        self.rospack = rospkg.RosPack()
        self.map_path = self.rospack.get_path('mcmh_localization') + '/maps/map.yaml'
        self.distance_map, self.resolution, self.origin = self.load_map()

        self.num_particles = 100
        self.particles = self.initialize_particles()

        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.pose_pub = rospy.Publisher('/mcmh_estimated_pose', PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/mcmh_particles', MarkerArray, queue_size=1)


        rospy.spin()

    def initialize_particles(self):
        # Inicialização simples, pode usar mapa no futuro
        particles = np.zeros((self.num_particles, 3))  # x, y, theta
        particles[:, 0] = np.random.uniform(-1, 1, self.num_particles)
        particles[:, 1] = np.random.uniform(-1, 1, self.num_particles)
        particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        return particles
    
    def load_map(self):
    # Parse o arquivo YAML
        with open(self.map_path, 'r') as f:
            map_yaml = yaml.safe_load(f)

        map_img = cv2.imread(map_yaml['image'], cv2.IMREAD_GRAYSCALE)
        map_img = cv2.threshold(map_img, 250, 1, cv2.THRESH_BINARY_INV)[1]

        resolution = map_yaml['resolution']
        origin = map_yaml['origin']

        distance_map = distance_transform_edt(map_img) * resolution  # em metros
        return distance_map, resolution, origin

    def odom_callback(self, msg):
        # Pode ser usada para mover as partículas se desejar usar o modelo de movimento
        pass

    def publish_particles(self):
        marker_array = MarkerArray()
        for i, p in enumerate(self.particles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            marker.pose.orientation.z = np.sin(p[2]/2)
            marker.pose.orientation.w = np.cos(p[2]/2)
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def lidar_callback(self, msg):
        # Aplicar Metropolis-Hastings aqui
        self.update_particles_mh(msg)
        self.publish_estimate()
        self.publish_particles()

    def update_particles_mh(self, scan):
        for i in range(self.num_particles):
            x = self.particles[i]
            x_prime = x + np.random.normal(0, 0.05, size=3)  # Proposta simétrica

            p_x = self.likelihood(scan, x)
            p_x_prime = self.likelihood(scan, x_prime)

            alpha = min(1.0, p_x_prime / p_x) if p_x > 0 else 1.0

            if np.random.rand() < alpha:
                self.particles[i] = x_prime

    def likelihood(self,scan, pose):
        x, y, theta = pose
        total_prob = 1.0
        sigma_hit = 0.2  # metros

        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        for i, r in enumerate(scan.ranges[::10]):  # Use apenas 1 a cada 10 feixes para performance
            if np.isinf(r) or np.isnan(r):
                continue

            angle = angle_min + i * 10 * angle_increment
            lx = x + r * np.cos(theta + angle)
            ly = y + r * np.sin(theta + angle)

            mx = int((lx - self.origin[0]) / self.resolution)
            my = int((ly - self.origin[1]) / self.resolution)

            if 0 <= mx < self.distance_map.shape[1] and 0 <= my < self.distance_map.shape[0]:
                dist = self.distance_map[my, mx]
                prob = np.exp(-0.5 * (dist / sigma_hit)**2)
                total_prob *= prob + 1e-9
            else:
                total_prob *= 1e-3  # penalidade para fora do mapa

        return total_prob


    def publish_estimate(self):
        mean_pose = np.mean(self.particles, axis=0)
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position.x = mean_pose[0]
        pose.pose.position.y = mean_pose[1]
        pose.pose.orientation.z = np.sin(mean_pose[2] / 2.0)
        pose.pose.orientation.w = np.cos(mean_pose[2] / 2.0)
        self.pose_pub.publish(pose)

if __name__ == '__main__':
    try:
        MCMHLocalizer()
    except rospy.ROSInterruptException:
        pass
