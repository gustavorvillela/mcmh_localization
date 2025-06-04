#!/usr/bin/env python3
import rospy
import numpy as np
from numpy.random import choice
import tf.transformations as tft
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf
import rospkg
import cv2
from scipy.ndimage import distance_transform_edt
import yaml
import tf2_ros

class MCMHLocalizer:
    def __init__(self):
        rospy.init_node('mcmh_localizer')

        self.rospack = rospkg.RosPack()
        self.map_path = self.rospack.get_path('mcmh_localization') + '/maps/map.yaml'
        self.distance_map, self.resolution, self.origin = self.load_map()

        self.num_particles = 100
        self.particles = self.initialize_particles()
        self.particles_prop = self.initialize_particles()
        self.weights = np.ones(self.num_particles) / self.num_particles

        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.pose_pub = rospy.Publisher('/mcmh_estimated_pose', PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/mcmh_particles', MarkerArray, queue_size=1)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.alpha = [0.05, 0.05, 0.02, 0.02]  # Parâmetros do ruído: α1, α2, α3, α4
        self.last_odom = None

        rospy.spin()

    def initialize_particles(self):
        # Encontrar células livres no mapa (onde distance_map > 0)
        free_y, free_x = np.where(self.distance_map > 0.5)  # Coordenadas (linha, coluna) das células livres
        
        # Verificar se há células livres
        if len(free_x) == 0 or len(free_y) == 0:
            rospy.logwarn("Nenhuma célula livre encontrada no mapa! Inicializando aleatoriamente.")
            particles = np.zeros((self.num_particles, 3))
            particles[:, 0] = np.random.uniform(-1, 1, self.num_particles)
            particles[:, 1] = np.random.uniform(-1, 1, self.num_particles)
            particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
            return particles
        
        # Selecionar aleatoriamente células livres para as partículas
        selected_indices = np.random.choice(len(free_x), size=self.num_particles, replace=True)
        
        particles = np.zeros((self.num_particles, 3))
        # Converter coordenadas do mapa para posições reais (em metros)
        particles[:, 0] = free_x[selected_indices] * self.resolution + self.origin[0]  # coordenada x
        particles[:, 1] = free_y[selected_indices] * self.resolution + self.origin[1]  # coordenada y
        particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)        # orientação
        
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


    def publish_particles(self):
        weights = self.weights
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
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
            marker.color.r = norm_weights[i]
            marker.color.g = 0.0
            marker.color.b = 1- norm_weights[i]
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            marker.pose.orientation.z = np.sin(p[2]/2)
            marker.pose.orientation.w = np.cos(p[2]/2)
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    #================================================
    # Odometry
    #================================================

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        current_odom = np.array([position.x, position.y, yaw])

        if self.last_odom is not None:
            delta = self.compute_motion(self.last_odom, current_odom)
            self.apply_motion_model(delta)

        self.last_odom = current_odom

    def compute_motion(self, odom1, odom2):
        dx = odom2[0] - odom1[0]
        dy = odom2[1] - odom1[1]
        dtheta = self.angle_diff(odom2[2], odom1[2])

        rot1 = np.arctan2(dy, dx) - odom1[2]
        trans = np.hypot(dx, dy)
        rot2 = dtheta - rot1

        return rot1, trans, rot2

    def apply_motion_model(self, delta):
        rot1, trans, rot2 = delta
        a1, a2, a3, a4 = self.alpha

        for i in range(self.num_particles):
            r1_hat = rot1 + np.random.normal(0, a1 * abs(rot1) + a2 * abs(trans))
            t_hat  = trans + np.random.normal(0, a3 * abs(trans) + a4 * (abs(rot1) + abs(rot2)))
            r2_hat = rot2 + np.random.normal(0, a1 * abs(rot2) + a2 * abs(trans))

            x, y, theta = self.particles[i]
            x_new = x + t_hat * np.cos(theta + r1_hat)
            y_new = y + t_hat * np.sin(theta + r1_hat)
            theta_new = theta + r1_hat + r2_hat

            self.particles_prop[i] = [x_new, y_new, self.normalize_angle(theta_new)]

    def angle_diff(self, a, b):
        """Retorna a diferença angular entre dois ângulos em rad."""
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    #================================================
    # LiDAR
    #================================================

    def lidar_callback(self, msg):

        self.update_particles_mh(msg)

        self.weights /= np.sum(self.weights)

        self.publish_estimate()
        self.resample_particles()
        self.publish_particles()


    def update_particles_mh(self, scan):
        for i in range(self.num_particles):
            x = self.particles[i]
            x_prime = self.particles_prop[i]  # Proposta simétrica

            p_x = self.likelihood(scan, x)
            p_x_prime = self.likelihood(scan, x_prime)

            alpha = min(1.0, p_x_prime / p_x) if p_x > 0 else 1.0

            if np.random.rand() < alpha:
                self.particles[i] = x_prime
                self.weights[i]   = p_x_prime
            else:
                self.particles[i] = x
                self.weights[i]   = p_x

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
    
    def resample_particles(self):
        """Realiza o resampling das partículas conforme os pesos"""
        # Seleciona índices com probabilidade proporcional aos pesos
        indices = choice(
            np.arange(self.num_particles), 
            size=self.num_particles, 
            p=self.weights,
            replace=True
        )
        
        # Cria novo conjunto de partículas
        new_particles = []
        for i in indices:
            new_particles.append(self.particles[i].copy())
        
        self.particles = new_particles
        #self.weights = np.ones(self.num_particles) / self.num_particles  # Reseta pesos

    
    #===============================
    #Broadcaster
    #===============================
    

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


    def compute_map_to_odom_tf(self,estimated_pose, odom_to_base):

        # Pose estimada do robô (em map)
        t_map_base = tft.translation_matrix([
            estimated_pose.pose.position.x,
            estimated_pose.pose.position.y,
            0.0
        ])
        r_map_base = tft.quaternion_matrix([
            estimated_pose.pose.orientation.x,
            estimated_pose.pose.orientation.y,
            estimated_pose.pose.orientation.z,
            estimated_pose.pose.orientation.w
        ])
        T_map_base = np.dot(t_map_base, r_map_base)

        # Transformação odom -> base
        t_odom_base = tft.translation_matrix([
            odom_to_base.transform.translation.x,
            odom_to_base.transform.translation.y,
            0.0
        ])
        r_odom_base = tft.quaternion_matrix([
            odom_to_base.transform.rotation.x,
            odom_to_base.transform.rotation.y,
            odom_to_base.transform.rotation.z,
            odom_to_base.transform.rotation.w
        ])
        T_odom_base = np.dot(t_odom_base, r_odom_base)

        # T_map_odom = T_map_base * inv(T_odom_base)
        T_map_odom = np.dot(T_map_base, np.linalg.inv(T_odom_base))
        trans = tft.translation_from_matrix(T_map_odom)
        rot = tft.quaternion_from_matrix(T_map_odom)

        return trans, rot

    def get_odom_to_base(self):
        try:
            now = rospy.Time.now()
            self.tf_buffer.can_transform("odom", "base_footprint", now, rospy.Duration(1.0))  # opcional para checar
            transform = self.tf_buffer.lookup_transform("odom", "base_footprint", rospy.Time(0), rospy.Duration(1.0))
            return transform
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return None

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
        odom_to_base = self.get_odom_to_base()
        trans, rot = self.compute_map_to_odom_tf(pose,odom_to_base)
        self.broadcast_transform(trans,rot)

if __name__ == '__main__':
    try:
        MCMHLocalizer()
    except rospy.ROSInterruptException:
        pass
