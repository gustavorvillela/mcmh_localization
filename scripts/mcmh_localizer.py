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

class MCMHLocalizer:
    def __init__(self):
        rospy.init_node('mcmh_localizer')

        # Parâmetros
        self.num_particles = 2000
        self.alpha = [0.4, 0.4, 0.2, 0.2]
        
        # Carrega o mapa
        self.load_map()
        
        # Inicializa partículas
        self.particles = self.initialize_particles()
        self.particles_prop = np.copy(self.particles)
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Subscribers
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Publishers
        self.pose_pub = rospy.Publisher('/mcmh_estimated_pose', PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/mcmh_particles', MarkerArray, queue_size=1)
        
        # TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.last_odom = None
        rospy.spin()

    def load_map(self):
        map_msg = rospy.wait_for_message("/map", OccupancyGrid)
        
        self.map_data = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
        self.resolution = map_msg.info.resolution
        self.origin = map_msg.info.origin.position
        
        # Processa células livres
        free_cells = np.where(self.map_data == 0)
        self.free_cells_coords = np.column_stack((
            free_cells[1] * self.resolution + self.origin.x,
            (self.map_data.shape[0] - free_cells[0] - 1) * self.resolution + self.origin.y
        ))
        
        self.kdtree = KDTree(self.free_cells_coords)
        self.distance_map = distance_transform_edt(self.map_data == 0) * self.resolution

    def initialize_particles(self):
        # Tenta primeiro com amostragem direta (rápida)
        try_indices = np.random.choice(len(self.free_cells_coords), 
                                    size=self.num_particles*2, replace=False)
        candidates = self.free_cells_coords[try_indices]
        
        # Se tiver suficientes, pega as primeiras
        if len(candidates) >= self.num_particles:
            particles = candidates[:self.num_particles]
        else:
            # Complementa com amostragem uniforme no espaço
            remaining = self.num_particles - len(candidates)
            min_coords = np.min(self.free_cells_coords, axis=0)
            max_coords = np.max(self.free_cells_coords, axis=0)
            
            extra_particles = []
            while len(extra_particles) < remaining:
                x = np.random.uniform(min_coords[0], max_coords[0])
                y = np.random.uniform(min_coords[1], max_coords[1])
                if self.is_valid_position(x, y):
                    extra_particles.append([x, y])
            
            particles = np.vstack([candidates, extra_particles])
        
        # Adiciona orientações
        final_particles = np.zeros((self.num_particles, 3))
        final_particles[:, :2] = particles[:self.num_particles]
        final_particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        
        return final_particles
    
    def world_to_map(self, x, y):
        mx = int((x - self.origin.x) / self.resolution)
        my = int((y - self.origin.y) / self.resolution)
        return mx, my

    def is_valid_position(self, x, y):
        mx, my = self.world_to_map(x, y)
        if not (0 <= mx < self.map_data.shape[1] and 0 <= my < self.map_data.shape[0]):
            return False
        return self.map_data[my, mx] == 0
    
    def get_nearest_valid(self, x, y):
        """Versão corrigida que retorna coordenadas x,y diretamente"""
        _, idx = self.kdtree.query([x, y])
        nearest = self.free_cells_coords[idx]
        return nearest[0], nearest[1]  # Retorna x,y diretamente

    def apply_motion_model(self, delta):
        rot1, trans, rot2 = delta
        a1, a2, a3, a4 = self.alpha

        for i in range(self.num_particles):
            r1_hat = rot1 + np.random.normal(0, a1*abs(rot1) + a2*abs(trans))
            t_hat = trans + np.random.normal(0, a3*abs(trans) + a4*(abs(rot1)+abs(rot2)))
            r2_hat = rot2 + np.random.normal(0, a1*abs(rot2) + a2*abs(trans))

            x, y, theta = self.particles[i]
            x_new = x + t_hat * np.cos(theta + r1_hat)
            y_new = y + t_hat * np.sin(theta + r1_hat)
            theta_new = theta + r1_hat + r2_hat

            if not self.is_valid_position(x_new, y_new):
                x_new, y_new = self.get_nearest_valid(x_new, y_new)
            
            self.particles_prop[i] = [x_new, y_new, self.normalize_angle(theta_new)]

    def update_particles_mh(self, scan):
        for i in range(self.num_particles):
            x = self.particles[i]
            x_prime = self.particles_prop[i]
            
            if not self.is_valid_position(x_prime[0], x_prime[1]):
                x_prime[0], x_prime[1] = self.get_nearest_valid(x_prime[0], x_prime[1])
                x_prime[2] = self.normalize_angle(x_prime[2])
            
            p_x = self.likelihood(scan, x)
            p_x_prime = self.likelihood(scan, x_prime)

            alpha = min(1.0, p_x_prime / p_x) if p_x > 0 else 1.0
            #alpha = 1

            if np.random.rand() < alpha:
                self.particles[i] = x_prime
                self.weights[i] = p_x_prime

    def resample_valid_particles(self):
        valid_indices = [i for i in range(self.num_particles) 
                        if self.is_valid_position(*self.particles[i,:2])]
        
        if not valid_indices:
            rospy.logwarn("Reinicializando partículas!")
            self.particles = self.initialize_particles()
            self.weights = np.ones(self.num_particles) / self.num_particles
            return

        valid_weights = self.weights[valid_indices]
        valid_weights /= np.sum(valid_weights)
        
        new_indices = choice(valid_indices, size=self.num_particles, p=valid_weights)
        self.particles = self.particles[new_indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def likelihood(self, scan, pose):
        x, y, theta = pose
        log_prob = 0.0
        sigma_hit = 0.2

        for i, r in enumerate(scan.ranges[::10]):
            if r < scan.range_min or r > scan.range_max:
                continue

            angle = scan.angle_min + i*10*scan.angle_increment
            lx = x + r * np.cos(theta + angle)
            ly = y + r * np.sin(theta + angle)

            mx = int((lx - self.origin.x) / self.resolution)
            my = int((self.map_data.shape[0] - (ly - self.origin.y))/self.resolution) - 1

            if 0 <= mx < self.distance_map.shape[1] and 0 <= my < self.distance_map.shape[0]:
                dist = self.distance_map[my, mx]
                prob = np.exp(-0.5 * (dist/sigma_hit)**2) + 1e-9
                log_prob += np.log(prob)
            else:
                log_prob += np.log(1e-3)

        return np.exp(log_prob)

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2*np.pi) - np.pi

    def publish_particles(self):
        marker_array = MarkerArray()
        weights = self.weights
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
        
        for i, p in enumerate(self.particles):
            if not self.is_valid_position(p[0], p[1]):
                continue
                
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
            marker.color.b = 1 - norm_weights[i]
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            marker.pose.orientation.z = np.sin(p[2]/2)
            marker.pose.orientation.w = np.cos(p[2]/2)
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([orientation.x, orientation.y, 
                                              orientation.z, orientation.w])

        current_odom = np.array([position.x, position.y, yaw])

        if self.last_odom is not None:
            delta = self.compute_motion(self.last_odom, current_odom)
            self.apply_motion_model(delta)

        self.last_odom = current_odom

    def compute_motion(self, odom1, odom2):
        dx = odom2[0] - odom1[0]
        dy = odom2[1] - odom1[1]
        dtheta = self.normalize_angle(odom2[2] - odom1[2])

        rot1 = np.arctan2(dy, dx) - odom1[2]
        trans = np.hypot(dx, dy)
        rot2 = dtheta - rot1

        return rot1, trans, rot2

    def lidar_callback(self, msg):
        self.update_particles_mh(msg)
        self.publish_estimate()
        self.resample_valid_particles()
        self.publish_particles()


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

if __name__ == '__main__':
    try:
        MCMHLocalizer()
    except rospy.ROSInterruptException:
        pass
    
