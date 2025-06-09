#!/usr/bin/env python3
import rospy
import numpy as np
from numpy.random import choice
import tf.transformations as tft
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from parallel_utils import compute_likelihoods, mh_resampling, apply_motion_model_parallel, normalize_angle, compute_valid_indices, generate_valid_particles, low_variance_resample_numba, normalize_angle_array, parallel_resample_simple

class MCMHLocalizer:
    def __init__(self):
        rospy.init_node('mcmh_localizer')

        # Parâmetros
        self.num_particles = 5000
        self.alpha = np.array([0.1, 0.1, 0.15, 0.15], dtype=np.float32)
        # Carrega o mapa
        self.load_map()
        
        # Inicializa partículas
        self.particles = self.initialize_particles()
        self.particles_prop = np.copy(self.particles)
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.weights_viz = self.weights.copy()

        self.last_odom = None
        
        # Subscribers
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Publishers
        self.pose_pub = rospy.Publisher('/mcmh_estimated_pose', PoseWithCovarianceStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/mcmh_particles', MarkerArray, queue_size=1)
        
        # TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        
        rospy.spin()

    def load_map(self):
        map_msg = rospy.wait_for_message("/map", OccupancyGrid)
        
        self.map_data = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
        self.resolution = map_msg.info.resolution
        self.origin = map_msg.info.origin.position
        self.origin_np = np.array([self.origin.x, self.origin.y])

        self.occupancy_map = np.where(self.map_data == 0, 0, 100)
        
        # Processa células livres
        free_cells = np.where(self.map_data == 0)
        self.free_cells_coords = np.column_stack((
            free_cells[1] * self.resolution + self.origin.x,
            (self.map_data.shape[0] - free_cells[0] - 1) * self.resolution + self.origin.y
        ))
        
        self.kdtree = KDTree(self.free_cells_coords)
        self.distance_map = distance_transform_edt(self.map_data == 0) * self.resolution

    def initialize_particles(self):

        min_coords = np.min(self.free_cells_coords, axis=0)
        max_coords = np.max(self.free_cells_coords, axis=0)

        final_particles = generate_valid_particles(self.num_particles, min_coords, max_coords,
                                             self.occupancy_map, self.resolution, 
                                             self.origin_np[0], self.origin_np[1])

        
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



    def get_lidar_angles(self, scan):
        num_ranges = len(scan.ranges)
        return np.linspace(scan.angle_min, scan.angle_max, num_ranges, dtype=np.float32)
    
    def convert_scores(self,scores):

        max_score = np.max(scores)
        weights = np.zeros_like(scores)
        weights = np.exp(scores - max_score)
        weights =  weights/np.sum(weights)

        return weights




    def update_particles_mh(self, scan):

        scan_ranges = np.array(scan.ranges, dtype=np.float32)
        angles = self.get_lidar_angles(scan)
        

        scores_pre = compute_likelihoods(
        scan_ranges, angles, self.particles,
        self.distance_map, self.resolution, self.origin_np
        )

        weights_pre = self.convert_scores(scores_pre)

        scores_post = compute_likelihoods(
        scan_ranges, angles, self.particles_prop,
        self.distance_map, self.resolution, self.origin_np
        )

        weights_post = self.convert_scores(scores_post)

        
        self.particles, self.weights = mh_resampling(self.particles,self.particles_prop,weights_post,weights_pre)
        self.weights_viz = self.weights.copy()

        #print("Peso máximo:", np.max(self.weights))
        #print("Peso médio:", np.mean(self.weights))
        #print("Número de pesos > 1e-3:", np.sum(self.weights > 1e-3))


    def resample_simple(self):

        resampled_particles = parallel_resample_simple(self.particles,self.weights,N=self.num_particles)

        self.particles = resampled_particles



    def resample_valid_particles_lvr(self):

        valid_indices = compute_valid_indices(
            self.particles,
            self.occupancy_map,       # 2D numpy array do mapa
            self.resolution,
            self.origin_np[0],
            self.origin_np[1]
        )

        if len(valid_indices) == 0:
            rospy.logwarn("Reinicializando partículas!")
            self.particles = self.initialize_particles()
            return
        
        valid_particles = self.particles[valid_indices]
        valid_weights = self.weights[valid_indices]
        valid_weights /= np.sum(valid_weights)

        if np.sum(valid_weights) < 1e-6:
            rospy.logwarn("Pesos degeneraram! Reinicializando partículas.")
            self.particles = self.initialize_particles()
            self.weights.fill(1.0 / self.num_particles)
            return
        
        if valid_particles.shape[0] < self.num_particles:
            rospy.logwarn(f"Apenas {valid_particles.shape[0]} partículas válidas, fazendo oversampling com LVR.")
            # Ajusta pesos e partículas para repetir via LVR
            expanded_particles = np.repeat(valid_particles, repeats=(self.num_particles // valid_particles.shape[0]) + 1, axis=0)
            expanded_weights = np.repeat(valid_weights, repeats=(self.num_particles // valid_particles.shape[0]) + 1)
            expanded_weights = expanded_weights[:expanded_particles.shape[0]]
            expanded_weights /= np.sum(expanded_weights)

            resampled_particles, resampled_weights = low_variance_resample_numba(
                expanded_particles[:self.num_particles],
                expanded_weights[:self.num_particles]
            )

            self.particles = resampled_particles
            self.weights = resampled_weights

        else:

            resampled_particles, resampled_weights = low_variance_resample_numba(
                self.particles[valid_indices],
                valid_weights
            )

        self.particles = resampled_particles
        self.weights = resampled_weights


    def publish_particles(self):
        marker_array = MarkerArray()
        weights = self.weights_viz
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)

        cos_half_theta = np.cos(self.particles[:,2] / 2.0)
        sin_half_theta = np.sin(self.particles[:,2] / 2.0)
        
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
            marker.pose.orientation.z = sin_half_theta[i]
            marker.pose.orientation.w = cos_half_theta[i]
            
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
            self.particles_prop = apply_motion_model_parallel(self.particles,delta,self.alpha,
                                                              self.occupancy_map, self.resolution,
                                                              self.origin_np[0], self.origin_np[1])

        self.last_odom = current_odom

    def compute_motion(self, odom1, odom2):
        dx = odom2[0] - odom1[0]
        dy = odom2[1] - odom1[1]
        dtheta = normalize_angle(odom2[2] - odom1[2])

        rot1 = np.arctan2(dy, dx) - odom1[2]
        trans = np.hypot(dx, dy)
        rot2 = dtheta - rot1

        return rot1, trans, rot2

    def lidar_callback(self, msg):
        self.update_particles_mh(msg)
        self.publish_estimate()
        self.resample_simple()
        self.publish_particles()

    
    def publish_estimate(self):

        mean_pose = np.average(self.particles, axis=0,weights=self.weights)
        diffs = self.particles.copy()
        diffs[:, 0] -= mean_pose[0]
        diffs[:, 1] -= mean_pose[1]
        diffs[:, 2] = normalize_angle_array(self.particles[:, 2], mean_pose[2])
        cov = np.cov(diffs.T, aweights=self.weights)
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.pose.position.x = mean_pose[0]
        pose.pose.pose.position.y = mean_pose[1]
        pose.pose.pose.orientation.z = np.sin(mean_pose[2] / 2.0)
        pose.pose.pose.orientation.w = np.cos(mean_pose[2] / 2.0)

        # Preenche a matriz de covariância (6x6 flatten)
        # Usamos apenas as dimensões x, y, theta → [0,0], [1,1], [5,5]
        cov_flat = np.zeros(36)
        cov_flat[0] = cov[0, 0]           # x-x
        cov_flat[1] = cov[0, 1]           # x-y
        cov_flat[5] = cov[0, 2]           # x-theta

        cov_flat[6] = cov[1, 0]           # y-x
        cov_flat[7] = cov[1, 1]           # y-y
        cov_flat[11] = cov[1, 2]          # y-theta

        cov_flat[30] = cov[2, 0]          # theta-x
        cov_flat[31] = cov[2, 1]          # theta-y
        cov_flat[35] = cov[2, 2]          # theta-theta

        pose.pose.covariance = cov_flat.tolist()
        self.pose_pub.publish(pose)


if __name__ == '__main__':
    try:
        MCMHLocalizer()
    except rospy.ROSInterruptException:
        pass
    
