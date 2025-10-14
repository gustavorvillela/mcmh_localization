#!/usr/bin/env python3
import rospy
import numpy as np
from numpy.random import choice
import tf.transformations as tft
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from parallel_utils import compute_likelihoods, mh_resampling, apply_motion_model_parallel, normalize_angle, compute_valid_indices, generate_valid_particles, low_variance_resample_numba, normalize_angle_array, kld_sampling_amcl, initialize_gaussian_parallel,parallel_resample_simple,compute_likelihoods_raycast, assym_mh_resampling, motion_model_odometry_parallel

class AMCMHLocalizer:
    def __init__(self):
        rospy.init_node('mcmh_localizer')
        self.mode = rospy.get_param('localization_mode', 'MHAMCL')  # padrão: MHAMCL
        self.use_mh = 'MH' in self.mode
        self.use_adaptive = 'AMCL' in self.mode  # AMCL ou MHAMCL usam KLD
        self.assym = 'AMH' in self.mode  # MHAMCL usa transição assimétrica

        rospy.loginfo(f"Modo de localização: {self.mode} | MH: {self.use_mh}, Augmented: {self.use_adaptive},  Assymetric: {self.assym}")


        # Parâmetros gerais
        self.num_particles = rospy.get_param('init_particles', 2000) # do not touch
        self.alpha = np.array([
                                rospy.get_param('alpha1', 0.2),
                                rospy.get_param('alpha2', 0.2),
                                rospy.get_param('alpha3', 0.2),
                                rospy.get_param('alpha4', 0.2)
                            ], dtype=np.float32) #do not touch
        self.alpha_slow = rospy.get_param('alpha_slow', 0.01) # taxa de aprendizado lenta
        self.alpha_fast = rospy.get_param('alpha_fast', 0.1)  # taxa de aprendizado rápida

        self.dt = 0.02 #intervalo de tempo do scan

        self.delta = (0.0, 0.0, 0.0)  # (rot1, trans, rot2)

        # Parâmetros KLD
        self.kld_epsilon = rospy.get_param('kld_epsilon', 0.025)
        self.kld_delta = rospy.get_param('kld_delta', 0.99)
        self.kld_bin_size_xy = rospy.get_param('kld_bin_size_xy', 0.1)  # metros
        self.kld_bin_size_theta = rospy.get_param('kld_bin_size_theta', np.deg2rad(10))  # radianos
        self.kld_n_max = self.num_particles
        self.kld_z = rospy.get_param('kld_z', 2)
        

        self.initial_pose = None  # Armazenará a pose inicial [x, y, theta]
        self.initial_cov = np.diag([0.05, 0.05, 0.1])  # Covariância inicial (x, y em metros, theta em rad)
        self.initialized = rospy.get_param('initialized', False)  # Flag para controle

        self.sigma_hit = rospy.get_param('sigma_hit', 0.2)  # Parâmetro para a função de probabilidade gaussiana
        self.max_range = rospy.get_param('max_range', 10.0)  # Alcance máximo do LiDAR para considerar (em metros)
        self.z_hit = rospy.get_param('z_hit', 0.8)  # Peso para a parte "hit"
        self.z_rand = rospy.get_param('z_rand', 0.2)  # Peso para a parte "random"
        self.step = rospy.get_param('step', 1)  # Usar cada 'step' medidas do LiDAR para acelerar

        self.timeout = 10

        if self.initialized == True:
            rospy.loginfo("Aguardando pose inicial (máx. %.1fs)..." % self.timeout)

            # Primeiro verifica se o tópico existe
            try:
                rospy.wait_for_message('/initial_pose', PoseWithCovarianceStamped, timeout=10.0)
            except rospy.ROSException:
                rospy.logwarn("Tópico /initial_pose não encontrado. Verifique se o publisher está ativo.")
                pass

            

            try:
                msg = rospy.wait_for_message('/initial_pose', PoseWithCovarianceStamped, timeout=10.0)
                self.initial_pose_callback(msg)
            except:
                pass
        
        else:
            rospy.loginfo("Inicializando partículas uniformemente no mapa")

        #AMCL
        self.min_particles = self.min_particles = rospy.get_param('min_particles', 100)
        self.max_particles = self.max_particles = rospy.get_param('max_particles', 5000)
        self.w_slow = 1e-3
        self.w_fast = 1e-6
 
        # Carrega o mapa
        
        
        self.load_map()

        # Inicializa partículas
        self.particles = self.initialize_particles()
        self.particles_prop = np.copy(self.particles)
        self.particles_prev = np.copy(self.particles_prop)
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
        
    #======================================================================
    # Map 
    #======================================================================

    def load_map(self):
        # wait for /map once
        map_msg = rospy.wait_for_message("/map", OccupancyGrid)

        # basic params
        width = map_msg.info.width
        height = map_msg.info.height
        resolution = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        # 2D map in row-major C order (shape = (height, width))
        map_2d = np.array(map_msg.data, dtype=np.int8).reshape((height, width))

        # --- IMPORTANT: do NOT flip here. Keep map_2d exactly as ROS provides ---
        # If you previously flipped, revert that. We will use world->grid index
        # formula (mx,my) -> index = my * width + mx consistent with this layout.

        # store members
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = map_msg.info.origin.position
        self.origin_np = np.array([origin_x, origin_y])

        # flattened 1D map for fast indexing (same order as map_msg.data)
        self.map_data = map_2d.flatten()       # dtype int8

        # occupancy_map (binary: 0 free, 1 occupied) for distance transform
        occupancy_binary = (map_2d != 0).astype(np.uint8)  # occupied=1, free=0

        rospy.loginfo("Gerando mapa de distância...")
        dist_2d = distance_transform_edt(occupancy_binary == 0) * resolution
        self.distance_map = dist_2d.flatten().astype(np.float32)
        rospy.loginfo("Mapa de distância gerado.")

        # free cell coordinates in world frame (consistent with map_2d ordering)
        free_rows, free_cols = np.where(map_2d == 0)  # row=y_index, col=x_index
        # world coords of cell centers
        xs = origin_x + (free_cols + 0.5) * resolution
        ys = origin_y + (free_rows + 0.5) * resolution
        self.free_cells_coords = np.column_stack((xs, ys))

        # Save limits
        self.limits = np.array([
            origin_x,
            origin_x + width * resolution,
            origin_y,
            origin_y + height * resolution
        ])

        # Keep typed references for Numba calls (1D arrays)
        self.map_data = self.map_data.astype(np.int8)
        self.distance_map = self.distance_map.astype(np.float32)

    def initialize_particles(self):

        if self.initialized == True:
            rospy.loginfo("Inicializando partículas com distribuição gaussiana")
            final_particles = initialize_gaussian_parallel(self.initial_pose,self.initial_cov,self.num_particles,
                                                           self.distance_map,self.resolution,self.origin_np)
            
        else:
            #rospy.loginfo("Inicializando partículas uniformemente no mapa")
            final_particles = generate_valid_particles(self.num_particles,
                                             self.map_data, self.resolution,
                                             self.origin_np[0], self.origin_np[1], self.width, self.height)

        print(f"[DEBUG] Generated {final_particles.shape[0]} valid particles")

        if final_particles.shape[0] == 0:
            rospy.logerr("No valid particles generated! Check map indexing and limits.")
        
        return final_particles
    
    def initial_pose_callback(self, msg):
        """Callback para receber a pose inicial (geometry_msgs/PoseWithCovarianceStamped)"""
        pose = msg.pose.pose
        self.initial_pose = np.array([
            pose.position.x,
            pose.position.y,
            self.get_yaw_from_quaternion(pose.orientation)  # Implemente esta função
        ])
        self.initialized = True
        rospy.loginfo(f"Pose inicial recebida: {self.initial_pose}")

    def wait_for_initial_pose(self, timeout=5.0):
        """
        Espera a pose inicial por um tempo (em segundos). 
        Se não receber dentro do tempo, segue com inicialização uniforme.
        """
        rospy.loginfo("Aguardando pose inicial (máx. %.1fs)..." % timeout)
        start_time = rospy.Time.now().to_sec()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.initialized:
                rospy.loginfo("Pose inicial recebida.")
                break
            if rospy.Time.now().to_sec() - start_time > timeout:
                rospy.logwarn("Timeout: pose inicial não recebida. Inicializando uniformemente.")
                break
            rate.sleep()


    def get_yaw_from_quaternion(self, quat):
        """Converte quaternion para ângulo yaw (em radianos)"""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return yaw
    
    def world_to_map(self, x, y):
        mx = int((x - self.origin.x) / self.resolution)
        my = int((y - self.origin.y) / self.resolution)
        return mx, my

    def is_valid_position(self, x, y):
        mx, my = self.world_to_map(x, y)
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return False
        index = my * self.width + mx
        return self.map_data[index] == 0

    #======================================================================
    # Weights
    #======================================================================


    def update_weights(self):

        scores_pre = compute_likelihoods(
        self.scan_ranges, self.angles, self.particles_prev,
        self.distance_map, self.resolution, self.origin_np,
        self.width, self.height,self.sigma_hit,
        self.z_hit, self.z_rand, self.max_range, self.step
        )

        weights_pre = self.convert_scores(scores_pre)

        scores_post = compute_likelihoods(
        self.scan_ranges, self.angles, self.particles,
        self.distance_map, self.resolution, self.origin_np,
        self.width, self.height,self.sigma_hit,
        self.z_hit, self.z_rand, self.max_range, self.step
        )

        weights_post = self.convert_scores(scores_post)


        return weights_pre, weights_post
    

    def update_acml_weights(self,weights):

        self.weights = weights/np.sum(weights)

        alpha_slow_eff = 1 - (1 - self.alpha_slow) ** self.dt
        alpha_fast_eff = 1 - (1 - self.alpha_fast) ** self.dt 

        # Atualiza w_slow e w_fast
        w_avg = np.mean(self.weights)  # média dos pesos normalizados
        self.w_slow += self.alpha_slow *(w_avg - self.w_slow)
        self.w_fast += self.alpha_fast *(w_avg - self.w_fast)


    #======================================================================
    # LiDAR
    #======================================================================


    def lidar_callback(self, msg):

        self.update_scans(msg)
        #self.particles = generate_valid_particles(self.num_particles,
        #                                     self.map_data, self.resolution,
        #                                     self.origin_np[0], self.origin_np[1], self.width, self.height)

        #Corretion step
        
        #rospy.loginfo("Atualizando pesos das partículas")
        
        weights_pre, weights_post = self.update_weights()
        
        if self.use_mh:

            weights = self.update_particles_mh(weights_pre, weights_post)

        else:

            weights = weights_post

        
        if self.use_adaptive:

            self.update_acml_weights(weights)

        else:

            self.weights = weights

            
        #Publish and resampling
        #rospy.loginfo("Publicando pose estimada")
        self.publish_estimate()
        
        if self.use_adaptive:

            self.resample_amcl_kld()

        else:

            self.resample_lvr()
        
        #rospy.loginfo("Publicando partículas")
        self.publish_particles()


    def update_scans(self,scan):

        self.scan_ranges = np.array(scan.ranges, dtype=np.float32)
        self.angles = self.get_lidar_angles(scan)

    def get_lidar_angles(self, scan):
        num_ranges = len(scan.ranges)
        return np.linspace(scan.angle_min, scan.angle_max, num_ranges, dtype=np.float32)
    

    def convert_scores(self,scores):

        max_score = np.max(scores)
        weights = np.zeros_like(scores)
        weights = np.exp(scores - max_score)
        weights =  weights/np.sum(weights)

        return weights

    def update_particles_mh(self,weights_pre, weights_post):

        if not self.assym:
            mh_particles, weights = mh_resampling(self.particles_prev,self.particles,weights_post,weights_pre)
        else:

            trans_forward, trans_backward = self.transition_probability()
            mh_particles, weights = assym_mh_resampling(self.particles_prev,self.particles,weights_post,weights_pre,trans_forward,trans_backward)


        self.particles = mh_particles
        
        return weights

    #======================================================================
    # Odom
    #======================================================================


    def odom_callback(self, msg):

        #rospy.loginfo("Movendo partículas com odometria")
        self.move_particles(msg)

    def move_particles(self,msg):

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([orientation.x, orientation.y, 
                                              orientation.z, orientation.w])

        current_odom = np.array([position.x, position.y, yaw])

        if self.last_odom is not None:

            self.delta = self.compute_motion(self.last_odom, current_odom)
            
            self.particles_prop = apply_motion_model_parallel(self.particles,self.delta,self.alpha,
                                                              self.map_data, self.resolution,
                                                              self.origin_np[0], self.origin_np[1],
                                                              self.width,self.height)
            
            #rospy.loginfo(f"Partículas movidas: {len(self.particles_prop)}\n")
            
            self.particles_prev = self.particles.copy()
            self.particles = self.particles_prop.copy()
            

        self.last_odom = current_odom

    def compute_motion(self, odom1, odom2):
        dx = odom2[0] - odom1[0]
        dy = odom2[1] - odom1[1]
        dtheta = normalize_angle(odom2[2] - odom1[2])

        rot1 = np.arctan2(dy, dx) - odom1[2]
        trans = np.hypot(dx, dy)
        rot2 = dtheta - rot1

        

        return rot1, trans, rot2


    def transition_probability(self):

        trans_forward = motion_model_odometry_parallel(self.particles_prev,self.particles,
                                                       np.array(self.delta), self.alpha)
        
        dx, dy, dtheta = self.delta
        backward_delta = np.array([
            -dx * np.cos(dtheta) - dy * np.sin(dtheta),
            dx * np.sin(dtheta) - dy * np.cos(dtheta),
            -dtheta
        ])
        
        trans_backward = motion_model_odometry_parallel(self.particles,self.particles_prev,
                                                        backward_delta, self.alpha)

        return trans_forward, trans_backward
    #======================================================================
    # Resample
    #======================================================================

    def resample_amcl_simple(self):

        p_random = max(0.0, 1.0 - self.w_fast / (self.w_slow + 1e-9))

        N = self.num_particles
        N_random = int(p_random * N)
        N_resampled = N - N_random

        resampled_particles = parallel_resample_simple(self.particles,self.weights,N_resampled)

        random_particles = generate_valid_particles(N_random,self.map_data,
                                                    self.resolution,self.origin_np[0],self.origin_np[1],self.width,self.height)
        
        self.particles = np.vstack((resampled_particles,random_particles))
        self.weights   = np.full(N,1/N)

    def resample_amcl_lvr(self):

        p_random = max(0.0, 1.0 - self.w_fast / (self.w_slow + 1e-9))

        N = self.num_particles
        resampled_particles = np.zeros_like(self.particles)

        resampled_index, _ = low_variance_resample_numba(np.arange(N), self.weights, N)
        resampled_index = resampled_index.astype(np.int64)

        for i in range(N):
            if np.random.rand() < p_random:
                resampled_particles[i,:] = generate_valid_particles(1,self.map_data,
                                                    self.resolution,self.origin_np[0],self.origin_np[1],self.width,self.height)

            else:
                resampled_particles[i,:] = self.particles[resampled_index[i],:]
        
        self.particles = resampled_particles.copy()
        self.weights   = np.full(N,1/N)


    def resample_simple(self):

        resampled_particles = parallel_resample_simple(self.particles,self.weights,N=self.num_particles)

        self.particles = resampled_particles

    def resample_lvr(self): #not fixed

        resampled_particles, _ = low_variance_resample_numba(self.particles,self.weights,N=self.num_particles)

        self.particles = resampled_particles



    def resample_amcl_kld(self):
        p_random = max(0.0, 1.0 - self.w_fast / (self.w_slow + 1e-9))

        N = self.num_particles
        N_random = int(p_random * N)
        N_resampled = N - N_random

        # KLD Sampling com Numba
        resampled_particles = kld_sampling_amcl(
            self.particles,
            self.weights,
            self.kld_bin_size_xy,
            self.kld_bin_size_theta,
            self.kld_epsilon,
            self.kld_z,
            N_resampled,
            self.min_particles
        )


        random_particles = generate_valid_particles(N_random,self.map_data,
                                                    self.resolution,self.origin_np[0],self.origin_np[1],self.width,self.height)

        # Junta
        self.num_particles = len(self.particles)
        self.particles = np.vstack((random_particles, resampled_particles))
        self.weights   = np.full(len(self.particles), 1.0 / len(self.particles))

        if len(self.particles) != N:

            rospy.loginfo(f"Particle update!\n From: {N}  To: {len(self.particles)}")
        

        




    #======================================================================
    # Publish
    #======================================================================

    def publish_particles(self):
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        weights = self.weights[:len(self.particles)]
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)

        cos_half_theta = np.cos(self.particles[:,2] / 2.0)
        sin_half_theta = np.sin(self.particles[:,2] / 2.0)
        marker_id =0
        for p, w in zip(self.particles, norm_weights):
            if not self.is_valid_position(p[0], p[1]):
                continue
                
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "particles"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 1.0
            marker.color.r = w
            marker.color.g = 0.0
            marker.color.b = 1 - w
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            theta = p[2]
            z = np.sin(theta / 2.0)
            marker.pose.orientation.z = z
            marker.pose.orientation.w = np.cos(theta / 2.0)
            
            
            marker_array.markers.append(marker)

        if not rospy.is_shutdown():
            self.marker_pub.publish(marker_array)

    
    def publish_estimate(self):

        mean_pose = np.average(self.particles, axis=0,weights=self.weights)
        cos_mean = np.sum(np.cos(self.particles[:,2]) * self.weights)
        sin_mean = np.sum(np.sin(self.particles[:,2]) * self.weights)
        mean_theta = np.arctan2(sin_mean, cos_mean)
        diffs = self.particles.copy()
        diffs[:, 0] -= mean_pose[0]
        diffs[:, 1] -= mean_pose[1]
        diffs[:, 2] = normalize_angle_array(self.particles[:, 2], mean_theta)
        if len(self.particles) < 2:
            rospy.logwarn("Not enough particles to compute covariance")
            return
        cov = np.cov(diffs.T, aweights=self.weights)
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.pose.position.x = mean_pose[0]
        pose.pose.pose.position.y = mean_pose[1]
        pose.pose.pose.orientation.z = np.sin(mean_theta / 2.0)
        pose.pose.pose.orientation.w = np.cos(mean_theta / 2.0)

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
        if not rospy.is_shutdown():
            self.pose_pub.publish(pose)


if __name__ == '__main__':
    try:
        AMCMHLocalizer()
    except rospy.ROSInterruptException:
        pass
    
