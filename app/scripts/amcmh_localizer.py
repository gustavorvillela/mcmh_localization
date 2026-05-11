#!/usr/bin/env python3
import rospy
import numpy as np
from numpy.random import choice
import tf.transformations as tft
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from parallel_utils import compute_likelihoods, mh_resampling, apply_motion_model_parallel, normalize_angle, generate_valid_particles, low_variance_resample_numba, normalize_angle_array, kld_sampling_amcl, initialize_gaussian_parallel,parallel_resample_simple, motion_model_odometry_parallel, accumulate_meta_particles, finalize_meta_particles
import message_filters
import time

class AMCMHLocalizer:
    def __init__(self):
        rospy.init_node('mcmh_localizer')
        self.mode = rospy.get_param('localization_mode', 'MCL')  # default: MCL
        self.use_mh = 'MH' in self.mode
        self.use_adaptive = 'AMCL' in self.mode  # AMCL or MHAMCL use KLD
        self.meta = '3' in self.mode  # 3MCL or Meta-MH-MCL uses path history in MH step

        rospy.loginfo(f"Localization mode: {self.mode} | MH: {self.use_mh}, Augmented: {self.use_adaptive},  Meta: {self.meta}")


        # General parameters
        self.num_particles = rospy.get_param('init_particles', 2000) 
        self.alpha = np.array([
                                rospy.get_param('alpha1', 0.2),
                                rospy.get_param('alpha2', 0.2),
                                rospy.get_param('alpha3', 0.2),
                                rospy.get_param('alpha4', 0.2),
                                rospy.get_param('alpha5', 0.2),
                                rospy.get_param('alpha6', 0.2)
                            ], dtype=np.float32) #do not touch
        self.alpha_slow = rospy.get_param('alpha_slow', 0.01) # slow learning rate for AMCL
        self.alpha_fast = rospy.get_param('alpha_fast', 0.1)  # fast learning rate for AMCL

        self.dt = 0.02 # scan time interval

        self.delta = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # (rot1, trans, rot2)
        self.delta_path = np.empty((0, 3), dtype=np.float32)      # delta history for Meta-MH-MCL
        self.odom_eps = 1e-6
        self.accept_odom = False

        # Parâmetros KLD
        self.kld_epsilon = rospy.get_param('kld_epsilon', 0.025)
        self.kld_delta = rospy.get_param('kld_delta', 0.99)
        self.kld_bin_size_xy = rospy.get_param('kld_bin_size_xy', 0.1)  # meters
        self.kld_bin_size_theta = rospy.get_param('kld_bin_size_theta', np.deg2rad(10))  # radians
        self.kld_n_max = self.num_particles
        self.kld_z = rospy.get_param('kld_z', 2)

        self.initial_pose = None  # Will store the initial pose [x, y, theta]
        self.initial_cov = np.diag([0.05, 0.05, 0.1])  # Initial covariance (x, y in meters, theta in rad)
        self.initialized = rospy.get_param('initialized', False)  # control flag

        self.sigma_hit = rospy.get_param('sigma_hit', 0.2)  # Parameter for Gaussian probability function
        self.max_range = rospy.get_param('max_range', 10.0)  # Maximum LiDAR range to consider (in meters)
        self.z_hit = rospy.get_param('z_hit', 0.8)  # Peso para a parte "hit"
        self.z_rand = rospy.get_param('z_rand', 0.2)  # Peso para a parte "random"
        self.z_short = rospy.get_param('z_short', 0.05)  # Weight for the "short" part (unexpected obstacles)
        self.z_max = rospy.get_param('z_max', 0.05)  # Weight for the "max" part (readings at maximum range)
        self.lambda_short = rospy.get_param('lambda_short', 0.1)  # Lambda for exponential distribution of the "short" part
        self.step = rospy.get_param('step', 1)  # Use every 'step' LiDAR measurements to speed up
        self.headless = rospy.get_param('headless', False)  # If True, do not publish markers for visualization
        self.timeout = 10

        self.initial_pose_topic = rospy.get_param('initial_pose_topic', '/initial_pose')

        if self.initialized == True:
            rospy.loginfo("Waiting for initial pose (max %.1fs)..." % self.timeout)

            # First check if the topic exists
            try:
                rospy.wait_for_message(self.initial_pose_topic, PoseWithCovarianceStamped, timeout=10.0)
            except rospy.ROSException:
                rospy.logwarn("Topic %s not found. Check if the publisher is active." % self.initial_pose_topic)
                pass

            

            try:
                msg = rospy.wait_for_message(self.initial_pose_topic, PoseWithCovarianceStamped, timeout=10.0)
                self.initial_pose_callback(msg)
            except:
                pass
        
        else:
            rospy.loginfo("Initializing particles uniformly on the map")

        #AMCL
        self.min_particles = self.min_particles = rospy.get_param('min_particles', 100)
        self.max_particles = self.max_particles = rospy.get_param('max_particles', 5000)
        self.w_slow = 1e-3
        self.w_fast = 1e-3
 
        # Load map
        
        
        self.load_map()

        self.warmup_numba()

        # Initialize particles
        self.particles = self.initialize_particles(self.num_particles).astype(np.float32)
        self.particles_prop = np.copy(self.particles)
        self.particles_prev = np.copy(self.particles_prop)
        self.meta_particles = np.copy(self.particles)
        
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.weights_pre = self.weights.copy()
        self.scan_ranges = None
        self.odom_count = 0
        # Exponential recency weighting for Meta-MH
        self.meta_lambda = rospy.get_param("meta_lambda", 0.85)

        # Equivalent decay factor
        self.meta_decay = np.exp(-self.meta_lambda)

        # Current recency multiplier
        self.meta_time_weight = 1.0

        self.meta_xy = self.meta_particles[:, :2].copy() * self.weights_pre.copy()[:, np.newaxis]
        self.meta_cos = np.cos(self.meta_particles[:, 2]).copy() * self.weights_pre
        self.meta_sin = np.sin(self.meta_particles[:, 2]).copy() * self.weights_pre
        
        self.meta_weights = self.weights_pre.copy()  # Initialize meta weights as zeros
        self.weights_viz = self.weights.copy()

        self.last_odom = None

        

        self.odom_topic = rospy.get_param('odom_topic', '/odom')
        self.scan_topic = rospy.get_param('scan_topic', '/scan')
        
        # Subscribers
        if not self.meta:
            scan_sub = message_filters.Subscriber(self.scan_topic, LaserScan)
            odom_sub = message_filters.Subscriber(self.odom_topic, Odometry)
            ts = message_filters.ApproximateTimeSynchronizer([scan_sub, odom_sub], queue_size=10, slop=0.1)
            ts.registerCallback(self.sync_callback)
        else:
            rospy.Subscriber(self.scan_topic, LaserScan, self.lidar_callback)
            rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

        # Publishers
        self.pose_pub = rospy.Publisher('/mcmh_estimated_pose', PoseWithCovarianceStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/mcmh_particles', MarkerArray, queue_size=1)
        self.acc_rate = rospy.Publisher('/mh_rate', Float64, queue_size=1)
        
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

        rospy.loginfo("Generating distance map...")
        self.dist_2d = distance_transform_edt(occupancy_binary == 0) * resolution
        self.distance_map = self.dist_2d.flatten().astype(np.float32)
        rospy.loginfo("Distance map generated.")

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

    def warmup_numba(self):

        rospy.loginfo("Warming up numba kernels...")
        t = time.time()

        N  = 5
        Ns = 10
        
        if self.initialized:
            particles = initialize_gaussian_parallel(self.initial_pose,self.initial_cov,N,
                                                           self.dist_2d,self.resolution,self.origin_np).astype(np.float32)
        else:
            particles = generate_valid_particles(N, self.map_data, self.resolution, self.origin_np[0], self.origin_np[1], self.width, self.height)

        
        dummy_particles = particles.astype(np.float32)
        dummy_weights = np.ones(N, dtype=np.float32) / N

        dummy_scan = np.ones(Ns, dtype=np.float32)
        dummy_angles = np.linspace(-1.0, 1.0, Ns, dtype=np.float32)

        dummy_delta = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # motion model
        apply_motion_model_parallel(
            dummy_particles,
            dummy_delta,
            self.alpha,
            self.map_data,
            self.resolution,
            self.origin_np[0],
            self.origin_np[1],
            self.width,
            self.height
        )

        # sensor model
        compute_likelihoods(
            dummy_scan,
            dummy_angles,
            dummy_particles,
            self.distance_map,
            self.resolution,
            self.origin_np,
            self.width,
            self.height,
            self.sigma_hit,
            self.z_hit,
            self.z_rand,
            self.max_range,
            self.step,
            self.z_short,
            self.z_max,
            self.lambda_short
        )

        # MH
        if self.use_mh:
            mh_resampling(
                dummy_particles,
                dummy_particles.copy(),
                dummy_weights,
                dummy_weights
            )

        
        if self.use_adaptive:
            # KLD
            kld_sampling_amcl(
                dummy_particles,
                dummy_weights,
                self.kld_bin_size_xy,
                self.kld_bin_size_theta,
                self.kld_epsilon,
                self.kld_z,
                10,
                5
            )
        else:
            # LVR
            low_variance_resample_numba(
                dummy_particles,
                dummy_weights,
                5
            )


        rospy.loginfo(f"Numba warmup done in {time.time() - t:.2f} seconds.")

    def initialize_particles(self,num_particles=100):

        if self.initialized == True:
            rospy.loginfo("Initialize particles around initial pose with Gaussian distribution")
            final_particles = initialize_gaussian_parallel(self.initial_pose,self.initial_cov,num_particles,
                                                           self.dist_2d,self.resolution,self.origin_np)
            
        else:
            final_particles = generate_valid_particles(num_particles,
                                             self.map_data, self.resolution,
                                             self.origin_np[0], self.origin_np[1], self.width, self.height)

        print(f"[DEBUG] Generated {final_particles.shape[0]} valid particles")

        if final_particles.shape[0] == 0:
            rospy.logerr("No valid particles generated! Check map indexing and limits.")
        
        return final_particles
    
    def initial_pose_callback(self, msg):
        """Callback to receive the initial pose (geometry_msgs/PoseWithCovarianceStamped)"""
        pose = msg.pose.pose
        self.initial_pose = np.array([
            pose.position.x,
            pose.position.y,
            self.get_yaw_from_quaternion(pose.orientation)  # Implement this function
        ])
        self.initialized = True
        rospy.loginfo(f"Initial pose received: {self.initial_pose}")

    def wait_for_initial_pose(self, timeout=5.0):
        """
        Wait for the initial pose for a given time (in seconds).
        If not received within the timeout, proceed with uniform initialization.
        """
        rospy.loginfo("Waiting for initial pose (max %.1fs)..." % timeout)
        start_time = rospy.Time.now().to_sec()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.initialized:
                rospy.loginfo("Initial pose received.")
                break
            if rospy.Time.now().to_sec() - start_time > timeout:
                rospy.logwarn("Timeout: initial pose not received. Initializing uniformly.")
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

    def convert_scores(self,scores):

        max_score = np.max(scores)
        weights = np.zeros_like(scores)
        weights = np.exp(scores - max_score)  # Subtract max for numerical stability
        weights =  weights/np.sum(weights)

        return weights

    def calculate_weights(self, particles):

        scores = compute_likelihoods(
            self.scan_ranges, self.angles, particles,
            self.distance_map, self.resolution, self.origin_np,
            self.width, self.height,self.sigma_hit,
            self.z_hit, self.z_rand, self.max_range, self.step,
            self.z_short, self.z_max, self.lambda_short
        )

        weights = self.convert_scores(scores)

        return weights

    def calculate_unorm_weights(self, particles):

        scores = compute_likelihoods(
            self.scan_ranges, self.angles, particles,
            self.distance_map, self.resolution, self.origin_np,
            self.width, self.height,self.sigma_hit,
            self.z_hit, self.z_rand, self.max_range, self.step,
            self.z_short, self.z_max, self.lambda_short
        )

        max_score = np.max(scores)
        weights = np.exp(scores)  # Subtract max for numerical stability

        return weights


    def update_weights(self, particles_prev, particles_post):

        #print(f"[DEBUG] Calculating weights for {len(self.particles)} particles post and {len(self.particles_prev)} particles pre...")

        weights_pre = self.calculate_weights(particles_prev)

        weights_post = self.calculate_weights(particles_post)

        #print(f"[DEBUG] Nb weights pre: {len(weights_pre)} | Nb weights post: {len(weights_post)} | Nb particles: {len(self.particles)}")


        return weights_pre, weights_post
    

    def update_acml_weights(self,weights):

        self.weights = weights/np.sum(weights)

        # Atualiza w_slow e w_fast
        w_avg = np.mean(self.weights)  # mean of normalized weights
        self.w_slow += self.alpha_slow *(w_avg - self.w_slow)
        self.w_fast += self.alpha_fast *(w_avg - self.w_fast)


    #======================================================================
    # LiDAR
    #======================================================================


    def lidar_callback(self, msg):

        self.accept_odom = False

        # Reset exponential history
        self.meta_time_weight = 1.0
        #print(f"[DEBUG] Total odom steps used: {self.odom_count} | Resetting meta time weight.")
        

        weight_safe = self.meta_weights.copy()
        weight_safe[weight_safe == 0] = 1e-6  # Avoid division by zero
        #print(f"[DEBUG] Meta weights before normalization: {self.meta_weights}")
        meta_xy =self.meta_xy / self.meta_weights[:, np.newaxis] # Compute mean x and y from weighted sum
        
        self.odom_count = 0
        meta_theta = np.arctan2(self.meta_sin, self.meta_cos)  # Compute mean angle from weighted sin and cos

        self.meta_particles = np.column_stack((meta_xy, meta_theta)).astype(np.float32)  # Final meta particles for this scan
        #print(f"[DEBUG] Meta particles after incorporating path history (before scan update): {self.meta_particles}")
        self.update_scans(msg)
        self.particles = self.meta_particles.copy()  # Update particles to the meta particles before resampling, so that the resampling step works with the updated distribution that incorporates the path history and the new scan information.
        weights = self.calculate_weights(self.particles)  # Final weight update for the meta particles based on the current scan

        #print(f"[DEBUG] Final meta particles (before normalization): {self.particles} | weights: {weights}")
        # ==========================
        # Final meta weights update
        # ==========================
        
        if self.use_adaptive:
        
            self.update_acml_weights(weights)
        
        else:
        
            self.weights = weights

        # =======================
        # Publish and resampling
        # =======================
        #rospy.loginfo("Publicando pose estimada")
        
        
        if self.use_adaptive:
        
            self.resample_amcl_kld()
        
        else:
        
            self.resample_lvr()
        
        self.meta_particles = self.particles.copy()  # Update meta particles for the next iteration
        self.particles_prev = self.particles.copy()  # Update previous particles for the next iteration

        #print(f"[DEBUG] Finished lidar callback")

        self.meta_weights = self.calculate_unorm_weights(self.meta_particles)  # Update meta weights for the next iteration

        self.meta_xy = self.meta_particles[:, :2] * self.meta_weights[:, np.newaxis]  # Update meta xy for the next iteration
        self.meta_cos = np.cos(self.meta_particles[:, 2]) * self.meta_weights  # Update meta cos for the next iteration
        self.meta_sin = np.sin(self.meta_particles[:, 2]) * self.meta_weights  # Update meta sin for the next iteration

        self.accept_odom = True
        # rospy.loginfo("Publishing particles")
        self.publish_particles()
        self.publish_estimate()

    def update_scans(self,scan):

        self.scan_ranges = np.array(scan.ranges, dtype=np.float32)
        self.angles = self.get_lidar_angles(scan)

    def get_lidar_angles(self, scan):
        num_ranges = len(scan.ranges)
        return np.linspace(scan.angle_min, scan.angle_max, num_ranges, dtype=np.float32)
    

    

    def update_particles_mh(self,weights_pre, weights_post, particles_prev=None, particles_post=None):

        if particles_prev is None:
            particles_prev = self.particles_prev.copy()
        if particles_post is None:
            particles_post = self.particles_prop.copy()

        mh_particles, weights, acc_rate = mh_resampling(particles_prev,particles_post,weights_post,weights_pre)
        
        
        return weights, mh_particles, acc_rate

    #======================================================================
    # Odom
    #======================================================================


    def odom_callback(self, msg):
        
        if not self.accept_odom:
            return

        #rospy.loginfo("Moving particles with odometry")
        self.delta, current_odom = self.get_delta_odom(msg)
        #current_path = self.delta_path.copy()
        #self.delta_path = np.vstack((current_path,self.delta.reshape(1,3)))

        # apply motion model and update particles 
        # mh particles updated here and particles (meta-particles in 3MCL) in the lidar callback 
        # after processing the scan with the new path history
        self.particles_prop, _  = apply_motion_model_parallel(self.particles_prev,self.delta,self.alpha,
                                                          self.map_data, self.resolution,
                                                          self.origin_np[0], self.origin_np[1],
                                                          self.width,self.height)
  
        # compute weights for particles before and after motion to use in MH step.
        #print(f"[DEBUG] Proposed particles after motion : {self.particles_prop}")

        self.weights_post = self.calculate_unorm_weights(self.particles_prop)


        # MH step to decide which particles to keep for the next iteration, with update on meta set
        # being made on lidar callback after processing the new scan.

        mh_weights, mh_particles, _ = self.update_particles_mh(self.weights_pre, self.weights_post,
                                                               self.particles_prev, self.particles_prop)

        mh_xy = mh_particles[:, :2]
        mh_cos = np.cos(mh_particles[:, 2])
        mh_sin = np.sin(mh_particles[:, 2])

        # Meta distribution update: we accumulate the accepted particles and their weights across all deltas in the path history for a given
        # scan, so that the meta particles represent a more informed distribution that considers multiple recent movements, not just the
        # last one. This is the core idea of Meta-MH-MCL.

        # Older samples get exponentially less importance
        self.meta_xy *=  self.meta_decay
        self.meta_cos *=  self.meta_decay
        self.meta_sin *=  self.meta_decay
        self.meta_weights *=  self.meta_decay

        self.meta_xy +=  mh_xy * mh_weights[:, np.newaxis]
        self.meta_cos +=  mh_cos * mh_weights
        self.meta_sin += mh_sin * mh_weights
        self.meta_weights += mh_weights
        
        self.meta_time_weight *= self.meta_decay  # Decay the time weight for the next iteration

        self.particles_prev = mh_particles.copy()  # Update previous particles to the MH result for the next iteration

        self.weights_pre = mh_weights.copy()  # Update previous weights to the MH result for the next iteration
        self.last_odom = current_odom   
        self.odom_count += 1

    def get_delta_odom(self,msg):

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([orientation.x, orientation.y, 
                                              orientation.z, orientation.w])

        current_odom = np.array([position.x, position.y, yaw])

        if self.last_odom is not None:

            delta = np.array(self.compute_motion(self.last_odom, current_odom), dtype=np.float32)        
        else:

            delta = np.array((0.0, 0.0, 0.0), dtype=np.float32)
            print("[DEBUG] First odometry received, no motion applied.")
            
        return delta, current_odom
    
    def update_particle_set(self,delta):

        particles_prop, _ = apply_motion_model_parallel(self.particles,delta,self.alpha,
                                                          self.map_data, self.resolution,
                                                          self.origin_np[0], self.origin_np[1],
                                                          self.width,self.height)
        
        return particles_prop

    def move_particles(self,msg):

        self.delta, current_odom = self.get_delta_odom(msg)
            
        self.particles_prop = self.update_particle_set(self.delta)
        
        # rospy.loginfo(f"Particles moved: {len(self.particles_prop)}\n")
        #print(f"[DEBUG] Odom delta: rot1={self.delta[0]:.4f}, trans={self.delta[1]:.4f}, rot2={self.delta[2]:.4f}")
        #print(f"[DEBUG] Sampled deltas (first 5): {deltas[:5]}")
        self.particles_prev = self.particles.copy()
        self.particles = self.particles_prop.copy()

                
        self.last_odom = current_odom


    def compute_motion(self, odom1, odom2):
        dx = odom2[0] - odom1[0]
        dy = odom2[1] - odom1[1]
        trans = np.hypot(dx, dy)

        dtheta = normalize_angle(odom2[2] - odom1[2])

        rot1 = normalize_angle(np.arctan2(dy, dx) - odom1[2])
        rot2 = normalize_angle(dtheta - rot1)

        return rot1, trans, rot2


    def transition_probability(self):
        #print(f"[DEBUG] Delta for transition probability: {self.delta}")
        if self.delta == (0.0, 0.0, 0.0):
            return np.ones(len(self.particles)), np.ones(len(self.particles_prev))

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
        #rospy.loginfo("Realizando amostragem KLD...")
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

        #print(f"[DEBUG] KLD resampled {resampled_particles.shape[0]} particles | Nans: {np.isnan(resampled_particles).sum()} | Infs: {np.isinf(resampled_particles).sum()}")


        random_particles = generate_valid_particles(N_random,self.map_data,
                                                    self.resolution,self.origin_np[0],self.origin_np[1],self.width,self.height)

        # Junta
        self.num_particles = len(self.particles)
        self.particles = np.vstack((random_particles, resampled_particles))
        self.weights   = np.full(len(self.particles), 1.0 / len(self.particles))
        #print(f"[DEBUG] Total particles after KLD + random: {len(self.particles)} | Random: {N_random} | Resampled: {resampled_particles.shape[0]}")
        #print(f"[DEBUG] KLD weights sum: {np.sum(self.weights)} | min: {np.min(self.weights)} | max: {np.max(self.weights)} | Nb_weights: {len(self.weights)}")

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


        weights = self.calculate_weights(self.particles)
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
        #print(f"[DEBUG] Publishing {len(self.particles)} particles with normalized weights (min: {norm_weights.min():.4f}, max: {norm_weights.max():.4f})")
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
        # We use only the dimensions x, y, theta -> [0,0], [1,1], [5,5]
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

    def sync_callback(self, scan_msg, odom_msg):
        
        # 1. MOVE: Apply Odometry first
        # This keeps particles_prev and particles at the same size
        #print("[DEBUG] Sync callback triggered: moving particles with odometry...")
        t = time.time()
        self.move_particles(odom_msg) 
        #print(f"[DEBUG] Particle movement took {time.time() - t:.4f} seconds")
        
        # 2. WEIGHT: Use the LiDAR scan to update weights
        #print("[DEBUG] Updating weights with LiDAR scan...")
        t = time.time()
        self.update_scans(scan_msg)
        #print(f"[DEBUG] Scan processing took {time.time() - t:.4f} seconds")

        t = time.time()
        weights_pre, weights_post = self.update_weights(self.particles_prev, self.particles)
        #print(f"[DEBUG] Weight update took {time.time() - t:.4f} seconds")
        
        # 3. MH STEP: Perform MH resampling
        if self.use_mh:
            weights, self.particles, acc_rate = self.update_particles_mh(weights_pre, weights_post)
        else:
            weights = weights_post
            acc_rate = 0.0

        # 4. RESAMPLE: This is where KLD might change the size for the NEXT frame
        t = time.time()
        if self.use_adaptive:
            self.update_acml_weights(weights)
            self.resample_amcl_kld()
        else:
            self.weights = weights
            self.resample_lvr()
        #print(f"[DEBUG] Resampling took {time.time() - t:.4f} seconds")

        # 5. PUBLISH
        t = time.time()
        
        self.acc_rate.publish(Float64(acc_rate))
        self.publish_particles()
        self.publish_estimate()
        #print(f"[DEBUG] Publishing took {time.time() - t:.4f} seconds")

if __name__ == '__main__':
    try:
        AMCMHLocalizer()
    except rospy.ROSInterruptException:
        pass