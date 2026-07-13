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
from parallel_utils import compute_likelihoods, mh_resampling, apply_motion_model_parallel, apply_random_walk_parallel,\
                        normalize_angle, generate_valid_particles, low_variance_resample_numba, normalize_angle_array,\
                        kld_sampling_amcl,initialize_gaussian_parallel,parallel_resample_simple, motion_model_odometry_parallel,\
                        accumulate_meta_particles, finalize_meta_particles
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
                                rospy.get_param('alpha4', 0.2)
                            ], dtype=np.float32) #do not touch
        
        self.alpha_rw = np.array([
                                rospy.get_param('alpha5', 0.02),  # additional noise for in-place rotation
                                rospy.get_param('alpha6', 0.04),  # additional noise for in-place translation
                                rospy.get_param('alpha7', 0.01),  # additional noise for small forward movement
                                rospy.get_param('alpha8', 0.01)   # additional noise for small backward movement
                            ], dtype=np.float32) # do not touch, only used for the random walk step in 3MCL/Augmented-3MCL to encourage exploration and prevent particle deprivation, especially in the early stages of localization or when the robot is lost. Tune these based on your environment and robot's motion characteristics.
        self.alpha_slow = rospy.get_param('alpha_slow', 0.01) # slow learning rate for AMCL
        self.alpha_fast = rospy.get_param('alpha_fast', 0.1)  # fast learning rate for AMCL

        self.dt = 0.02 # scan time interval

        self.delta = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # (rot1, trans, rot2)
        self.delta_path = np.empty((0, 3), dtype=np.float32)      # delta history for Meta-MH-MCL
        self.odom_eps = 1e-4  # Threshold to consider translation as zero for in-place rotation handling
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

        self.meta_lambda = rospy.get_param("meta_lambda", 0.85)
        self.Nr = rospy.get_param("random_steps", 10)  # Number of random walk steps per odometry update
        self.Neff = self.num_particles  # Initialize effective sample size


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
        self.min_particles = rospy.get_param('min_particles', self.num_particles/2)
        self.max_particles = rospy.get_param('max_particles', self.num_particles*2)
        self.w_slow = 1/self.num_particles
        self.w_fast = 1/self.num_particles
 
        # Load map
        
        
        self.load_map()

        

        # Initialize particles
        self.particles = self.initialize_particles(self.num_particles).astype(np.float32)
        self.particles_prop = np.copy(self.particles)
        self.particles_prev = np.copy(self.particles_prop)
        self.meta_particles = np.copy(self.particles)
        
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        self.weights_pre = self.weights.copy()
        self.scan_ranges = None
        self.last_scan = None
        self.odom_count = 0
        # Exponential recency weighting for Meta-MH

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
            rospy.Subscriber(self.scan_topic, LaserScan, self.lidar_callback, queue_size=10, buff_size=2**24)
            rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10, buff_size=2**24)

        # Publishers
        self.pose_pub = rospy.Publisher('/mcmh_estimated_pose', PoseWithCovarianceStamped, queue_size=10)
        self.marker_pub = rospy.Publisher('/mcmh_particles', MarkerArray, queue_size=10)
        self.acc_rate = rospy.Publisher('/mh_rate', Float64, queue_size=10)
        self.Neff_pub = rospy.Publisher('/effective_sample_size', Float64, queue_size=10)
        
        # TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.publish_particles()
        self.acc_rate.publish(Float64(1.0))
        self.Neff_pub.publish(Float64(self.num_particles))
        self._viz_count = 0
        
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
        self.map_data = map_2d.flatten()      # dtype int8
        #print(f"[DEBUG] Map loaded: width={width}, height={height}, resolution={resolution:.3f} m/cell, origin=({origin_x:.2f}, {origin_y:.2f})")

        # occupancy_map (binary: 0 free, 1 occupied) for distance transform
        obstacles_binary = (map_2d != 0).astype(np.uint8)  # occupied=1, free=0, unknown treated as free for distance map
        #occupancy_binary = (map_2d != 0).astype(np.uint8)  # occupied=1, free=0

        #rospy.loginfo("Generating distance map...")
        self.dist_2d = distance_transform_edt(obstacles_binary == 0) * resolution
        unknown_mask = (map_2d == -1)
        self.dist_2d[unknown_mask] = 10*self.max_range  # Assign large distance to unknown cells to encourage exploration, can be tuned based on the environment and desired behavior.
        self.distance_map = self.dist_2d.flatten().astype(np.float32)
        #rospy.loginfo("Distance map generated.")

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
        #print(f"[DEBUG] Max score: {max_score:.4f} | Min score: {np.min(scores):.4f} | Mean score: {np.mean(scores):.4f}")
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

        weights = np.exp(scores)  # Subtract max for numerical stability, but do NOT normalize to sum to 1, as we want to keep the unnormalized weights for the meta distribution update in Meta-MH-MCL.

        return weights


    def update_weights(self, particles_prev, particles_post):

        #print(f"[DEBUG] Calculating weights for {len(self.particles)} particles post and {len(self.particles_prev)} particles pre...")

        weights_pre = self.calculate_weights(particles_prev)

        weights_post = self.calculate_weights(particles_post)

        #print(f"[DEBUG] Nb weights pre: {len(weights_pre)} | Nb weights post: {len(weights_post)} | Nb particles: {len(self.particles)}")


        return weights_pre, weights_post
    

    def update_acml_weights(self,weights):

        #self.weights = weights/np.sum(weights)

        # Updates w_slow and w_fast
        w_avg = np.sum(weights)/len(weights)  # mean of normalized weights
        
        self.w_slow += self.alpha_slow *(w_avg - self.w_slow)
        self.w_fast += self.alpha_fast *(w_avg - self.w_fast)
        print(f"[DEBUG] w_slow: {self.w_slow:.6f} | w_fast: {self.w_fast:.6f} | w_avg: {w_avg:.6f} | w_max: {np.max(weights):.6f} | w_min: {np.min(weights):.6f}")

        #self.weights = weights/np.sum(weights)


    #======================================================================
    # LiDAR
    #======================================================================


    def lidar_callback(self, msg):

        
        self.accept_odom = False
        #print(f"[DEBUG] Total odom steps used: {self.odom_count}")

        # Recover each meta-particle pose as the weighted mean meta_xy / Σw.
        # When Σw collapses to ~0 the mean is undefined (numerator ~0 too), so a
        # denominator floor would send the particle to the origin. Instead, mark
        # those particles and fall back to their last known pose (particles_prev,
        # which meta_xy/meta_weights were built from, so indices align).
        # Degeneracy test is RELATIVE to the strongest particle, so it is
        # invariant to the absolute weight scale (the likelihood is now a
        # per-beam average, so weights are O(0.01..1), never ~1e-300).
        weight_safe = self.meta_weights.copy()
        max_w = np.max(weight_safe)
        degenerate = weight_safe < 1e-12 * max(max_w, 1e-300)
        weight_safe[degenerate] = 1e-12  # placeholder; these rows are overwritten below
        #print(f"[DEBUG] Meta weights: mean={np.mean(self.meta_weights):.6e}, max={max_w:.6e}, min={np.min(self.meta_weights):.6e}, degenerate={int(np.sum(degenerate))}")
        #print(f"[DEBUG] Meta weights before normalization: {self.meta_weights}")
        meta_xy =self.meta_xy / weight_safe[:, np.newaxis] # Compute mean x and y from weighted sum

        meta_theta = np.arctan2(self.meta_sin, self.meta_cos)  # Compute mean angle from weighted sin and cos

        # Degenerate (or any non-finite) reconstruction keeps the particle's last
        # known pose rather than collapsing to (0,0).
        bad = degenerate | ~np.isfinite(meta_xy).all(axis=1) | ~np.isfinite(meta_theta)
        meta_xy[bad] = self.particles_prev[bad, :2]
        meta_theta[bad] = self.particles_prev[bad, 2]

        self.particles = np.column_stack((meta_xy, meta_theta)).astype(np.float32)  # Final meta particles for this scan
        #print(f"[DEBUG] Meta particles after incorporating path history (before scan update): {self.meta_particles}")
        self.update_scans(msg)


        # ==========================
        # Final meta weights update
        # ==========================
        
        if self.use_adaptive:
        
            self.update_acml_weights(self.meta_weights)
        
        else:
            
            weights = self.meta_weights/np.sum(self.meta_weights)  # Normalize meta weights to get the final weights for this scan, so that the MH step in the next odometry update uses the updated meta distribution that incorporates the path history up to this point.
            self.weights = weights

        # =======================
        # Publish and resampling
        # =======================
        #rospy.loginfo("Publicando pose estimada")
        
        self.num_particles = len(self.particles)  # Update number of particles for the next iteration, in case it changed due to KLD resampling

        t = time.time()
        Neff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        #print(f"[DEBUG] Effective sample size (Neff): {Neff:.2f} | Threshold: {self.num_particles / 2.0:.2f}")
        self.Neff_pub.publish(Float64(Neff))
        if self.use_adaptive:
        
            if Neff < self.num_particles / 2.0 or self.last_odom is None:
                self.resample_amcl_kld()
        
        else:
            if Neff < self.num_particles/2 or self.last_odom is None:  # Resample if effective sample size is too low or if we haven't received any odometry yet (e.g., at the very beginning)
                self.resample_lvr()
        
        self.particles_prev = self.particles.copy()  # Update previous particles for the next iteration
        #print(f"[DEBUG] Updated particles prev")

        t = time.time()
        self.meta_weights = self.calculate_unorm_weights(self.particles_prev)  # Update meta weights for the next iteration
        self.weights_pre = self.meta_weights.copy()  # Update weights_pre to the new meta weights for the next iteration, so that the MH step in the next odometry update uses the updated meta distribution that incorporates the path history up to this point.

        self.meta_xy = self.particles_prev[:, :2] * self.meta_weights[:, np.newaxis]  # Update meta xy for the next iteration
        self.meta_cos = np.cos(self.particles_prev[:, 2]) * self.meta_weights  # Update meta cos for the next iteration
        self.meta_sin = np.sin(self.particles_prev[:, 2]) * self.meta_weights  # Update meta sin for the next iteration

        self.accept_odom = True
        self.updated_dist = True
        # rospy.loginfo("Publishing particles")
        #self.weights = self.calculate_weights(self.particles)
        self.weights = self.meta_weights.copy()  # Update self.weights to the new weights for visualization and any other use in the next iteration, so that it reflects the current scan update.
        self.publish_estimate()
        t = time.time()
        self.publish_particles()

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
        
        if self.scan_ranges is None or self.accept_odom == False:
            #print(f"[DEBUG] Odometry update skipped: scan_ranges is None or accept_odom is False (accept_odom={self.accept_odom})")
            return
        
        #rospy.loginfo("Moving particles with odometry using particle prev")
        self.do_mh_random_walk = False  
        self.updated_dist = False  # Flag to indicate that we have not yet updated the meta distribution with the new odometry, so we should not perform the MH random walk in the lidar callback until we have done so, to ensure that the random walk is based on the updated meta distribution that incorporates the path history up to this point.
        self.delta, current_odom = self.get_delta_odom(msg)
        self.last_odom = current_odom  

        # apply motion model and update particles 
        # mh particles updated here and particles (meta-particles in 3MCL) in the lidar callback 
        # after processing the scan with the new path history
        #print(f"[DEBUG] Applying motion model to {len(self.particles_prev)} particles with delta")
        self.particles_prop, _  = apply_motion_model_parallel(self.particles_prev,self.delta,self.alpha,
                                                          self.map_data, self.resolution,
                                                          self.origin_np[0], self.origin_np[1],
                                                          self.width,self.height)
  
        # compute weights for particles before and after motion to use in MH step.
        #print(f"[DEBUG] Calculating weights for MH step with {len(self.particles_prev)} previous particles and {len(self.particles_prop)} proposed particles...")
        self.weights_post = self.calculate_unorm_weights(self.particles_prop)


        # MH step to decide which particles to keep for the next iteration, with update on meta set
        # being made on lidar callback after processing the new scan.

        #print(f"[DEBUG] Performing MH resampling step with {len(self.particles_prev)} previous particles and {len(self.particles_prop)} proposed particles...")
        mh_weights, mh_particles, acc_rate = self.update_particles_mh(self.weights_pre, self.weights_post,
                                                               self.particles_prev, self.particles_prop)

        mh_xy = mh_particles[:, :2]
        mh_cos = np.cos(mh_particles[:, 2])
        mh_sin = np.sin(mh_particles[:, 2])

        # Meta distribution update: we accumulate the accepted particles and their weights across all deltas in the path history for a given
        # scan, so that the meta particles represent a more informed distribution that considers multiple recent movements, not just the
        # last one. This is the core idea of Meta-MH-MCL.

        if self.updated_dist == True:
            #print(f"[DEBUG] New scan received, resetting meta distribution update for odometry updates until next scan.")
            return 
        
        self.particles_prev = mh_particles.copy()  # Update previous particles to the MH result for the next iteration, so that the next odometry update applies the motion model to the MH result that incorporates the path history up to this point.
        self.weights_pre = mh_weights.copy()  # Update previous weights to the MH result for the next iteration, so that the MH step in the next odometry update uses the updated meta distribution that incorporates the path history up to this point.
        # Older samples get exponentially less importance
        self.meta_xy *=  self.meta_decay
        self.meta_cos *=  self.meta_decay
        self.meta_sin *=  self.meta_decay
        self.meta_weights *=  self.meta_decay

        self.meta_xy +=  mh_xy * mh_weights[:, np.newaxis]
        self.meta_cos +=  mh_cos * mh_weights
        self.meta_sin += mh_sin * mh_weights
        self.meta_weights += mh_weights

        self.N_count = 0
        self.acc_rate.publish(Float64(acc_rate))

        #print(f"[DEBUG] Meta distribution updated with decay factor {self.meta_decay:.4f} after odometry update, before MH random walk.")
        self.do_mh_random_walk = True  # Flag to indicate that we should perform MH random walk in the lidar callback after processing the new scan, so that the random walk is based on the updated meta distribution that incorporates the path history up to this point.
        self.mh_random_walk(mh_particles, mh_weights)
        #print(f"[DEBUG] {self.N_count} MH random walk steps completed for current odometry update.")
        
        #print(f"[DEBUG] Meta distribution updated with decay factor {self.meta_decay:.4f} after odometry update.")
        
        #self.particles_prev = mh_particles.copy()  # Update previous particles to the MH result for the next iteration

        #self.weights_pre = mh_weights.copy()  # Update previous weights to the MH result for the next iteration
         
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
            #print("[DEBUG] First odometry received, no motion applied.")
            
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

        if abs(trans) < self.odom_eps:
            rot1 = 0.0
        else:
            rot1 = normalize_angle(np.arctan2(dy, dx) - odom1[2])
        
        if abs(rot1) >= np.pi / 2:
            trans = -trans
            rot1 = normalize_angle(rot1 - np.pi)
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

    
    def mh_random_walk(self, particles, weights):

        #delta = np.zeros(3, dtype=np.float32)
        particles_prev = particles.copy()
        weights_pre = weights.copy()
        alpha_rw = self.alpha_rw 
        #alpha_rw[1:] = self.alpha[1:]*2  # Less noise on translation for random walk to keep it more focused on local exploration

        for i in range(self.Nr):

            #noise = ((np.random.rand(3) - 0.5)*0.001).astype(np.float32)  # Random noise in range [-0.025, 0.025] for each component, can be tuned
            #noise[1] *=10
            #delta = noise # Add small random noise to the odometry delta for the random walk, to encourage exploration around the proposed particles from the MH step. The noise scale can be tuned based on the expected odometry uncertainty and desired exploration level.


            particles_prop = apply_random_walk_parallel(particles_prev,alpha_rw,
                                                        self.map_data, self.resolution,
                                                        self.origin_np[0], self.origin_np[1],
                                                        self.width,self.height, self.Nr)

            #particles_prev[:,2] = particles_prev[:,2]                                   
            
            weights_post = self.calculate_unorm_weights(particles_prop)

            mh_weights, mh_particles, acc_rate = self.update_particles_mh(weights_pre, weights_post,
                                                                particles_prev, particles_prop)

            mh_xy = mh_particles[:, :2]
            mh_cos = np.cos(mh_particles[:, 2])
            mh_sin = np.sin(mh_particles[:, 2])

            if self.accept_odom == False or self.do_mh_random_walk == False or self.updated_dist == True:
                #print(f"[DEBUG] MH random walk stopped early at step {i} due to new odometry or scan update.")
                break

            #self.meta_xy *=  self.meta_decay
            #self.meta_cos *=  self.meta_decay
            #self.meta_sin *=  self.meta_decay
            #self.meta_weights *=  self.meta_decay

            self.meta_xy +=  mh_xy * mh_weights[:, np.newaxis]
            self.meta_cos +=  mh_cos * mh_weights
            self.meta_sin += mh_sin * mh_weights
            self.meta_weights += mh_weights

            particles_prev = mh_particles.copy()  # Update previous particles to the MH result for the next iteration
            weights_pre = mh_weights.copy()  # Update previous weights to the MH result for the next iteration
            self.N_count += 1
            #print(f"[DEBUG] MH random walk step {i+1}/{self.Nr} completed.")
            self.acc_rate.publish(Float64(acc_rate))



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

        parts = np.ascontiguousarray(self.particles, dtype=np.float32)
        w = np.ascontiguousarray(self.weights, dtype=np.float64)
        resampled_particles, _ = low_variance_resample_numba(parts, w, self.num_particles)
        self.particles = resampled_particles


    def resample_amcl_kld(self):

        # ================================================================
        # AMCL random particle injection
        # ================================================================

        ratio = self.w_fast / (self.w_slow + 1e-12)

        p_random = max(0.0, 1.0 - ratio)

        N_target = self.max_particles

        N_random = int(p_random * N_target)

        N_resampled_max = N_target - N_random

        N_prev = len(self.particles)

        # ================================================================
        # Normalize weights BEFORE KLD
        # ================================================================

        weight_sum = np.sum(self.weights)

        if weight_sum <= 1e-12:
            self.weights[:] = 1.0 / len(self.weights)
        else:
            self.weights /= weight_sum

        # ================================================================
        # Adaptive KLD resampling
        # ================================================================

        resampled_particles = kld_sampling_amcl(
            self.particles,
            self.weights,
            self.kld_bin_size_xy,
            self.kld_bin_size_theta,
            self.kld_epsilon,
            self.kld_z,
            N_resampled_max,
            self.min_particles
        )

        # ================================================================
        # Random particle injection
        # ================================================================

        if N_random > 0:

            random_particles = generate_valid_particles(
                N_random,
                self.map_data,
                self.resolution,
                self.origin_np[0],
                self.origin_np[1],
                self.width,
                self.height
            )

            self.particles = np.ascontiguousarray(
                np.vstack((random_particles, resampled_particles)),
                dtype=np.float32
            )

        else:

            self.particles = np.ascontiguousarray(
                resampled_particles,
                dtype=np.float32
            )

        # ================================================================
        # Reset weights uniformly
        # ================================================================

        self.num_particles = len(self.particles)

        self.weights = np.full(
            self.num_particles,
            1.0 / self.num_particles,
            dtype=np.float64
        )

        # ================================================================
        # Debug
        # ================================================================

        print(
            f"[DEBUG] "
            f"w_slow={self.w_slow:.6f} "
            f"w_fast={self.w_fast:.6f} "
            f"ratio={ratio:.6f} "
            f"p_random={p_random:.4f} "
            f"N_random={N_random} "
            f"N_final={self.num_particles}"
        )

        if self.num_particles != N_prev:

            rospy.loginfo(
                f"Particle update!\n"
                f" From: {N_prev}  To: {self.num_particles}"
            )
        

        




    #======================================================================
    # Publish
    #======================================================================

    def publish_particles(self):

        if self.headless:
            return
        
        self._viz_count = getattr(self, '_viz_count', 0) + 1
        if self._viz_count % 1:        # publish on every 3rd call
            return

        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)


        weights = self.weights.copy()
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
        #print(f"[DEBUG] Publishing {len(self.particles)} particles with normalized weights (min: {norm_weights.min():.4f}, max: {norm_weights.max():.4f})")
        marker_id =0
        now = rospy.Time.now()
        for p, w in zip(self.particles, norm_weights):

                
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now
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
        if not self.use_adaptive:
            weights_pre, weights_post = self.update_weights(self.particles_prev, self.particles)
        else:
            weights_pre  = self.calculate_unorm_weights(self.particles_prev)
            weights_post = self.calculate_unorm_weights(self.particles)
        #print(f"[DEBUG] Weight update took {time.time() - t:.4f} seconds")
        
        # 3. MH STEP: Perform MH resampling
        if self.use_mh:
            weights, self.particles, acc_rate = self.update_particles_mh(weights_pre, weights_post)
        else:
            weights = weights_post
            acc_rate = 0.0

        Neff = np.sum(self.weights)**2 / np.sum(self.weights**2)
        #print(f"[DEBUG] Effective sample size (Neff): {Neff:.2f} | Threshold: {self.num_particles / 2.0:.2f}")
        self.Neff_pub.publish(Float64(Neff))

        # 4. RESAMPLE: This is where KLD might change the size for the NEXT frame
        t = time.time()
        if self.use_adaptive:
            self.update_acml_weights(weights)
            if Neff < self.num_particles / 1.0:
                self.resample_amcl_kld()
                self.num_particles = len(self.particles)
        else:
            self.weights = weights
            if Neff < self.num_particles / 2.0:
                self.resample_lvr()
        #print(f"[DEBUG] Resampling took {time.time() - t:.4f} seconds")

        # 5. PUBLISH
        t = time.time()
        self.weights = self.calculate_weights(self.particles)  # Recalculate weights for the new particle set after resampling, to use in the visualization and estimate publication. This is important because resampling changes the particle set and we want the published weights to reflect the current particles.

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