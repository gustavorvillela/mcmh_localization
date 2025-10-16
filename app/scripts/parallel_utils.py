from numba import njit, prange
import numpy as np

@njit
def raycast(pose, angle,max_range, limits, resolution, grid_map, grid_width, grid_height):
  """Raycasting para encontrar obstáculos."""
  x, y = pose[0], pose[1]
  dx = np.cos(angle)
  dy = np.sin(angle)
  step_size = 0.1
  max_steps = int(max_range / step_size)

  for i in range(1, max_steps + 1):
      current_x = x + i * step_size * dx
      current_y = y + i * step_size * dy

      # Converte para coordenadas do grid
      grid_x = int((current_x - limits[0]) / resolution)
      grid_y = int((current_y - limits[2]) / resolution)

      # Verifica limites
      if not (0 <= grid_x < grid_width and 0 <= grid_y < grid_height):
          return max_range

      # Verifica ocupação
      if grid_map[grid_y, grid_x] > 0.5:
          return i * step_size

  return max_range

@njit
def gaussian_prob(diff, sigma):
    return np.exp(-0.5 * (diff / sigma)**2)/np.sqrt(2*np.pi*sigma**2)


@njit
def p_hit(z, z_expected, sigma_hit, max_range):
    """Probabilidade de hit - medição correta."""
    if 0 <= z <= max_range:
        return (1 / (np.sqrt(2*np.pi) * sigma_hit)) * np.exp(-0.5 * ((z - z_expected) / sigma_hit)**2)
    return 0.0

@njit
def p_short(z, z_expected, lambda_short):
    """Probabilidade de short - obstáculo mais próximo do que o esperado."""
    if 0 <= z <= z_expected:
        return lambda_short * np.exp(-lambda_short * z)
    return 0.0

@njit
def p_max(z, max_range):
    """Probabilidade de max - sensor retornando valor máximo."""
    return 1.0 if abs(z - max_range) < 0.001 else 0.0

@njit
def p_rand(z, max_range):
    """Probabilidade de rand - medição aleatória."""
    return 1.0 / max_range if 0 <= z <= max_range else 0.0



@njit
def normalize_angle(theta):
    """
    Normaliza ângulo para o intervalo [-pi, pi].
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi

@njit(parallel=True)
def normalize_angle_array(angles, mean_angle):
    """
    Normaliza vetor de ângulos em relação ao ângulo médio.
    Args:
        angles: (N,) array de ângulos
        mean_angle: float, ângulo médio
    Returns:
        (N,) array de ângulos normalizados
    """
    n = angles.shape[0]
    result = np.empty(n, dtype=np.float32)
    for i in prange(n):
        result[i] = normalize_angle(angles[i] - mean_angle)
    return result

@njit(parallel=True)
def compute_likelihoods(scan_ranges, angles, particles, distance_map,
                        map_resolution, map_origin, width, height,
                        sigma_hit=0.35, z_hit=0.9, z_rand=0.1, max_range=10, step=1):
    
    """
    Compute particle likelihoods using beam model with distance map.
    Args:
        scan_ranges: (M,) array of LIDAR ranges
        angles: (M,) array of LIDAR angles
        particles: (N, 3) array of particle poses [x, y, theta]
        distance_map: (H*W,) flattened distance map
        map_resolution: float, map resolution in meters/cell
        map_origin: (2,) array, map origin [origin_x, origin_y]
        width: int, map width in cells
        height: int, map height in cells
        sigma_hit: float, standard deviation for hit model
        z_hit: float, weight for hit model
        z_rand: float, weight for random model
        max_range: float, maximum sensor range
        step: int, downsampling step for beams
    Returns:
        scores: (N,) array of log-likelihoods for each particle
    """

    N = particles.shape[0]
    scores = np.zeros(N, dtype=np.float32)

    for i in prange(N):
        x, y, theta = particles[i]
        log_score = 0.0
        valid_count = 0

        for j in range(0,len(scan_ranges),step):
            #if j % 10 != 0:  # downsample a cada 10 feixes
            #    continue

            r = scan_ranges[j]
            if np.isfinite(r) and r < max_range:
                valid_count += 1

                lx = x + r * np.cos(theta + angles[j])
                ly = y + r * np.sin(theta + angles[j])
                mx = int((lx - map_origin[0]) / map_resolution)
                my = int((ly - map_origin[1]) / map_resolution)

                if mx < 0 or mx >= width or my < 0 or my >= height:
                    continue   # skip this beam
                index = my * width + mx
                dist = distance_map[index]
                if dist <= max_range:
                    p_hit = np.exp(-0.5 * (dist ** 2) / (sigma_hit ** 2))/np.sqrt(2*np.pi*sigma_hit**2)
                else:
                    p_hit = 0.0
                p_rand = 1.0 / max_range if 0 <= r <= max_range else 0.0
                p = z_hit * p_hit + z_rand * p_rand
                p = max(p, 1e-6)  # evita log(0)
                log_score += np.log(p)

        if valid_count > 0:
            scores[i] = log_score/valid_count 
        else:
            scores[i] = -50  # penaliza partículas cegas

    return scores

@njit(parallel=True)
def compute_likelihoods_raycast(scan_ranges, angles, particles, grid_map, map_resolution, limits):
    """
    Compute particle likelihoods using beam model with raycasting.
    """
    N = particles.shape[0]
    scores = np.zeros(N, dtype=np.float32)

    sigma_hit = 0.05
    z_hit = 0.8
    z_rand = 0.1
    max_range = 10.0

    grid_height, grid_width = grid_map.shape

    for i in prange(N):
        x, y, theta = particles[i]
        log_score = 0.0
        valid_count = 0

        for j in range(len(scan_ranges)):
            r_meas = scan_ranges[j]
            if np.isfinite(r_meas) and r_meas < max_range:
                valid_count += 1

                # Predicted range via raycasting
                r_pred = raycast(
                    np.array([x, y]),
                    theta + angles[j],
                    max_range,
                    limits,
                    map_resolution,
                    grid_map,
                    grid_width,
                    grid_height
                )


                # Probability from Gaussian model
                prob_hit = p_hit(r_meas, r_pred, sigma_hit, max_range)
                prob_rand = p_rand(r_meas, max_range)
                p = z_hit * prob_hit + z_rand * prob_rand
                p = max(p, 1e-6)
                log_score += np.log(p)

        if valid_count > 0:
            scores[i] = log_score / valid_count
        else:
            scores[i] = -np.inf

    return scores


#=======================================================================
# Metropolis-Hastings
#=======================================================================

@njit(parallel=True)
def mh_resampling(particles, proposed_particles, likelihoods, old_weights):

    """
    Metropolis-Hastings resampling step.
    Args:
        particles: (N, 3) array of current particle poses
        proposed_particles: (N, 3) array of proposed particle poses
        likelihoods: (N,) array of likelihoods for proposed particles
        old_weights: (N,) array of weights for current particles
    Returns:
        new_particles: (N, 3) array of resampled particle poses
        new_weights: (N,) array of updated weights
    """

    N = particles.shape[0]
    new_particles = particles.copy()
    new_weights = old_weights.copy()

    for i in prange(N):
        p_old = old_weights[i]
        p_new = likelihoods[i]
        alpha = min(1.0, p_new / p_old) if p_old > 0 else 1.0
        if np.random.rand() < alpha:
            new_particles[i] = proposed_particles[i]
            new_weights[i] = p_new


    return new_particles, new_weights

@njit(parallel=True)
def assym_mh_resampling(particles, proposed_particles, likelihoods, old_weights, trans_forward, trans_backward):

    """
    Asymmetric Metropolis-Hastings resampling step.
    Args:
        particles: (N, 3) array of current particle poses
        proposed_particles: (N, 3) array of proposed particle poses
        likelihoods: (N,) array of likelihoods for proposed particles
        old_weights: (N,) array of weights for current particles
        trans_forward: (N,) array of transition probabilities p(x'|x)
        trans_backward: (N,) array of transition probabilities p(x|x')
    Returns:
        new_particles: (N, 3) array of resampled particle poses
        new_weights: (N,) array of updated weights
    """

    N = particles.shape[0]
    new_particles = particles.copy()
    new_weights = old_weights.copy()

    log_dist_pre = np.log(old_weights + 1e-10)
    log_dist_post = np.log(likelihoods + 1e-10)
    log_trans_forward = np.log(trans_forward + 1e-10)
    log_trans_backward = np.log(trans_backward + 1e-10)

    for i in prange(N):
        
        log_num = log_dist_post[i] + log_trans_backward[i]
        log_den = log_dist_pre[i] + log_trans_forward[i]
        log_alpha = log_num - log_den
        alpha = min(1.0, np.exp(log_alpha)) if log_den > 0 else 1.0

        if np.random.rand() < alpha:
            new_particles[i] = proposed_particles[i]
            new_weights[i] = likelihoods[i]


    return new_particles, new_weights

#=======================================================================
# Funções de motion model
#=======================================================================

@njit(parallel=True)
def motion_model_odometry_parallel(particles_prev, particles_curr, delta, alpha):
    """
    Compute p(x_t | x_{t-1}, u_t) for all particles in parallel using odometry motion model.

    Args:
        particles_prev: (N, 3) previous particle poses [x, y, theta]
        particles_curr: (N, 3) current particle poses [x, y, theta]
        delta: array (rot1, trans, rot2) - observed odometry motion
        alpha: tuple (a1, a2, a3, a4) - motion noise parameters

    Returns:
        probs: (N,) array of motion model probabilities (normalized)
    """
    rot1, trans, rot2 = delta
    a1, a2, a3, a4 = alpha
    N = particles_prev.shape[0]
    probs = np.empty(N, dtype=np.float64)

    for i in prange(N):
        x_prev, y_prev, theta_prev = particles_prev[i]
        x_curr, y_curr, theta_curr = particles_curr[i]

        # Estimated motion from the particle
        dx = x_curr - x_prev
        dy = y_curr - y_prev

        trans_hat = np.sqrt(dx*dx + dy*dy)
        rot1_hat = normalize_angle(np.arctan2(dy, dx) - theta_prev)
        rot2_hat = normalize_angle(theta_curr - theta_prev - rot1_hat)

        # Noise parameters
        sigma_rot1 = a1 * abs(rot1) + a2 * abs(trans)
        sigma_trans = a3 * abs(trans) + a4 * (abs(rot1) + abs(rot2))
        sigma_rot2 = a1 * abs(rot2) + a2 * abs(trans)

        # Gaussian motion likelihoods
        p1 = gaussian_prob(normalize_angle(rot1 - rot1_hat), sigma_rot1)
        p2 = gaussian_prob(trans - trans_hat, sigma_trans)
        p3 = gaussian_prob(normalize_angle(rot2 - rot2_hat), sigma_rot2)

        probs[i] = p1 * p2 * p3

    # Normalize probabilities
    s = np.sum(probs)
    if s > 0:
        probs /= s

    return probs

@njit(parallel=True)
def apply_motion_model_parallel(particles, delta, alpha, map_data, map_resolution, origin_x, origin_y, width, height):
    rot1, trans, rot2 = delta
    a1, a2, a3, a4 = alpha
    num_particles = particles.shape[0]
    new_particles = np.empty_like(particles)

    max_attempts = 1000


    for i in prange(num_particles):
        success = False
        for _ in range(max_attempts):
            r1_hat = rot1 + np.random.normal(0, a1 * abs(rot1) + a2 * abs(trans))
            t_hat = trans + np.random.normal(0, a3 * abs(trans) + a4 * (abs(rot1) + abs(rot2)))
            r2_hat = rot2 + np.random.normal(0, a1 * abs(rot2) + a2 * abs(trans))


            x, y, theta = particles[i]
            x_new = x + t_hat * np.cos(theta + r1_hat)
            y_new = y + t_hat * np.sin(theta + r1_hat)
            theta_new = normalize_angle(theta + r1_hat + r2_hat)

            if is_valid_position(x_new, y_new, map_data, width, height, map_resolution, origin_x, origin_y):
                new_particles[i] = [x_new, y_new, theta_new]
                success = True
                break

        if not success:
            new_particles[i] = particles[i]  # fallback: mantém partícula antiga

    return new_particles

#=======================================================================
# Funções de resample e validação
#=======================================================================

@njit
def compute_valid_indices(particles, map_data, map_resolution, origin_x, origin_y,width, height):
    num_particles = particles.shape[0]
    valid_indices = []


    for i in range(num_particles):
        x, y = particles[i, 0], particles[i, 1]
        mx = int((x - origin_x) / map_resolution)
        my = int((y - origin_y) / map_resolution)

        if 0 <= mx < width and 0 <= my < height:
            index = my * width + mx

            if map_data[index] <= 10:  # livre
                valid_indices.append(i)

    return np.array(valid_indices, dtype=np.int32)

@njit
def is_valid_position(x, y, map_data, width, height, map_resolution, origin_x, origin_y):
    mx = int((x - origin_x) / map_resolution)
    my = int((y - origin_y) / map_resolution)

    if 0 <= mx < width and 0 <= my < height:
        index = my * width + mx
        return map_data[index] == 0
    return False

@njit(parallel=True)
def compute_valid_mask(particles, map_data, width, height, resolution, origin_x, origin_y):
    num_particles = particles.shape[0]
    valid_mask = np.zeros(num_particles, dtype=np.bool_)

    for i in prange(num_particles):
        x, y = particles[i, 0], particles[i, 1]
        map_x = int((x - origin_x) / resolution)
        map_y = int((y - origin_y) / resolution)

        if 0 <= map_x < width and 0 <= map_y < height:
            index = map_y * width + map_x
            if map_data[index] == 0:  
                valid_mask[i] = True

    return valid_mask


@njit
def low_variance_resample_numba(particles, weights, N):

    """ 
    Low-variance resampling algorithm (stochastic universal sampling).
    Args:
        particles: (N, 3) array of particle poses
        weights: (N,) array of particle weights
        N: int, number of particles to resample
    Returns:
        new_particles: (N, 3) array of resampled particle poses
        new_weights: (N,) array of uniform weights after resampling 
    """

    weights = weights / np.sum(weights)  # ensure normalization
    new_particles = np.zeros_like(particles)
    new_weights = np.full(N, 1.0 / N, dtype=np.float32)

    step = 1.0 / N
    r = np.random.uniform(0.0, step)
    c = weights[0]
    i = 0

    for m in range(N):
        U = r + m * step
        while U > c and i < N - 1:
            i += 1
            c += weights[i]
        new_particles[m] = particles[i]

    return new_particles, new_weights



@njit
def generate_valid_particles(num_particles,
                             map_data, map_resolution, origin_x, origin_y,width,height):
    max_trials = max(50 * num_particles, 500)
    x = np.random.uniform(origin_x, origin_x + width * map_resolution, size=max_trials)
    y = np.random.uniform(origin_y, origin_y + height * map_resolution, size=max_trials)
    theta = np.random.uniform(-np.pi, np.pi, size=max_trials)
    
    all_particles = np.column_stack((x, y, theta))
    valid_mask = compute_valid_mask(all_particles, map_data, width, height, map_resolution, origin_x, origin_y)
    valid_particles = all_particles[valid_mask]

    if valid_particles.shape[0] >= num_particles:
        return valid_particles[:num_particles]
    else:
        return valid_particles

@njit(parallel=True)
def parallel_resample_simple(particles, weights, N):
    new_particles = np.empty_like(particles)
    cum_weights = np.cumsum(weights)
    
    for i in prange(N):
        u = np.random.rand()
        idx = np.searchsorted(cum_weights, u)
        new_particles[i] = particles[idx]
    
    return new_particles

#=============
# AMCL
#=============

from numba import njit, prange
import numpy as np

@njit
def low_variance_resample_amcl(particles, weights, target_size):
    N = len(particles)
    new_particles = np.empty((target_size, 3), dtype=np.float32)
    
    r = np.random.uniform(0.0, 1.0 / target_size)
    c = weights[0]
    i = 0

    for m in range(target_size):
        U = r + m / target_size
        while U > c and i < N - 1:
            i += 1
            c += weights[i]
        new_particles[m] = particles[i % N]  # % N para evitar overflow

    return new_particles, np.full(target_size, 1.0/target_size)

@njit(parallel=True)
def reinitialize_particles_numba(num_new, occupancy_map, res, origin_x, origin_y):
    new_particles = np.empty((num_new, 3), dtype=np.float32)
    valid_cells = np.argwhere(occupancy_map == 0)  # Pré-computa células válidas
    
    # Caso não haja células válidas (improvável)
    if len(valid_cells) == 0:
        for i in prange(num_new):
            new_particles[i] = np.array([origin_x, origin_y, np.random.uniform(-np.pi, np.pi)])
        return new_particles
    
    for i in prange(num_new):
        # Amostra diretamente de células válidas
        idx = np.random.randint(0, len(valid_cells))
        my, mx = valid_cells[idx]  # Note a ordem (y,x)
        
        x = mx * res + origin_x
        y = my * res + origin_y
        theta = np.random.uniform(-np.pi, np.pi)
        
        new_particles[i] = np.array([x, y, theta])
                
    return new_particles


@njit
def kld_sampling_amcl(particles, weights, bin_size_xy, bin_size_theta,
                       epsilon, z, max_samples, min_particles):
    
    '''
    KLD-sampling para AMCL.
    Resampling com critério de parada baseado em Kullback-Leibler Divergence.
    Args:
        particles: (N, 3) array de partículas
        weights: (N,) array de pesos normalizados
        bin_size_xy: tamanho da célula em x e y (metros)
        bin_size_theta: tamanho da célula em theta (radianos)   
        epsilon: erro máximo permitido
        z: valor z para intervalo de confiança
        max_samples: número máximo de amostras
        min_particles: número mínimo de partículas
    Returns:
        sampled_particles: (M, 3) array de partículas amostradas
        
    '''
    bins = set()
    sampled_particles = np.empty((max_samples, 3), dtype=np.float32)
    count = 0
    noise_std = np.array([0.001, 0.001, 0.02], dtype=np.float64)
    
    r = np.random.uniform(0, 1/max_samples)
    c = weights[0]
    i = 0
    
    while count < max_samples:
        # Low-variance sampling
        u = r + count / max_samples
        while u > c and i < len(weights)-1:
            i += 1
            c += weights[i]
        
        p = particles[i]
        
        # Adiciona ruído gaussiano
        noisy_particle = np.empty(3, dtype=np.float64)
        for j in range(3):
            noisy_particle[j] = p[j] + np.random.normal(0, noise_std[j])
        
        # Atualiza bins
        x_bin = int(noisy_particle[0] / bin_size_xy)
        y_bin = int(noisy_particle[1] / bin_size_xy)
        theta_bin = int(noisy_particle[2] / bin_size_theta)
        bin_id = (x_bin, y_bin, theta_bin)
        
        if bin_id not in bins:
            bins.add(bin_id)
            k = len(bins)
            
            # Critério de parada KLD
            if k > 1 and count >= min_particles:
                chi2 = (k-1)*(1 - 2/(9*(k-1)) + np.sqrt(2/(9*(k-1)))*z)**3
                if count > chi2/(2*epsilon):
                    break
        
        sampled_particles[count] = noisy_particle
        count += 1
    
    return sampled_particles[:count]  # Retorna apenas as amostradas


def initialize_gaussian_parallel(mean, cov, num_particles, distance_map, resolution, origin):
    samples = np.random.multivariate_normal(mean, cov, size=num_particles)
    particles = validate_samples(samples, distance_map, resolution, origin)
    return particles

@njit(parallel=True)
def validate_samples(samples, distance_map, resolution, origin):
    num_particles = samples.shape[0]
    valid_particles = np.empty((num_particles, 3), dtype=np.float64)

    for i in prange(num_particles):
        sample = samples[i]
        mx = int((sample[0] - origin[0]) / resolution)
        my = int((sample[1] - origin[1]) / resolution)

        # Se inválido, zera — pode ajustar para lidar depois
        if 0 <= mx < distance_map.shape[1] and 0 <= my < distance_map.shape[0] and distance_map[my, mx] < 1.0:
            valid_particles[i] = sample
        else:
            valid_particles[i] = np.array([0.0, 0.0, 0.0])  # Pode fazer pós-processamento depois

    return valid_particles