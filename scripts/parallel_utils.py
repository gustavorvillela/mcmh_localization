from numba import njit, prange
import numpy as np

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
    """
    n = angles.shape[0]
    result = np.empty(n, dtype=np.float32)
    for i in prange(n):
        result[i] = normalize_angle(angles[i] - mean_angle)
    return result

@njit(parallel=True)
def compute_likelihoods(scan_ranges, angles, particles, distance_map, map_resolution, map_origin):
    N = particles.shape[0]
    scores = np.zeros(N, dtype=np.float32)

    sigma_hit = 1.2         # maior tolerância → pesos mais "generosos"
    z_hit = 0.95
    z_rand = 0.15
    max_range = 10.0

    for i in prange(N):
        x, y, theta = particles[i]
        log_score = 0.0
        valid_count = 0

        for j in range(len(scan_ranges)):
            #if j % 10 != 0:  # downsample a cada 5 feixes
            #    continue

            r = scan_ranges[j]
            if np.isfinite(r) and r < max_range:
                valid_count += 1

                lx = x + r * np.cos(theta + angles[j])
                ly = y + r * np.sin(theta + angles[j])
                mx = int((lx - map_origin[0]) / map_resolution)
                my = int((ly - map_origin[1]) / map_resolution)

                if 0 <= mx < distance_map.shape[1] and 0 <= my < distance_map.shape[0]:
                    dist = distance_map[my, mx]
                    p_hit = np.exp(-0.5 * (dist ** 2) / (sigma_hit ** 2))
                else:
                    p_hit = 0.0

                p = z_hit * p_hit + z_rand * (1.0 / max_range)
                p = max(p, 1e-6)  # evita log(0)
                log_score += np.log(p)

        if valid_count > 0:
            scores[i] = log_score / (valid_count* sigma_hit**2)
        else:
            scores[i] = -np.inf  # penaliza partículas cegas

    return scores





@njit(parallel=True)
def mh_resampling(particles, proposed_particles, likelihoods, old_weights):
    N = particles.shape[0]
    new_particles = particles.copy()
    new_weights = old_weights.copy()

    for i in prange(N):
        p_old = old_weights[i]
        p_new = likelihoods[i]
        alpha = min(1.0, p_new / p_old) if p_old > 0 else 1.0
        #alpha = 1
        if np.random.rand() < alpha:
            new_particles[i] = proposed_particles[i]
            new_weights[i] = p_new

    return new_particles, new_weights

@njit(parallel=True)
def apply_motion_model_parallel(particles, delta, alpha, occupancy_map, map_resolution, origin_x, origin_y):
    rot1, trans, rot2 = delta
    a1, a2, a3, a4 = alpha
    num_particles = particles.shape[0]
    new_particles = np.empty_like(particles)

    max_attempts = 10

    #if abs(trans) < 1e-3 and abs(rot1) < 1e-3 and abs(rot2) < 1e-3:
    #    return particles.copy()

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

            if is_valid_position(x_new, y_new, occupancy_map, map_resolution, origin_x, origin_y):
                new_particles[i] = [x_new, y_new, theta_new]
                success = True
                break

        if not success:
            new_particles[i] = particles[i]  # fallback: mantém partícula antiga

    return new_particles


@njit
def compute_valid_indices(particles, occupancy_map, map_resolution, origin_x, origin_y):
    num_particles = particles.shape[0]
    valid_indices = []

    for i in range(num_particles):
        x, y = particles[i, 0], particles[i, 1]
        mx = int((x - origin_x) / map_resolution)
        my = int((y - origin_y) / map_resolution)

        if 0 <= mx < occupancy_map.shape[1] and 0 <= my < occupancy_map.shape[0]:
            if occupancy_map[my, mx] <= 10:  # livre
                valid_indices.append(i)

    return np.array(valid_indices, dtype=np.int32)

@njit
def is_valid_position(x, y, occupancy_map, map_resolution, origin_x, origin_y):
    mx = int((x - origin_x) / map_resolution)
    my = int((y - origin_y) / map_resolution)

    if 0 <= mx < occupancy_map.shape[1] and 0 <= my < occupancy_map.shape[0]:
        return occupancy_map[my, mx] == 0
    return False

@njit(parallel=True)
def compute_valid_mask(particles, occupancy_map, map_resolution, origin_x, origin_y):
    num_particles = particles.shape[0]
    valid_mask = np.zeros(num_particles, dtype=np.bool_)

    height, width = occupancy_map.shape

    for i in prange(num_particles):
        x, y = particles[i, 0], particles[i, 1]
        mx = int(np.floor((x - origin_x) / map_resolution))
        my = int(np.floor((y - origin_y) / map_resolution))

        if 0 <= mx < width and 0 <= my < height:
            if occupancy_map[my, mx] <= 10:  # ou 50, dependendo do seu uso
                valid_mask[i] = True

    return valid_mask


@njit
def low_variance_resample_numba(particles, weights):
    N = particles.shape[0]
    new_particles = np.empty_like(particles)
    new_weights = np.full(N, 1.0 / N, dtype=np.float32)

    r = np.random.uniform(0.0, 1.0 / N)
    c = weights[0]
    i = 0

    for m in range(N):
        U = r + m / N
        while U > c and i < N - 1:  # Proteção contra índice fora do vetor
            i += 1
            c += weights[i]
        new_particles[m] = particles[i]

    return new_particles, new_weights


@njit
def generate_valid_particles(num_particles, min_coords, max_coords,
                             occupancy_map, map_resolution, origin_x, origin_y):
    max_trials = num_particles * 10

    # Geração de partículas aleatórias
    x = np.random.uniform(min_coords[0], max_coords[0], size=max_trials)
    y = np.random.uniform(min_coords[1], max_coords[1], size=max_trials)
    theta = np.random.uniform(-np.pi, np.pi, size=max_trials)

    all_particles = np.column_stack((x, y, theta))
    valid_mask = compute_valid_mask(all_particles, occupancy_map, map_resolution, origin_x, origin_y)

    valid_particles = all_particles[valid_mask]

    if valid_particles.shape[0] >= num_particles:
        return valid_particles[:num_particles]
    else:
        return valid_particles  # menos do que o pedido, mas o que deu
    


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