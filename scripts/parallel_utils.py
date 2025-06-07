from numba import njit, prange
import numpy as np

@njit
def normalize_angle(theta):
    while theta > np.pi:
        theta -= 2 * np.pi
    while theta < -np.pi:
        theta += 2 * np.pi
    return theta

@njit(parallel=True)
def compute_likelihoods(scan_ranges, angles, particles, distance_map, map_resolution, map_origin):
    N = particles.shape[0]
    scores = np.zeros(N, dtype=np.float32)

    for i in prange(N):
        x, y, theta = particles[i]
        score = 0.0
        for j in range(len(scan_ranges)):
            r = scan_ranges[j]
            if np.isfinite(r) and r < 10.0:
                lx = x + r * np.cos(theta + angles[j])
                ly = y + r * np.sin(theta + angles[j])
                mx = int((lx - map_origin[0]) / map_resolution)
                my = int((ly - map_origin[1]) / map_resolution)

                if 0 <= mx < distance_map.shape[1] and 0 <= my < distance_map.shape[0]:
                    dist = distance_map[my, mx]
                    score += np.exp(-dist**2 / 0.1)
        scores[i] = score

    return scores

@njit(parallel=True)
def mh_resampling(particles, proposed_particles, likelihoods, old_weights):
    N = particles.shape[0]
    new_particles = particles.copy()
    new_weights = old_weights.copy()

    for i in prange(N):
        p_old = old_weights[i]
        p_new = likelihoods[i]
        #alpha = min(1.0, p_new / p_old) if p_old > 0 else 1.0
        alpha = 1
        if np.random.rand() < alpha:
            new_particles[i] = proposed_particles[i]
            new_weights[i] = p_new

    return new_particles, new_weights

@njit(parallel=True)
def apply_motion_model_parallel(particles, delta, alpha):
    rot1, trans, rot2 = delta
    a1, a2, a3, a4 = alpha
    num_particles = particles.shape[0]
    new_particles = np.empty_like(particles)

    for i in prange(num_particles):
        r1_hat = rot1 + np.random.normal(0, a1 * abs(rot1) + a2 * abs(trans))
        t_hat = trans + np.random.normal(0, a3 * abs(trans) + a4 * (abs(rot1) + abs(rot2)))
        r2_hat = rot2 + np.random.normal(0, a1 * abs(rot2) + a2 * abs(trans))

        x, y, theta = particles[i]
        x_new = x + t_hat * np.cos(theta + r1_hat)
        y_new = y + t_hat * np.sin(theta + r1_hat)
        theta_new = normalize_angle(theta + r1_hat + r2_hat)

        new_particles[i] = [x_new, y_new, theta_new]
    
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
            if occupancy_map[my, mx] == 0:  # livre
                valid_indices.append(i)

    return np.array(valid_indices, dtype=np.int32)


@njit
def generate_valid_particles(num_particles, min_coords, max_coords,
                       occupancy_map, map_resolution, origin_x, origin_y):
    max_trials = num_particles * 10  # limite de tentativas
    all_particles = np.zeros((max_trials, 3), dtype=np.float32)

    for i in range(max_trials):
        all_particles[i, 0] = np.random.uniform(min_coords[0], max_coords[0])
        all_particles[i, 1] = np.random.uniform(min_coords[1], max_coords[1])
        all_particles[i, 2] = normalize_angle(np.random.uniform(-np.pi, np.pi))

    valid_indices = compute_valid_indices(all_particles, occupancy_map, map_resolution, origin_x, origin_y)

    if len(valid_indices) < num_particles:
        # fallback: retorna os válidos que conseguiu
        return all_particles[valid_indices[:num_particles]]

    selected_indices = valid_indices[:num_particles]
    return all_particles[selected_indices]