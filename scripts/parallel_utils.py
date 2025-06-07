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

    # Parâmetros do modelo AMCL
    sigma_hit = 0.02         # desvio padrão do ruído gaussiano (em metros)
    z_hit = 0.95            # peso da componente gaussiana
    z_rand = 0.05           # peso da componente aleatória
    max_range = 10.0        # alcance máximo do laser
    valid_count = 0 

    for i in prange(N):
        x, y, theta = particles[i]
        score = 0.0

        for j in range(len(scan_ranges)):
            r = scan_ranges[j]

            if np.isfinite(r) and r < max_range:
                valid_count += 1
                # Coordenada do feixe no mundo
                lx = x + r * np.cos(theta + angles[j])
                ly = y + r * np.sin(theta + angles[j])

                # Converte para índice de mapa
                mx = int((lx - map_origin[0]) / map_resolution)
                my = int((ly - map_origin[1]) / map_resolution)

                # Se dentro do mapa, calcula distância
                if 0 <= mx < distance_map.shape[1] and 0 <= my < distance_map.shape[0]:
                    dist = distance_map[my, mx]
                    p_hit = np.exp(-0.5 * (dist ** 2) / (sigma_hit ** 2))
                else:
                    p_hit = 0.0  # fora do mapa → distância infinita

                # Mistura com ruído aleatório
                p = z_hit * p_hit + z_rand * (1.0 / max_range)
                score += p

        scores[i] = score
        scores[i] = score / max(valid_count, 1)

    return scores


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

    if abs(trans) < 1e-3 and abs(rot1) < 1e-3 and abs(rot2) < 1e-3:
        return particles.copy()

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
            if occupancy_map[my, mx] == 0:  # livre
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
        mx = int((x - origin_x) / map_resolution)
        my = int((y - origin_y) / map_resolution)

        if 0 <= mx < width and 0 <= my < height:
            if occupancy_map[my, mx] == 0:
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
    max_trials = num_particles * 10  # limite de tentativas
    all_particles = np.zeros((max_trials, 3), dtype=np.float32)

    for i in range(max_trials):
        all_particles[i, 0] = np.random.uniform(min_coords[0], max_coords[0])
        all_particles[i, 1] = np.random.uniform(min_coords[1], max_coords[1])
        all_particles[i, 2] = normalize_angle(np.random.uniform(-np.pi, np.pi))

    # Obtem máscara booleana dos válidos (paralelizada)
    valid_mask = compute_valid_mask(all_particles, occupancy_map, map_resolution, origin_x, origin_y)

    # Conta quantos são válidos
    count_valid = 0
    for i in range(max_trials):
        if valid_mask[i]:
            count_valid += 1

    # Se não tem válidos suficientes, retorna o que tem (ou um array vazio)
    if count_valid < num_particles:
        # coleta os válidos
        valid_particles = np.zeros((count_valid, 3), dtype=np.float32)
        idx = 0
        for i in range(max_trials):
            if valid_mask[i]:
                valid_particles[idx] = all_particles[i]
                idx += 1
        return valid_particles

    # Se tem suficientes, seleciona os primeiros num_particles válidos
    selected_particles = np.zeros((num_particles, 3), dtype=np.float32)
    idx = 0
    for i in range(max_trials):
        if valid_mask[i]:
            selected_particles[idx] = all_particles[i]
            idx += 1
            if idx == num_particles:
                break

    return selected_particles