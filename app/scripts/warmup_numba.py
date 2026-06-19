#!/usr/bin/env python3
"""
Standalone warmup for the Numba (@njit) functions in parallel_utils.

All kernels are decorated with cache=True, so the first compilation writes
an on-disk cache. Running this once (e.g. before launching the localizer or
as a build/setup step) pays the JIT cost up front; later imports load the
compiled code from cache instead of recompiling.

Numba caches per concrete argument-type signature, so the dummy inputs below
MUST use the same dtypes/shapes the localizer uses at runtime, otherwise a
fresh specialization gets compiled on the first real call and the warmup is
wasted. The dtypes here mirror amcmh_localizer.py:

    particles      float32      (self.particles)
    weights        float64      (np.ones(N)/N)
    map_data       int8 (flat)  (self.map_data)
    distance_map   float32 flat (self.distance_map)
    dist_2d        float64 2D   (self.dist_2d, used by validate_samples)
    origin_np      float64      (self.origin_np)
    alpha/alpha_rw float32      (self.alpha, self.alpha_rw)
    scan/angles    float32
    delta          float32

If you change a call signature in the real code, mirror it here.

Usage:
    python3 warmup_numba.py
"""

import time
import numpy as np

import parallel_utils as pu


def _timed(label, fn):
    t0 = time.perf_counter()
    try:
        fn()
        dt = time.perf_counter() - t0
        print(f"  [ok]   {label:<32} {dt*1000:8.1f} ms")
    except Exception as e:  # keep going so one failure doesn't abort the rest
        dt = time.perf_counter() - t0
        print(f"  [FAIL] {label:<32} {dt*1000:8.1f} ms  -> {type(e).__name__}: {e}")


def main():
    print("Warming up Numba kernels in parallel_utils ...")
    t_start = time.perf_counter()

    # --- shared dummy map / particle data (runtime dtypes) ----------------
    width, height = 40, 30
    resolution = 0.05
    origin_x, origin_y = -1.0, -1.0
    origin_np = np.array([origin_x, origin_y], dtype=np.float64)

    map_data = np.zeros(width * height, dtype=np.int8)          # flat int8
    distance_map = np.ones(width * height, dtype=np.float32)    # flat float32
    dist_2d = np.ones((height, width), dtype=np.float64)        # 2D float64

    N = 8
    particles = np.zeros((N, 3), dtype=np.float32)
    weights = np.full(N, 1.0 / N, dtype=np.float64)

    M = 12
    scan = np.ones(M, dtype=np.float32)
    angles = np.linspace(-1.0, 1.0, M).astype(np.float32)

    alpha = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    alpha_rw = np.array([0.02, 0.04, 0.01, 0.01], dtype=np.float32)
    delta = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # sensor model scalar params (defaults from amcmh_localizer.py)
    sigma_hit, z_hit, z_rand, max_range = 0.2, 0.8, 0.2, 10.0
    step, z_short, z_max, lambda_short = 1, 0.05, 0.05, 0.1
    kld_bin_xy, kld_bin_theta, kld_eps, kld_z = 0.1, np.deg2rad(10), 0.025, 2

    # ======================================================================
    # Runtime-critical kernels (imported / called by amcmh_localizer.py)
    # ======================================================================
    print("\n[runtime-critical kernels]")

    _timed("apply_motion_model_parallel", lambda: pu.apply_motion_model_parallel(
        particles, delta, alpha, map_data, resolution,
        origin_np[0], origin_np[1], width, height))

    _timed("apply_random_walk_parallel", lambda: pu.apply_random_walk_parallel(
        particles, alpha_rw, map_data, resolution,
        origin_np[0], origin_np[1], width, height, 1))

    _timed("compute_likelihoods", lambda: pu.compute_likelihoods(
        scan, angles, particles, distance_map, resolution, origin_np,
        width, height, sigma_hit, z_hit, z_rand, max_range,
        step, z_short, z_max, lambda_short))

    _timed("mh_resampling", lambda: pu.mh_resampling(
        particles, particles.copy(), weights, weights))

    _timed("motion_model_odometry_parallel", lambda: pu.motion_model_odometry_parallel(
        particles, particles, delta, alpha))

    _timed("normalize_angle", lambda: pu.normalize_angle(3.5))
    _timed("normalize_angle_array", lambda: pu.normalize_angle_array(
        np.zeros(N, dtype=np.float32), 0.0))

    _timed("low_variance_resample_numba", lambda: pu.low_variance_resample_numba(
        particles, weights, N))
    _timed("parallel_resample_simple", lambda: pu.parallel_resample_simple(
        particles, weights, N))

    # meta-particle accumulation (3MCL / meta modes)
    meta_xy = np.zeros((N, 2), dtype=np.float64)
    meta_cos = np.zeros(N, dtype=np.float64)
    meta_sin = np.zeros(N, dtype=np.float64)
    meta_w = np.ones(N, dtype=np.float64)
    _timed("accumulate_meta_particles", lambda: pu.accumulate_meta_particles(
        particles, weights, meta_xy, meta_cos, meta_sin, meta_w))
    _timed("finalize_meta_particles", lambda: pu.finalize_meta_particles(
        meta_xy, meta_cos, meta_sin, meta_w))

    # AMCL modes
    _timed("kld_sampling_amcl", lambda: pu.kld_sampling_amcl(
        particles, weights, kld_bin_xy, kld_bin_theta, kld_eps, kld_z, 50, 5))
    _timed("generate_valid_particles", lambda: pu.generate_valid_particles(
        2 * N, map_data, resolution, origin_np[0], origin_np[1], width, height))

    # gaussian init helper (validate_samples runs on dist_2d, float64 particles)
    _timed("validate_samples", lambda: pu.validate_samples(
        particles.astype(np.float64), dist_2d, resolution, origin_np))

    # internal helpers warmed implicitly above: gaussian_prob, is_valid_position,
    # compute_valid_mask. Touch the standalone scalar one for completeness:
    _timed("gaussian_prob", lambda: pu.gaussian_prob(0.1, sigma_hit))

    # ======================================================================
    # Not imported by the current localizer (raycast path / AMCL variants /
    # asymmetric MH). Warmed best-effort so the cache is fully primed; dtypes
    # are reasonable guesses, not pinned to a live call site.
    # ======================================================================
    print("\n[not currently called by amcmh_localizer.py]")

    limits = np.array([origin_x, origin_x + width * resolution,
                       origin_y, origin_y + height * resolution], dtype=np.float64)
    grid_map = np.zeros((height, width), dtype=np.float64)

    _timed("raycast", lambda: pu.raycast(
        np.array([0.0, 0.0]), 0.0, max_range, limits, resolution,
        grid_map, width, height))
    _timed("compute_likelihoods_raycast", lambda: pu.compute_likelihoods_raycast(
        scan, angles, particles, grid_map, resolution, limits))
    _timed("p_hit", lambda: pu.p_hit(1.0, 1.0, 0.05, max_range))
    _timed("p_short", lambda: pu.p_short(0.5, 1.0, lambda_short))
    _timed("p_max", lambda: pu.p_max(max_range, max_range))
    _timed("p_rand", lambda: pu.p_rand(1.0, max_range))
    _timed("assym_mh_resampling", lambda: pu.assym_mh_resampling(
        particles, particles.copy(), weights, weights,
        np.full(N, 0.5, dtype=np.float64), np.full(N, 0.5, dtype=np.float64)))
    _timed("compute_valid_indices", lambda: pu.compute_valid_indices(
        particles, map_data, resolution, origin_x, origin_y, width, height))
    _timed("is_valid_position", lambda: pu.is_valid_position(
        0.0, 0.0, map_data, width, height, resolution, origin_x, origin_y))
    _timed("compute_valid_mask", lambda: pu.compute_valid_mask(
        particles, map_data, width, height, resolution, origin_x, origin_y))
    _timed("low_variance_resample_amcl", lambda: pu.low_variance_resample_amcl(
        particles, weights, N))
    _timed("reinitialize_particles_numba", lambda: pu.reinitialize_particles_numba(
        N, grid_map, resolution, origin_x, origin_y))

    total = time.perf_counter() - t_start
    print(f"\nDone. Total warmup time: {total:.2f} s")
    print("Subsequent runs should load these kernels from the Numba cache.")


if __name__ == "__main__":
    main()
