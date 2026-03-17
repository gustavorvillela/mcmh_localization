#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_particles(filename):
    match = re.search(r'_(\d+)p_', filename)
    return int(match.group(1)) if match else None

def extract_algorithm(filename):
    parts = filename.replace('.txt', '').split('_')
    for algo in ['MCL', 'AMCL', 'MHMCL', 'MHAMCL', 'AMHMCL', 'AMHAMCL']:
        if algo in parts:
            return algo
    return None


############################################
# NEW FUNCTION
############################################

def rebuild_error_file_from_pose(pose_path, results_dir):
    """
    Reconstructs the error file from poses file.
    """
    base = os.path.basename(pose_path).replace("poses_", "")
    error_path = os.path.join(results_dir, base)

    times = []
    errors = []

    with open(pose_path) as f:
        next(f)  # skip header
        for line in f:
            vals = line.strip().split(",")
            if len(vals) != 7:
                continue

            t, est_x, est_y, est_yaw, gt_x, gt_y, gt_yaw = map(float, vals)

            error = np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2)

            times.append(t)
            errors.append(error)

    if not errors:
        return None

    rmse = np.sqrt(np.mean(np.square(errors)))

    with open(error_path, "w") as f:
        f.write("time,error\n")
        for t, e in zip(times, errors):
            f.write(f"{t:.3f},{e:.4f}\n")

        f.write(f"\nRMSE final: {rmse:.4f}\n")

    print(f"Rebuilt {base} RMSE={rmse:.4f}")

    return rmse


############################################
# ORIGINAL RMSE READER
############################################

def extract_rmse(filepath):
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("RMSE final:"):
                    return float(line.split(":")[1].strip())
    except Exception as e:
        print(f"Erro lendo {filepath}: {e}")
    return None


def main():

    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    ############################################
    # STEP 1: rebuild missing error files
    ############################################

    for filename in os.listdir(results_dir):

        if filename.startswith("poses_") and filename.endswith(".txt"):

            pose_path = os.path.join(results_dir, filename)

            base = filename.replace("poses_", "")
            error_path = os.path.join(results_dir, base)

            if not os.path.exists(error_path):
                rebuild_error_file_from_pose(pose_path, results_dir)

    ############################################
    # STEP 2: normal processing
    ############################################

    data = defaultdict(lambda: defaultdict(list))

    for filename in os.listdir(results_dir):

        if filename.endswith(".txt") and not filename.startswith("poses_"):

            algo = extract_algorithm(filename)
            particles = extract_particles(filename)

            if algo and particles:

                rmse = extract_rmse(os.path.join(results_dir, filename))

                if rmse is not None:
                    data[algo][particles].append(rmse)

    if not data:
        print("Nenhum dado válido encontrado.")
        return

    avg_data = {}
    for algo, p_dict in data.items():
        avg_data[algo] = {
            p: (np.mean(rmses), np.std(rmses))
            for p, rmses in sorted(p_dict.items())
        }

    styles = {
        'MCL': {'color': '#000000', 'linestyle': '-', 'marker': 'o', 'label': 'MCL'},
        'AMCL': {'color': '#1f77b4', 'linestyle': ':', 'marker': 'o', 'label': 'AMCL'},
        'MHMCL': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 'o', 'label': 'MHMCL'},
        'MHAMCL': {'color': '#2ca02c', 'linestyle': '-.', 'marker': 'o', 'label': 'MHAMCL'},
        'AMHMCL': {'color': '#b4331f', 'linestyle': '-', 'marker': 'o', 'label': 'AMHMCL'},
        'AMHAMCL': {'color': '#9467bd', 'linestyle': '--', 'marker': 'o', 'label': 'AMHAMCL'}
    }

    plot_path = os.path.join(plots_dir, "particle_sweep_rmse.png")

    plt.figure(figsize=(8, 6))
    plt.title("RMSE vs Número de Partículas")
    plt.xlabel("Número de Partículas")
    plt.ylabel("RMSE (m)")

    for algo, results in avg_data.items():

        particles = sorted(results.keys())
        means = [results[p][0] for p in particles]
        stds = [results[p][1] for p in particles]

        style = styles.get(algo, {'color': '#666666', 'linestyle': '-', 'marker': 'o', 'label': algo})

        plt.errorbar(
            particles,
            means,
            yerr=stds,
            label=style['label'],
            color=style['color'],
            linestyle=style['linestyle'],
            marker=style['marker'],
            linewidth=2,
            capsize=4
        )

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Gráfico salvo em: {plot_path}")


if __name__ == "__main__":
    main()