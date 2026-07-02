#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import statsmodels.api as sm

list_algos = ['MCL', 'AMCL', 'MHMCL', 'MHAMCL', 'AMHMCL', 'AMHAMCL', '3MCL']

GT_X_OFFSET = float(os.environ.get("MCMH_GT_X_OFFSET", "0.7"))
RECALL_THRESHOLDS = {
    "recall_t1": (0.25, np.deg2rad(2.0)),
    "recall_t2": (0.50, np.deg2rad(5.0)),
    "recall_t3": (5.00, np.deg2rad(10.0)),
}
FAILURE_POS_THRESHOLD = 5.0
FAILURE_YAW_THRESHOLD = np.deg2rad(10.0)
METRIC_KEYS = [
    "pos",
    "yaw",
    "success",
    "spl",
    "recall_t1",
    "recall_t2",
    "recall_t3",
    "failure_rate",
]

def extract_particles(filename):
    match = re.search(r'_(\d+)p_', filename)
    return int(match.group(1)) if match else None

def extract_algorithm(filename):
    parts = filename.replace('.txt', '').split('_')
    for algo in list_algos:
        if algo in parts:
            return algo
    return None

def extract_scenario(filename):
    name = filename.replace(".txt", "")

    # remove poses_ prefix if present
    name = name.replace("poses_", "")

    # remove neff_ prefix if present
    name = name.replace("neff_", "")

    # remove particle specification
    name = re.sub(r'_\d+p_', '_', name)

    # remove algorithm names
    for algo in list_algos:
        name = name.replace("_" + algo, "")

    # remove run index if present
    name = re.sub(r'_run\d+', '', name)

    return name.strip("_")

# Action: Extract witch run procude the result from the file name
# I/ filename: String
# O/ run: String
# Necessity: A filename where every data is separate by "_" and where the run number is the last one
# Produce: A string run wi9tch only contain the number of the run
def extract_run (filename) :
    name = filename.replace(".txt", "")
    parts = name.split('_')
    run = parts[-1]
    return run.replace('run', '')

def extract_rmse(filepath):
    rmse_pos = None
    rmse_yaw = None
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("RMSE position:") or line.startswith("RMSE final:"):
                    rmse_pos = float(line.split(":")[1].strip())
                elif line.startswith("RMSE yaw"):
                    rmse_yaw = float(line.split(":")[1].strip())
    except Exception as e:
        print(f"Erro lendo {filepath}: {e}")
    return rmse_pos, rmse_yaw

def normalize_yaw(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def trajectory_length_xy(points):
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

def count_failure_events(pos_errors, yaw_abs):
    failure_mask = (pos_errors > FAILURE_POS_THRESHOLD) | (yaw_abs > FAILURE_YAW_THRESHOLD)
    if len(failure_mask) == 0:
        return 0

    events = int(failure_mask[0])
    if len(failure_mask) > 1:
        events += int(np.sum(failure_mask[1:] & ~failure_mask[:-1])) # Don't understand
    return events

def calculate_navigation_metrics(est, gt):
    if est.shape[0] != gt.shape[0]:
        min_len = min(est.shape[0], gt.shape[0])
        est = est[:min_len]
        gt = gt[:min_len]

    if est.size == 0 or gt.size == 0:
        return {}

    pos_errors = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
    yaw_abs = np.abs(normalize_yaw(est[:, 2] - gt[:, 2]))
    gt_path_m = trajectory_length_xy(gt[:, :2])
    est_path_m = trajectory_length_xy(est[:, :2])
    gt_path_km = gt_path_m / 1000.0
    failure_events = count_failure_events(pos_errors, yaw_abs)
    success = 1.0 if failure_events == 0 else 0.0

    recalls = {
        key: float(np.mean((pos_errors < pos_thr) & (yaw_abs < yaw_thr)))
        for key, (pos_thr, yaw_thr) in RECALL_THRESHOLDS.items()
    }

    return {
        "success": success,
        "spl": float(success * gt_path_m / max(est_path_m, gt_path_m, 1e-9)),
        "recall_t1": recalls["recall_t1"],
        "recall_t2": recalls["recall_t2"],
        "recall_t3": recalls["recall_t3"],
        "failure_rate": (
            failure_events / gt_path_km if gt_path_km > 0.0 else float("nan")
        ),
    }

def extract_neff(filepath):
    neff = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line[0].isdigit():
                    neff.append(int(float(line)))
    except Exception as e:
        print(f"Error opening {filepath} in extract_neff: {e}")
    return neff

# Action: Classify one position/yaw error sample according to the recall thresholds.
# I/ err_pos: Float position error in meters
# I/ err_yaw: Float yaw error in radians
# O/ String "T1", "T2", "T3" or None
# Produce:
#   - T1 if under threshold 1
#   - T2 if under threshold 2 and above/equal threshold 1
#   - T3 if under threshold 3 and above/equal threshold 2
#   - None if above threshold 3
def classify_recall(err_pos, err_yaw):
    error_pos = abs(float(err_pos))
    error_yaw = abs(float(err_yaw))

    if (
        error_pos < RECALL_THRESHOLDS["recall_t1"][0]
        and error_yaw < RECALL_THRESHOLDS["recall_t1"][1]
    ):
        return "T1"
    if (
        error_pos < RECALL_THRESHOLDS["recall_t2"][0]
        and error_yaw < RECALL_THRESHOLDS["recall_t2"][1]
    ):
        return "T2"
    if (
        error_pos < RECALL_THRESHOLDS["recall_t3"][0]
        and error_yaw < RECALL_THRESHOLDS["recall_t3"][1]
    ):
        return "T3"
    return None

# Action: Calculate the Recall Rate class at every step depending on the threshold.
# I/ filepath: String
# O/ RR: List of "T1", "T2", "T3" or None
# Necessity: A file where every raw-data line follows time,error_pos,error_yaw.
# Produce: one threshold class per valid line.
def Recall_Rate(filepath):
    rr = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue

                parts = line.split(",")
                if len(parts) < 3:
                    continue

                _, error_pos_str, error_yaw_str = parts[:3]
                rr.append(classify_recall(float(error_pos_str), float(error_yaw_str)))
    except Exception as e:
        print(f"Error opening {filepath} in Recall_Rate: {e}")
    return rr

# Action: Determine if a run is successful based on the final valid error sample.
# I/ filepath: String
# O/ success: bool or None
# Produce: True when the final valid sample is inside T1/T2/T3, False when it is
#          outside T3, and None when no valid raw error sample exists.
def Success(filepath):
    rr = Recall_Rate(filepath)
    if not rr:
        return None
    return rr[-1] in ("T1", "T2", "T3")


def plot_rmse(data, scenario, plot_path, test="pos", stat="mean",styles=None):
    styles = styles or {} # What is it doing

    plt.figure(figsize=(8, 6))
    path_type, measure = ( "Position", "(m)" ) if test == "pos" else ("Yaw", "(deg)")
    stat_type = "Mean" if stat == "mean" else "Std Dev"
     
    ylabel = f"{path_type} - {stat_type} {measure}"
    title = f"{path_type} RMSE {stat_type} vs Number of Particles - {scenario}"

    plt.title(title)
    plt.xlabel("Number of Particles")
    plt.ylabel(ylabel)

    for algo, results in data.items():

        particles = []
        stats = []
        for particle_count in sorted(results.keys()):
            value = results[particle_count].get(f"{test}_{stat}")
            if value is None or np.isnan(value):
                continue
            particles.append(particle_count)
            stats.append(value)

        if not particles:
            continue

        style = styles.get(algo, {'color': '#666666', 'linestyle': '-', 'marker': 'o', 'label': algo})

        plt.plot(
            particles,
            stats,
            label=style['label'],
            color=style['color'],
            linestyle=style['linestyle'],
            marker=style['marker'],
            linewidth=2
        )

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved at: {plot_path}")

def plot_sweep_metric(data, scenario, plot_path, metric, ylabel, title, styles=None, scale=1.0, ylim=None):
    styles = styles or {}
    plt.figure(figsize=(8, 6))
    plt.title(f"{title} vs Number of Particles - {scenario}")
    plt.xlabel("Number of Particles")
    plt.ylabel(ylabel)

    for algo, results in data.items():
        particles = []
        stats = []

        for particle_count in sorted(results.keys()):
            value = results[particle_count].get(f"{metric}_mean")
            if value is None or np.isnan(value):
                continue
            particles.append(particle_count)
            stats.append(value * scale)

        if not particles:
            continue

        style = styles.get(algo, {'color': '#666666', 'linestyle': '-', 'marker': 'o', 'label': algo})
        plt.plot(
            particles,
            stats,
            label=style['label'],
            color=style['color'],
            linestyle=style['linestyle'],
            marker=style['marker'],
            linewidth=2
        )

    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Plot saved at: {plot_path}")

def plot_recall_rates(data, scenario, plot_path, styles=None):
    styles = styles or {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    recall_specs = [
        ("recall_t1", "T1: <0.25 m, <2 deg"),
        ("recall_t2", "T2: <0.50 m, <5 deg"),
        ("recall_t3", "T3: <5.00 m, <10 deg"),
    ]

    for ax, (metric, title) in zip(axes, recall_specs):
        ax.set_title(title)
        ax.set_xlabel("Number of Particles")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.4)

        for algo, results in data.items():
            particles = []
            stats = []

            for particle_count in sorted(results.keys()):
                value = results[particle_count].get(f"{metric}_mean")
                if value is None or np.isnan(value):
                    continue
                particles.append(particle_count)
                stats.append(value * 100.0)

            if not particles:
                continue

            style = styles.get(algo, {'color': '#666666', 'linestyle': '-', 'marker': 'o', 'label': algo})
            ax.plot(
                particles,
                stats,
                label=style['label'],
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                linewidth=2
            )

    axes[0].set_ylabel("Recall Rate (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=max(1, min(len(labels), 6)))
        fig.subplots_adjust(bottom=0.22)
    fig.suptitle(f"Recall Rate vs Number of Particles - {scenario}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved at: {plot_path}")

# Action: Plot the quantile-quantile diagram for the best run of each algo
# I/ scenario: String
# I/ best_per_algo: Dic {Str algo:
#                           Tuple (
#                               Int particle,
#                               Int rmse, 
#                               Dic best_run {
#                                   List est [[x],[y],[yaw]],
#                                   List gt [[x],[y],[yaw]],
#                                   Float mh OR None
#                               }
#                           )
#                   }
# I/ plots_dir: path-like object
# O/ Nothing
# Necessity: A dictionnary best_per_algo matching the spec 
#           and plots_dir a valid path
# Produce: Per algo: a QQ plot per state variable dimension
#           (x,y,yaw for example) of the best run then store it with name
#           scenario_algo_qq_dimension.png
def plot_QQ (scenario, best_per_algo, plots_dir) :
    if not best_per_algo:
        return

    plt.figure(figsize=(8, 6))
    dict_intern = {0:"x",1:"y",2:"yaw"}

    
    for algo, (particles, rmse, best_run) in best_per_algo.items():
    
        est = best_run["est"]
        gt = best_run["gt"]
        mh = best_run.get("mh")

        x_gt, y_gt = gt[:, 0], gt[:, 1]
        x_est, y_est = est[:, 0], est[:, 1]
        yaw_gt = np.degrees(gt[:, 2])
        yaw_est = np.degrees(est[:, 2])

        i = 0
        for est_ax, gt_ax in zip((x_est, y_est, yaw_est), (x_gt, y_gt, yaw_gt)) :
            title = f"QQ plot of {dict_intern[i]} for {algo} {particles}p - {scenario}"

            plot_path = os.path.join(plots_dir, f"{scenario}_{algo}_qq_{dict_intern[i]}_{particles}p.png")
            
            sm.qqplot_2samples(gt_ax, est_ax,
                line="45"
            )

            plt.title(title)
            plt.xlabel("Ground true")
            plt.ylabel("Estimated")

            plt.grid(True, linestyle='--', alpha=0.4)
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"QQ plot saved at: {plot_path}")
            
            i += 1

# Action: Plot the effective sample size vs time diagram for the best run
#           of each algo
# I/ scenario: String
# I/ best_info: Dicionnary of witch nb_of_particule and run 
#               was the best for every algo
# I/ data_metrics: Dictionnary of the metrics collected for every run
# I/ plots_dir: path-like object to save the lot at right place
# I/ styles: Dictionnary that record the style to use for each algo
# O/ Nothing
# Necessity: A dictionnary data_metrics matching the spec in main(),
#           plots_dir a valid path,
#           scenario a valid senario
#           and best_info matching the output of unpack_best_per_algo
# Produce: A plot of the ESS for the best run per algo then store it with name
#           scenario_ess_best.png
def plot_ess(scenario, best_info, data_metrics, plots_dir, styles=None):
    if not best_info:
        print(f"No best-run information available for ESS plot: {scenario}")
        return

    styles = styles or {}
    plt.figure(figsize=(8, 6))

    title = f"Effective Sample Size vs time - {scenario}"

    plt.title(title)
    plt.xlabel("Time (iteration)")
    plt.ylabel("Effective Sample Size (number of particle)")

    plot_path = os.path.join(plots_dir, f"{scenario}_ess_best.png")
    plotted = False

    for algo, (particles, run) in best_info.items():
        ess = data_metrics[scenario][algo][particles][run].get("effective_sample_size", [])
        if not ess:
            print(f"Warning: No ESS data for {scenario} | {algo} | {particles}p | run {run}")
            continue

        style = styles.get(algo, {'color': '#666666', 'linestyle': '-', 'marker': 'o', 'label': algo})

        plt.plot(
            ess,
            label=style['label']+f" {particles}p",
            color=style['color'],
            linestyle=style['linestyle'],
            marker=style['marker'],
            linewidth=2
        )
        N = len(ess)

        plt.plot(
            N*[particles/2],
            label=style['label']+f" min threshold ({particles/2})",
            color=style['color'],
            linestyle='-',
            linewidth=0.5
        )
        plotted = True

    if not plotted:
        plt.close()
        print(f"No ESS plot generated for {scenario}: no ESS samples found.")
        return

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"ESS plot saved at: {plot_path}")

def calculate_yaw_rmse(est, gt):
    
    if est.shape[0] != gt.shape[0]:
        print("Warning: Estimation and ground truth have different lengths for yaw RMSE calculation.")
        min_len = min(est.shape[0], gt.shape[0])
        est = est[:min_len]
        gt = gt[:min_len]

    yaw_diff = np.arctan2(np.sin(est[:, 2] - gt[:, 2]), np.cos(est[:, 2] - gt[:, 2]))
    rmse_yaw = np.sqrt(np.mean(yaw_diff**2))
    #print(f"Calculated Yaw RMSE:{rmse_yaw:.2f} degrees")
    return rmse_yaw

def calculate_path_rmse(est, gt):
    if est.shape[0] != gt.shape[0]:
        print("Warning: Estimation and ground truth have different lengths for path RMSE calculation.")
        min_len = min(est.shape[0], gt.shape[0])
        est = est[:min_len]
        gt = gt[:min_len]

    error = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
    rmse_pos = np.sqrt(np.mean(error**2))
    #print(f"Calculated Path RMSE: {rmse_pos:.4f} m")
    return rmse_pos

def load_trajectory(filepath):
    est = []
    gt = []
    mh = []  # if MH rate is included in the file, we can also load it here for later analysis
    try:
        with open(filepath, 'r') as f:
            next(f)  # skip header

            for line in f:
                parts = line.strip().split(',')

                if len(parts) < 7:
                    continue

                est_x = float(parts[1])
                est_y = float(parts[2])
                est_yaw = float(parts[3])
                gt_x = float(parts[4]) + GT_X_OFFSET
                gt_y = float(parts[5])
                gt_yaw = float(parts[6])
                mh_rate = float(parts[7]) if len(parts) > 7 else 0.0

                est.append((est_x, est_y, est_yaw))
                gt.append((gt_x, gt_y, gt_yaw))
                mh.append(mh_rate)

    except Exception as e:
        print(f"Error reading trajectory {filepath}: {e}")

    return np.array(est), np.array(gt), mh

def unpack_best_per_algo(summary_path, trajectories, current_scenario):
    best_runs = {}
    best_info = {}
    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found.")
        return {}, {}

    with open(summary_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3: continue
            if parts[0].strip().lower() == "file":
                continue
            
            fname = parts[0].strip()
            path_rmse = float(parts[1])
            
            # Check if this line belongs to the scenario we are currently plotting
            if extract_scenario(fname) == current_scenario:
                algo = extract_algorithm(fname)
                parts_count = extract_particles(fname)
                
                # Check if this is the best run for this specific algorithm
                if algo not in best_runs or path_rmse < best_runs[algo][1]:
                    if fname in trajectories:
                        best_runs[algo] = (parts_count, path_rmse, trajectories[fname])
                        run = extract_run(fname)
                        best_info[algo] = (parts_count, run)
                    else:
                        print(f"Warning: Found {fname} in summary but no trajectory data loaded.")
    return best_runs, best_info

def plot_best_paths_all_algos(scenario, best_per_algo, best_path, ate_path, mh_rate_path, styles=None):
    styles = styles or {}

    plt.figure(figsize=(8, 6))

    plotted_gt = False

    for algo, (particles, rmse, best_run) in best_per_algo.items():

        if best_run is None:
            continue

        est = best_run["est"]
        gt = best_run["gt"]
        mh = best_run.get("mh")

        x_gt, y_gt = gt[:, 0], gt[:, 1]
        x_est, y_est = est[:, 0], est[:, 1]

        style = styles.get(algo, {'color': '#666666', 'linestyle': '-', 'label': algo})

        if not plotted_gt:

            start = np.array([x_gt[0], y_gt[0]])
            end = np.array([x_gt[-1], y_gt[-1]])
            plt.plot(
                x_gt, y_gt,
                linestyle='--',
                linewidth=2,
                color="#C00F0F",
                label='Ground Truth'
            )

            plt.scatter(start[0], start[1], color="#C00F0F", marker='o', s=100, label='Start')
            plt.scatter(end[0], end[1], color="#C00F0F", marker='X', s=100, label='End')


            plotted_gt = True

        plt.plot(
            x_est, y_est,
            linewidth=2,
            linestyle=style['linestyle'],
            color=style['color'],
            label=f'{algo} ({particles}p)'
        )

    plt.title(f"Best Paths per Algorithm - {scenario}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(best_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))

    plotted_gt = False
    best_yaw_path = best_path.replace(".png", "_yaw.png")

    for algo, (particles, rmse, best_run) in best_per_algo.items():

        if best_run is None:
            continue

        est = best_run["est"]
        gt = best_run["gt"]

        yaw_gt = np.degrees(gt[:, 2])
        yaw_est = np.degrees(est[:, 2])

        style = styles.get(algo, {'color': '#666666', 'linestyle': '-', 'label': algo})

        if not plotted_gt:
            plt.plot(
                yaw_gt,
                linestyle='--',
                linewidth=2,
                color="#C00F0F",
                label='Ground Truth'
            )


            plotted_gt = True

        plt.plot(
            yaw_est,
            linewidth=2,
            linestyle=style['linestyle'],
            color=style['color'],
            label=f'{algo} ({particles}p)'
        )

    plt.title(f"Best Yaw per Algorithm - {scenario}")
    plt.xlabel("Timestep")
    plt.ylabel("Yaw (deg)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(best_yaw_path, dpi=200)
    plt.close()

    # ---------------- ATE CURVE ----------------
    plt.figure(figsize=(8, 6))

    for algo, (particles, rmse, best_run) in best_per_algo.items():

        if best_run is None:
            continue

        est = best_run["est"]
        gt = best_run["gt"]

        error = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)

        style = styles.get(algo, {'color': '#666666', 'linestyle': '-'})

        plt.semilogy(
            error,
            label=f'{algo} ({particles}p)',
            linestyle=style['linestyle'],
            color=style['color']
        )

    plt.title(f"ATE Comparison - {scenario}")
    plt.xlabel("Timestep")
    plt.ylabel("Position ATE (m)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ate_path, dpi=200)
    plt.close()

    # ---------------- MH RATE CURVE ----------------
    plt.figure(figsize=(8, 6))

    for algo, (particles, rmse, best_run) in best_per_algo.items():

        if best_run is None:
            continue

        mh = best_run.get("mh", [])

        style = styles.get(algo, {'color': '#666666', 'linestyle': '-'})

        plt.plot(
            mh,
            label=f'{algo} ({particles}p)',
            linestyle=style['linestyle'],
            color=style['color']
        )

    plt.title(f"MH Rate Comparison - {scenario}")
    plt.xlabel("Timestep")
    plt.ylabel("MH Rate")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(mh_rate_path, dpi=200)
    plt.close()

    print(f"Combined best path plot saved: {best_path}")

def make_metric_bucket():
    return {metric: [] for metric in METRIC_KEYS}

def summarize_metric_bucket(bucket):
    summary = {}
    for metric in METRIC_KEYS:
        values = bucket[metric]
        summary[f"{metric}_mean"] = safe_mean(values)
        summary[f"{metric}_std"] = safe_std(values)
    return summary

def safe_mean(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)] # What is it doing
    if array.size == 0:
        return None
    return float(np.mean(array))

def safe_std(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return None
    return float(np.std(array))

def discover_result_dirs(results_root):
    result_dirs = []

    for current_dir, dirnames, filenames in os.walk(results_root):
        dirnames[:] = [d for d in dirnames if d != "plots"]
        has_pose_files = any(
            filename.startswith("poses_") and filename.endswith(".txt")
            for filename in filenames
        )
        has_sweep_files = any(
            filename.endswith(".txt") and extract_particles(filename) is not None
            for filename in filenames
        )
        if has_pose_files or has_sweep_files:
            result_dirs.append(current_dir)

    return sorted(set(result_dirs))

def process_results_dir(results_dir, results_root):
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(make_metric_bucket)))
    trajectories = {}
    relative_dir = os.path.relpath(results_dir, results_root)
    report_label = "root" if relative_dir == "." else relative_dir

    # Data structure:
    # data_metrics[scenario][algorithm][particles][run] = {
    #     "recall_rate": [...],
    #     "effective_sample_size": [...]
    # }
    # This keeps per-run diagnostics such as ESS/Neff while `data` stores
    # aggregated metrics used in plots and the HTML report.
    data_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "recall_rate": None,
        "effective_sample_size": [],
        "success": None
    }))))

    for filename in os.listdir(results_dir):
        if not filename.endswith(".txt"):
            continue

        if filename == "summary_results.txt":
            continue

        file_path = os.path.join(results_dir, filename)

        if filename.startswith("poses_"):
            algo = extract_algorithm(filename)
            particles = extract_particles(filename)
            scenario = extract_scenario(filename)
            run = extract_run(filename)

            if algo and particles:
                est, gt, mh = load_trajectory(file_path)
                if est.size > 0:
                    clean_path = filename.replace("poses_", "")
                    trajectories[clean_path] = {
                        "est": est,
                        "gt": gt,
                        "mh": mh
                    }

                    metrics = calculate_navigation_metrics(est, gt)
                    for metric, value in metrics.items():
                        if metric in data[scenario][algo][particles]:
                            data[scenario][algo][particles][metric].append(value)

                    if "success" in metrics:
                        data_metrics[scenario][algo][particles][run]["success"] = bool(metrics["success"])

                    print(f"Loaded trajectory: {filename} | {report_label}/{scenario} | {algo} | {particles}p")

        elif filename.startswith("neff_"):
            algo = extract_algorithm(filename)
            particles = extract_particles(filename)
            scenario = extract_scenario(filename)
            run = extract_run(filename)

            if algo and particles:
                data_metrics[scenario][algo][particles][run]["effective_sample_size"] = extract_neff(file_path)
                print(f"Loaded ESS: {filename} | {report_label}/{scenario} | {algo} | {particles}p | run {run}")

        else:
            algo = extract_algorithm(filename)
            particles = extract_particles(filename)
            scenario = extract_scenario(filename)
            run = extract_run(filename)

            if algo and particles:
                rmse_pos, rmse_yaw = extract_rmse(file_path)
                if (rmse_pos is not None) and (rmse_yaw is not None):
                    data[scenario][algo][particles]["pos"].append(rmse_pos)
                    data[scenario][algo][particles]["yaw"].append(np.degrees(rmse_yaw))
                    print(
                        f"{filename}: {report_label}/{scenario} | {algo} | {particles}p "
                        f"-> RMSE Position={rmse_pos:.4f}, RMSE Yaw={rmse_yaw:.4f}"
                    )

                data_metrics[scenario][algo][particles][run]["recall_rate"] = Recall_Rate(file_path)
                file_success = Success(file_path)
                if file_success is not None:
                    data_metrics[scenario][algo][particles][run]["success"] = file_success

    if not data:
        print(f"No valid data found in {results_dir}.")
        return

    styles = {
        'MCL': {'color': "#6C747461", 'linestyle': '-', 'marker': 'o', 'label': 'MCL'},
        'AMCL': {'color': '#1f77b4', 'linestyle': ':', 'marker': 'o', 'label': 'AMCL'},
        'MHMCL': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 'o', 'label': 'MHMCL'},
        'MHAMCL': {'color': '#2ca02c', 'linestyle': '-.', 'marker': 'o', 'label': 'MHAMCL'},
        'AMHMCL': {'color': "#b4801f", 'linestyle': '-', 'marker': 'o', 'label': 'AMHMCL'},
        'AMHAMCL': {'color': '#9467bd', 'linestyle': '--', 'marker': 'o', 'label': 'AMHAMCL'},
        '3MCL': {'color': '#17becf', 'linestyle': '-', 'marker': 's', 'label': '3MCL'}
    }   

    for scenario, scenario_data in data.items():

        avg_data = {}
        for algo, p_dict in scenario_data.items():
            avg_data[algo] = {
                p: summarize_metric_bucket(p_dict[p])
                for p in sorted(p_dict.keys())
            }
        
        # --- Plot everything for this scenario 
        pos_mean_plot_path = os.path.join(plots_dir, f"{scenario}_particle_sweep_rmse.png")
        plot_rmse(avg_data, scenario, pos_mean_plot_path, test="pos", stat="mean", styles=styles)

        pos_std_plot_path = os.path.join(plots_dir, f"{scenario}_particle_sweep_std.png")
        plot_rmse(avg_data, scenario, pos_std_plot_path, test="pos", stat="std", styles=styles)

        yaw_mean_plot_path = os.path.join(plots_dir, f"{scenario}_particle_sweep_rmse_yaw.png")
        plot_rmse(avg_data, scenario, yaw_mean_plot_path, test="yaw", stat="mean", styles=styles)

        yaw_std_plot_path = os.path.join(plots_dir, f"{scenario}_particle_sweep_std_yaw.png")
        plot_rmse(avg_data, scenario, yaw_std_plot_path, test="yaw", stat="std", styles=styles)

        success_plot_path = os.path.join(plots_dir, f"{scenario}_success_rate.png")
        plot_sweep_metric(
            avg_data,
            scenario,
            success_plot_path,
            "success",
            "Success Rate (%)",
            "Success Rate",
            styles=styles,
            scale=100.0,
            ylim=(0, 100)
        )

        spl_plot_path = os.path.join(plots_dir, f"{scenario}_spl.png")
        plot_sweep_metric(
            avg_data,
            scenario,
            spl_plot_path,
            "spl",
            "SPL",
            "Success Weighted by Path Length",
            styles=styles,
            ylim=(0, 1.05)
        )

        recall_plot_path = os.path.join(plots_dir, f"{scenario}_recall_rates.png")
        plot_recall_rates(avg_data, scenario, recall_plot_path, styles=styles)

        failure_plot_path = os.path.join(plots_dir, f"{scenario}_failure_rate.png")
        plot_sweep_metric(
            avg_data,
            scenario,
            failure_plot_path,
            "failure_rate",
            "Failure Rate (events/km)",
            "Failure Rate",
            styles=styles
        )

        # --- Find best (lowest RMSE position) ---
        summary_path = os.path.join(results_dir, "summary_results.txt")
        best_per_algo, best_info = unpack_best_per_algo(summary_path, trajectories, scenario)

        best_path = os.path.join(plots_dir, f"{scenario}_best_paths_all.png")
        ate_path = os.path.join(plots_dir, f"{scenario}_ate_all.png")
        mh_rate_path = os.path.join(plots_dir, f"{scenario}_mh_rate_all.png")

        plot_best_paths_all_algos(
            scenario,
            best_per_algo,
            best_path,
            ate_path,
            mh_rate_path,
            styles
        )

        plot_QQ (scenario, best_per_algo, plots_dir)

        plot_ess (scenario, best_info, data_metrics, plots_dir, styles)

        
    generate_html_report(data, plots_dir, True, report_label)

def main():
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
    if not os.path.exists(results_root):
        print("Results directory not found.")
        return

    result_dirs = discover_result_dirs(results_root)
    if not result_dirs:
        print("No valid data found.")
        return

    for results_dir in result_dirs:
        print(f"\nProcessing particle sweep plots in: {results_dir}")
        process_results_dir(results_dir, results_root)

def generate_html_report(all_data, results_dir, same_dir=False, report_label=None):

    html_path = os.path.join(results_dir, 'particle_sweep_report.html')
    report_suffix = f" - {report_label}" if report_label else ""

    html = """
    <html>
    <head>
    <title>Particle Sweep Report""" + report_suffix + """</title>
    <style>
    body {font-family: Arial; margin:40px;}
    h1 {color:#2c3e50;}
    h2 {margin-top:40px; color:#2980b9;}
    table {border-collapse: collapse; margin-top:15px;}
    th, td {border:1px solid #ccc; padding:6px 12px; text-align:center;}
    th {background:#f2f2f2;}
    .best {background:#c8f7c5; font-weight:bold;}
    img {max-width:800px; margin-top:20px;}
    .metric {font-size: 0.9em; color: #444;}
    </style>
    </head>
    <body>

    <h1>Particle Sweep Results""" + report_suffix + """</h1>
    """

    for scenario, scenario_data in all_data.items():

        html += f"<h2>Scenario: {scenario}</h2>"

        rmse_plot = f"{scenario}_particle_sweep_rmse.png"
        yaw_plot = f"{scenario}_particle_sweep_rmse_yaw.png"
        std_plot = f"{scenario}_particle_sweep_std.png"
        std_yaw_plot = f"{scenario}_particle_sweep_std_yaw.png"

        best_ess_plot = f"{scenario}_ess_best.png"
        best_path_plot = f"{scenario}_best_paths_all.png"
        ate_curve_plot = f"{scenario}_ate_all.png"
        best_path_yaw_plot = f"{scenario}_best_paths_all_yaw.png"
        mh_rate_plot = f"{scenario}_mh_rate_all.png"
        success_plot = f"{scenario}_success_rate.png"
        spl_plot = f"{scenario}_spl.png"
        recall_plot = f"{scenario}_recall_rates.png"
        failure_plot = f"{scenario}_failure_rate.png"
        prefix = "" if same_dir else "plots/"

        html += f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:20px;">
            <img src="{prefix}{ate_curve_plot}">
            <img src="{prefix}{rmse_plot}">
            <img src="{prefix}{std_plot}">
            <img src="{prefix}{best_path_yaw_plot}">
            <img src="{prefix}{yaw_plot}">
            <img src="{prefix}{std_yaw_plot}">
            <img src="{prefix}{success_plot}">
            <img src="{prefix}{spl_plot}">
            <img src="{prefix}{recall_plot}">
            <img src="{prefix}{failure_plot}">
            <img src="{prefix}{mh_rate_plot}">
            <img src="{prefix}{best_path_plot}">
            <img src="{prefix}{best_ess_plot}">
        </div>
        """

        # collect particle counts
        particles = sorted({
            p for algo in scenario_data
            for p in scenario_data[algo]
        })

        algorithms = sorted(scenario_data.keys())

        html += "<table>"
        html += "<tr><th>Particles</th>"

        for algo in algorithms:
            html += f"<th>{algo}</th>"

        html += "<th>Best</th></tr>"

        for p in particles:

            row_vals = {}

            for algo in algorithms:
                if p in scenario_data[algo]:
                    pos_vals = scenario_data[algo][p]["pos"]                    
                    pos_mean = safe_mean(pos_vals)
                    if pos_mean is not None:
                        row_vals[algo] = pos_mean

            best_algo = min(row_vals, key=row_vals.get) if row_vals else None

            html += f"<tr><td>{p}</td>"

            for algo in algorithms:

                if p in scenario_data[algo]:

                    pos_vals = scenario_data[algo][p]["pos"]
                    yaw_vals = scenario_data[algo][p]["yaw"]
                    pos_mean = safe_mean(pos_vals)
                    pos_std = safe_std(pos_vals)
                    yaw_mean = safe_mean(yaw_vals)
                    yaw_std = safe_std(yaw_vals)
                    sr = safe_mean(scenario_data[algo][p]["success"])
                    spl = safe_mean(scenario_data[algo][p]["spl"])
                    recall_t1 = safe_mean(scenario_data[algo][p]["recall_t1"])
                    recall_t2 = safe_mean(scenario_data[algo][p]["recall_t2"])
                    recall_t3 = safe_mean(scenario_data[algo][p]["recall_t3"])
                    failure_rate = safe_mean(scenario_data[algo][p]["failure_rate"])

                    if pos_mean is not None and pos_std is not None:

                        cls = "best" if algo == best_algo else ""

                        html += f'<td class="{cls}">'
                        html += f'{pos_mean:.3f} ± {pos_std:.3f} m<br>'
                        if yaw_mean is not None:
                            html += f'{yaw_mean:.2f} ± {yaw_std:.2f} °'
                        if sr is not None and spl is not None:
                            html += (
                                f'<br><span class="metric">SR {sr * 100:.1f}% | '
                                f'SPL {spl:.3f}</span>'
                            )
                        if recall_t1 is not None:
                            html += (
                                '<br><span class="metric">'
                                f'R {recall_t1 * 100:.1f}/'
                                f'{recall_t2 * 100:.1f}/'
                                f'{recall_t3 * 100:.1f}%'
                                '</span>'
                            )
                        if failure_rate is not None:
                            html += (
                                f'<br><span class="metric">F {failure_rate:.3f} ev/km</span>'
                            )
                        html += '</td>'

                    else:

                        html += "<td>-</td>"

                else:
                    html += "<td>-</td>"

            html += f"<td><b>{best_algo}</b></td></tr>"

        html += "</table>"

    html += "</body></html>"

    with open(html_path,"w") as f:
        f.write(html)

    print("HTML report:", html_path)

if __name__ == "__main__":
    main()
