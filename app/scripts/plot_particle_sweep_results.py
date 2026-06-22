#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import statsmodels.api as sm

list_algos = ['MCL', 'AMCL', 'MHMCL', 'MHAMCL', 'AMHMCL', 'AMHAMCL', '3MCL']

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

    # remove particle specification
    name = re.sub(r'_\d+p_', '_', name)

    # remove algorithm names
    for algo in list_algos:
        name = name.replace("_" + algo, "")

    # remove run index if present
    name = re.sub(r'_run\d+', '', name)

    return name.strip("_")


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

def plot_rmse(data, scenario, plot_path, test="pos", stat="mean",styles=None):

    plt.figure(figsize=(8, 6))
    path_type, measure = ( "Position", "(m)" ) if test == "pos" else ("Yaw", "(deg)")
    stat_type = "Mean" if stat == "mean" else "Std Dev"
     
    ylabel = f"{path_type} - {stat_type} {measure}"
    title = f"{path_type} RMSE {stat_type} vs Number of Particles - {scenario}"

    plt.title(title)
    plt.xlabel("Number of Particles")
    plt.ylabel(ylabel)

    for algo, results in data.items():

        particles = sorted(results.keys())
        stats = [results[p][f"{test}_{stat}"] for p in particles]
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
                gt_x = float(parts[4]) + 0.7
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
    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found.")
        return {}

    with open(summary_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3: continue
            
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
                    else:
                        print(f"Warning: Found {fname} in summary but no trajectory data loaded.")
    return best_runs

def plot_best_paths_all_algos(scenario, best_per_algo, best_path, ate_path, mh_rate_path, styles=None):

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

def main():

    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "pos": [],
        "yaw": []
    })))

    trajectories = {}

    # Build data structure: data[scenario][algorithm][particles] = {"pos": [...], "yaw": [...]}
    for filename in os.listdir(results_dir):
        if filename.endswith(".txt") and not filename.startswith("poses_"):
            algo = extract_algorithm(filename)
            particles = extract_particles(filename)
            scenario = extract_scenario(filename)
            if algo and particles:
                rmse_pos, rmse_yaw = extract_rmse(os.path.join(results_dir, filename))
                if (rmse_pos is not None) and (rmse_yaw is not None):
                    data[scenario][algo][particles]["pos"].append(rmse_pos)
                    data[scenario][algo][particles]["yaw"].append(np.degrees(rmse_yaw))
                    print(f"{filename}: {scenario} | {algo} | {particles}p → RMSE Position={rmse_pos:.4f}, RMSE Yaw={rmse_yaw:.4f}")

        elif filename.endswith(".txt") and filename.startswith("poses_"):
            algo = extract_algorithm(filename)
            particles = extract_particles(filename)
            scenario = extract_scenario(filename)

            if algo and particles:
                path = os.path.join(results_dir, filename)
                est, gt, mh = load_trajectory(path)
                if est.size > 0:
                    clean_path = filename.replace("poses_", "")
                    trajectories[clean_path] = {
                            "est": est,
                            "gt": gt,
                            "mh": mh
                    }
                    print(f"Loaded trajectory: {filename} | {scenario} | {algo} | {particles}p")

    if not data:
        print("No valid data found.")
        return

    styles = {
        'MCL': {'color': "#6C747461", 'linestyle': '-', 'marker': 'o', 'label': 'MCL'},
        'AMCL': {'color': '#1f77b4', 'linestyle': ':', 'marker': 'o', 'label': 'AMCL'},
        'MHMCL': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 'o', 'label': 'MHMCL'},
        'MHAMCL': {'color': '#2ca02c', 'linestyle': '-.', 'marker': 'o', 'label': 'MHAMCL'},
        'AMHMCL': {'color': "#b4801f", 'linestyle': '-', 'marker': 'o', 'label': 'AMHMCL'},
        'AMHAMCL': {'color': '#9467bd', 'linestyle': '--', 'marker': 'o', 'label': 'AMHAMCL'}
    }   

    for scenario, scenario_data in data.items():

        avg_data = {}
        for algo, p_dict in scenario_data.items():
            avg_data[algo] = {
                p: {
                    "pos_mean": np.mean(p_dict[p]["pos"]),
                    "pos_std": np.std(p_dict[p]["pos"]),
                    "yaw_mean": np.mean(p_dict[p]["yaw"]) if p_dict[p]["yaw"] else None,
                    "yaw_std": np.std(p_dict[p]["yaw"]) if p_dict[p]["yaw"] else None,
                }
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

        # --- Find best (lowest RMSE position) ---
        summary_path = os.path.join(results_dir, "summary_results.txt")
        best_per_algo = unpack_best_per_algo(summary_path, trajectories, scenario)

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

        # --- Plot quantile-quantile for best run only
        plot_QQ (scenario, best_per_algo, plots_dir)

        
    generate_html_report(data, plots_dir, True)

def generate_html_report(all_data, results_dir, same_dir=False):

    html_path = os.path.join(results_dir, 'particle_sweep_report.html')

    html = """
    <html>
    <head>
    <title>Particle Sweep Report</title>
    <style>
    body {font-family: Arial; margin:40px;}
    h1 {color:#2c3e50;}
    h2 {margin-top:40px; color:#2980b9;}
    table {border-collapse: collapse; margin-top:15px;}
    th, td {border:1px solid #ccc; padding:6px 12px; text-align:center;}
    th {background:#f2f2f2;}
    .best {background:#c8f7c5; font-weight:bold;}
    img {max-width:800px; margin-top:20px;}
    </style>
    </head>
    <body>

    <h1>Particle Sweep Results</h1>
    """

    for scenario, scenario_data in all_data.items():

        html += f"<h2>Scenario: {scenario}</h2>"

        rmse_plot = f"{scenario}_particle_sweep_rmse.png"
        yaw_plot = f"{scenario}_particle_sweep_rmse_yaw.png"
        std_plot = f"{scenario}_particle_sweep_std.png"
        std_yaw_plot = f"{scenario}_particle_sweep_std_yaw.png"

        best_path_plot = f"{scenario}_best_paths_all.png"
        ate_curve_plot = f"{scenario}_ate_all.png"
        best_path_yaw_plot = f"{scenario}_best_paths_all_yaw.png"
        mh_rate_plot = f"{scenario}_mh_rate_all.png"

        if not same_dir:
            plots_dir = "plots"

            html += f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:20px;">
                <img src="{plots_dir}/{ate_curve_plot}">
                <img src="{plots_dir}/{rmse_plot}">
                <img src="{plots_dir}/{std_plot}">
                <img src="{plots_dir}/{best_path_yaw_plot}">
                <img src="{plots_dir}/{yaw_plot}">
                <img src="{plots_dir}/{std_yaw_plot}">
                <img src="{plots_dir}/{mh_rate_plot}">
                <img src="{plots_dir}/{best_path_plot}">
            </div>
            """

        else:
            html += f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:20px;">
                <img src="{ate_curve_plot}">
                <img src="{rmse_plot}">
                <img src="{std_plot}">
                <img src="{best_path_yaw_plot}">
                <img src="{yaw_plot}">
                <img src="{std_yaw_plot}">
                <img src="{mh_rate_plot}">
                <img src="{best_path_plot}">
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
                    if pos_vals:
                        row_vals[algo] = np.mean(pos_vals)

            best_algo = min(row_vals, key=row_vals.get) if row_vals else None

            html += f"<tr><td>{p}</td>"

            for algo in algorithms:

                if p in scenario_data[algo]:

                    pos_vals = scenario_data[algo][p]["pos"]
                    yaw_vals = scenario_data[algo][p]["yaw"]
                    if pos_vals:
                        pos_mean = np.mean(pos_vals)
                        pos_std = np.std(pos_vals)
                    else:
                        pos_mean = None
                        pos_std = None

                    if yaw_vals:
                        yaw_mean = np.mean(yaw_vals)
                        yaw_std = np.std(yaw_vals)
                    else:
                        yaw_mean = None
                        yaw_std = None

                    if pos_mean is not None and pos_std is not None:

                        cls = "best" if algo == best_algo else ""

                        html += f'<td class="{cls}">'
                        html += f'{pos_mean:.3f} ± {pos_std:.3f} m<br>'
                        if yaw_mean is not None:
                            html += f'{yaw_mean:.2f} ± {yaw_std:.2f} °'
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
