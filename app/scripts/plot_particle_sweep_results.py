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

def extract_scenario(filename):
    name = filename.replace(".txt", "")

    # remove particle specification
    name = re.sub(r'_\d+p_', '_', name)

    # remove algorithm names
    for algo in ['MCL','AMCL','MHMCL','MHAMCL','AMHMCL','AMHAMCL']:
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
    plt.title(f"Pose RMSE vs Número de Partículas\n{scenario}")
    plt.xlabel("Número de Partículas")
    ylabel = "Position RMSE (m)" if test == "pos" else "Yaw RMSE (deg)"
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
    print(f"Gráfico salvo em: {plot_path}")

def main():

    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "pos": [],
        "yaw": []
    })))

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
                    data[scenario][algo][particles]["yaw"].append(rmse_yaw)
                    print(f"{filename}: {scenario} | {algo} | {particles}p → RMSE Position={rmse_pos:.4f}, RMSE Yaw={rmse_yaw:.4f}")

    if not data:
        print("Nenhum dado válido encontrado.")
        return

    styles = {
        'MCL': {'color': '#000000', 'linestyle': '-', 'marker': 'o', 'label': 'MCL'},
        'AMCL': {'color': '#1f77b4', 'linestyle': ':', 'marker': 'o', 'label': 'AMCL'},
        'MHMCL': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 'o', 'label': 'MHMCL'},
        'MHAMCL': {'color': '#2ca02c', 'linestyle': '-.', 'marker': 'o', 'label': 'MHAMCL'},
        'AMHMCL': {'color': '#b4331f', 'linestyle': '-', 'marker': 'o', 'label': 'AMHMCL'},
        'AMHAMCL': {'color': '#9467bd', 'linestyle': '--', 'marker': 'o', 'label': 'AMHAMCL'}
    }   

    for scenario, scenario_data in data.items():

        avg_data = {}
        for algo, p_dict in scenario_data.items():
            avg_data[algo] = {
                p: {
                    "pos_mean": np.mean(p_dict[p]["pos"]),
                    "pos_std": np.std(p_dict[p]["pos"]),
                    "yaw_mean": np.mean(np.degrees(p_dict[p]["yaw"])) if p_dict[p]["yaw"] else None,
                    "yaw_std": np.std(np.degrees(p_dict[p]["yaw"])) if p_dict[p]["yaw"] else None,
                }
                for p in sorted(p_dict.keys())
            }

        pos_mean_plot_path = os.path.join(plots_dir, f"{scenario}_particle_sweep_rmse.png")
        plot_rmse(avg_data, scenario, pos_mean_plot_path, test="pos", stat="mean", styles=styles)

        pos_std_plot_path = os.path.join(plots_dir, f"{scenario}_particle_sweep_std.png")
        plot_rmse(avg_data, scenario, pos_std_plot_path, test="pos", stat="std", styles=styles)

        yaw_mean_plot_path = os.path.join(plots_dir, f"{scenario}_particle_sweep_rmse_yaw.png")
        plot_rmse(avg_data, scenario, yaw_mean_plot_path, test="yaw", stat="mean", styles=styles)

        yaw_std_plot_path = os.path.join(plots_dir, f"{scenario}_particle_sweep_std_yaw.png")
        plot_rmse(avg_data, scenario, yaw_std_plot_path, test="yaw", stat="std", styles=styles)
        
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

        if not same_dir:
            plots_dir = "plots"

            html += f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                <img src="{plots_dir}/{rmse_plot}">
                <img src="{plots_dir}/{std_plot}">
                <img src="{plots_dir}/{yaw_plot}">
                <img src="{plots_dir}/{std_yaw_plot}">
            </div>
            """

        else:
            html += f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                <img src="{rmse_plot}">
                <img src="{std_plot}">
                <img src="{yaw_plot}">
                <img src="{std_yaw_plot}">
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
                        yaw_mean = np.mean(np.degrees(yaw_vals))
                        yaw_std = np.std(np.degrees(yaw_vals))
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
