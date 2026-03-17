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

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for filename in os.listdir(results_dir):
        if filename.endswith(".txt") and not filename.startswith("poses_"):
            algo = extract_algorithm(filename)
            particles = extract_particles(filename)
            scenario = extract_scenario(filename)
            if algo and particles:
                rmse = extract_rmse(os.path.join(results_dir, filename))
                if rmse is not None:
                    data[scenario][algo][particles].append(rmse)
                    print(f"{filename}: {scenario} | {algo} | {particles}p → RMSE={rmse:.4f}")

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
                p: (np.mean(rmses), np.std(rmses))
                for p, rmses in sorted(p_dict.items())
            }

        plot_path = os.path.join(plots_dir, f"{scenario}_particle_sweep_rmse.png")

        plt.figure(figsize=(8, 6))
        plt.title(f"RMSE vs Número de Partículas\n{scenario}")
        plt.xlabel("Número de Partículas")
        plt.ylabel("RMSE (m)")

        for algo, results in avg_data.items():

            particles = sorted(results.keys())
            means = [results[p][0] for p in particles]
            stds = [results[p][1] for p in particles]

            style = styles.get(algo, {'color': '#666666', 'linestyle': '-', 'marker': 'o', 'label': algo})

            plt.plot(
                particles,
                means,
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

        std_plot_path = os.path.join(plots_dir, f"{scenario}_particle_sweep_std.png")

        plt.figure(figsize=(8,6))
        plt.title(f"Std Dev vs Número de Partículas\n{scenario}")
        plt.xlabel("Número de Partículas")
        plt.ylabel("RMSE Std Dev (m)")

        for algo, p_dict in scenario_data.items():

            particles = sorted(p_dict.keys())

            stds = [np.std(p_dict[p]) for p in particles]

            style = styles.get(algo, {'color':'#666','linestyle':'-','marker':'o','label':algo})

            plt.plot(
                particles,
                stds,
                label=style['label'],
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                linewidth=2
            )

        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(std_plot_path, dpi=200)
        plt.close()

    generate_html_report(data, plot_path, results_dir)

def generate_html_report(all_data, plots_dir, results_dir):

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

        plot_file = f"{scenario}_particle_sweep_rmse.png"
        rmse_plot = f"{scenario}_particle_sweep_rmse.png"
        std_plot = f"{scenario}_particle_sweep_std.png"

        html += f'<img src="plots/{rmse_plot}"><br>'
        html += f'<img src="plots/{std_plot}"><br>'

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
                    rmses = scenario_data[algo][p]
                    row_vals[algo] = np.mean(rmses)

            best_algo = min(row_vals, key=row_vals.get)

            html += f"<tr><td>{p}</td>"

            for algo in algorithms:

                if p in scenario_data[algo]:

                    rmses = scenario_data[algo][p]
                    mean = np.mean(rmses)
                    std = np.std(rmses)

                    cls = "best" if algo == best_algo else ""

                    html += f'<td class="{cls}">{mean:.3f} ± {std:.3f}</td>'

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
