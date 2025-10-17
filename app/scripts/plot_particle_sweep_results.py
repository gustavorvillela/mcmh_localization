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

    data = defaultdict(lambda: defaultdict(list))

    for filename in os.listdir(results_dir):
        if filename.endswith(".txt") and not filename.startswith("poses_"):
            algo = extract_algorithm(filename)
            particles = extract_particles(filename)
            if algo and particles:
                rmse = extract_rmse(os.path.join(results_dir, filename))
                if rmse is not None:
                    data[algo][particles].append(rmse)
                    print(f"{filename}: {algo} - {particles}p → RMSE={rmse:.4f}")

    if not data:
        print("Nenhum dado válido encontrado.")
        return

    avg_data = {}
    for algo, p_dict in data.items():
        avg_data[algo] = {
            p: (np.mean(rmses), np.std(rmses)) for p, rmses in sorted(p_dict.items())
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
        plt.errorbar(particles, means, yerr=stds,
                     label=style['label'],
                     color=style['color'],
                     linestyle=style['linestyle'],
                     marker=style['marker'],
                     linewidth=2,
                     capsize=4)

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nGráfico salvo em: {plot_path}")

    generate_html_report(avg_data, plot_path, results_dir)

def generate_html_report(avg_data, plot_path, results_dir):
    html_path = os.path.join(results_dir, 'particle_sweep_report.html')

    html_content = """
    <html>
    <head>
        <title>Relatório - Sweep de Partículas</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; }
            table { border-collapse: collapse; width: 60%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            img { max-width: 90%; border: 1px solid #ccc; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Relatório - RMSE vs Número de Partículas</h1>
        <p>Resultados médios e desvios padrão do sweep de partículas.</p>
        <img src="plots/particle_sweep_rmse.png" alt="Gráfico RMSE vs Número de Partículas">
        <h2>Tabela Resumo (Média ± Desvio Padrão)</h2>
        <table>
            <tr><th>Algoritmo</th><th>Número de Partículas</th><th>RMSE Médio (m)</th><th>Desvio Padrão</th></tr>
    """

    for algo, results in avg_data.items():
        for p, (mean, std) in sorted(results.items()):
            html_content += f"<tr><td>{algo}</td><td>{p}</td><td>{mean:.3f}</td><td>{std:.3f}</td></tr>"

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(html_path, 'w') as f:
        f.write(html_content)

    print(f"Relatório HTML gerado: {html_path}")

if __name__ == "__main__":
    main()
