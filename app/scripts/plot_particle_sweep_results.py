#!/usr/bin/env python3
"""
plot_particle_sweep_results.py
──────────────────────────────
Gera gráficos do particle sweep.

Nova estrutura esperada:
  results/
    {SCENARIO}/          ← C, M ou A  (ou diretório raiz se RESULTS_OVERRIDE aponta direto)
      {bag_name}/
        *.txt            ← arquivos de métricas
        poses_*.txt      ← trajetórias
        summary_results.txt
        plots/           ← gerado aqui

Pode ser invocado de duas formas:
  1. python3 plot_particle_sweep_results.py
     → varre results/C/, results/M/, results/A/ automaticamente

  2. RESULTS_OVERRIDE=/path/to/C/house python3 plot_particle_sweep_results.py
     → plota apenas aquele diretório específico
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ─── helpers de nome de arquivo ─────────────────────────────────────────────

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
    """Extrai o nome do cenário (bag) a partir do filename."""
    name = filename.replace(".txt", "").replace("poses_", "")
    name = re.sub(r'_\d+p_', '_', name)
    for algo in ['MCL', 'AMCL', 'MHMCL', 'MHAMCL', 'AMHMCL', 'AMHAMCL']:
        name = name.replace("_" + algo, "")
    name = re.sub(r'_run\d+', '', name)
    return name.strip("_")


# ─── leitura de arquivos ─────────────────────────────────────────────────────

def extract_rmse(filepath):
    """Lê RMSE de posição, yaw e métricas extras do arquivo de erros."""
    metrics = {
        "rmse_pos":     None,
        "rmse_yaw":     None,
        "recall_T1":    None,
        "recall_T2":    None,
        "recall_T3":    None,
        "failure_rate": None,
    }
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("RMSE position:") or line.startswith("RMSE final:"):
                    metrics["rmse_pos"] = float(line.split(":")[1])
                elif line.startswith("RMSE yaw"):
                    metrics["rmse_yaw"] = float(line.split(":")[1])
                elif line.startswith("Recall T1:"):
                    metrics["recall_T1"] = float(line.split(":")[1])
                elif line.startswith("Recall T2:"):
                    metrics["recall_T2"] = float(line.split(":")[1])
                elif line.startswith("Recall T3:"):
                    metrics["recall_T3"] = float(line.split(":")[1])
                elif line.startswith("Failure Rate"):
                    metrics["failure_rate"] = float(line.split(":")[1])
    except Exception as e:
        print(f"  [plot] Erro lendo {filepath}: {e}")
    return metrics

def load_trajectory(filepath):
    est, gt = [], []
    try:
        with open(filepath, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                est.append((float(parts[1]), float(parts[2]), float(parts[3])))
                gt.append( (float(parts[4]), float(parts[5]), float(parts[6])))
    except Exception as e:
        print(f"  [plot] Erro lendo trajetória {filepath}: {e}")
    return np.array(est), np.array(gt)

def unpack_best_per_algo(summary_path, trajectories, current_scenario):
    best_runs = {}
    if not os.path.exists(summary_path):
        return {}
    with open(summary_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            fname     = parts[0].strip()
            path_rmse = float(parts[1])
            if extract_scenario(fname) != current_scenario:
                continue
            algo        = extract_algorithm(fname)
            parts_count = extract_particles(fname)
            if algo and (algo not in best_runs or path_rmse < best_runs[algo][1]):
                if fname in trajectories:
                    best_runs[algo] = (parts_count, path_rmse, trajectories[fname])
    return best_runs


# ─── funções de plot ─────────────────────────────────────────────────────────

STYLES = {
    'MCL':     {'color': '#6C7474', 'linestyle': '-',  'marker': 'o', 'label': 'MCL'},
    'AMCL':    {'color': '#1f77b4', 'linestyle': ':',  'marker': 'o', 'label': 'AMCL'},
    'MHMCL':   {'color': '#ff7f0e', 'linestyle': '--', 'marker': 'o', 'label': 'MHMCL'},
    'MHAMCL':  {'color': '#2ca02c', 'linestyle': '-.', 'marker': 'o', 'label': 'MHAMCL'},
    'AMHMCL':  {'color': '#b4801f', 'linestyle': '-',  'marker': 'o', 'label': 'AMHMCL'},
    'AMHAMCL': {'color': '#9467bd', 'linestyle': '--', 'marker': 'o', 'label': 'AMHAMCL'},
}


def plot_metric_vs_particles(avg_data, scenario, plot_path,
                              metric_key, ylabel, title_suffix):
    plt.figure(figsize=(8, 6))
    plt.title(f"{title_suffix} vs Partículas — {scenario}")
    plt.xlabel("Número de Partículas")
    plt.ylabel(ylabel)

    for algo, results in avg_data.items():
        particles = sorted(results.keys())
        vals = [results[p].get(metric_key) for p in particles]
        if any(v is None for v in vals):
            continue
        style = STYLES.get(algo, {'color': '#666', 'linestyle': '-', 'marker': 'o', 'label': algo})
        plt.plot(particles, vals,
                 label=style['label'], color=style['color'],
                 linestyle=style['linestyle'], marker=style['marker'], linewidth=2)

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  Salvo: {plot_path}")


def plot_recall_vs_particles(avg_data, scenario, plots_dir):
    """Plota T1, T2, T3 em subplots lado a lado."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    labels = ["T1 (<0.25m, <2°)", "T2 (<0.5m, <5°)", "T3 (<5m, <10°)"]
    keys   = ["recall_T1_mean", "recall_T2_mean", "recall_T3_mean"]

    for ax, key, lbl in zip(axes, keys, labels):
        for algo, results in avg_data.items():
            particles = sorted(results.keys())
            vals = [results[p].get(key) for p in particles]
            if any(v is None for v in vals):
                continue
            style = STYLES.get(algo, {'color': '#666', 'linestyle': '-', 'marker': 'o', 'label': algo})
            ax.plot(particles, vals,
                    label=style['label'], color=style['color'],
                    linestyle=style['linestyle'], marker=style['marker'], linewidth=2)
        ax.set_title(f"Recall {lbl}")
        ax.set_xlabel("Partículas")
        ax.set_ylabel("Recall (proporção)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(f"Recall Rate vs Partículas — {scenario}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"{scenario}_recall_vs_particles.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Salvo: {path}")


def plot_best_paths_all_algos(scenario, best_per_algo, best_path, ate_path):
    # ── XY path ──────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    plotted_gt = False
    for algo, (particles, rmse, best_run) in best_per_algo.items():
        if best_run is None:
            continue
        est, gt = best_run["est"], best_run["gt"]
        style = STYLES.get(algo, {'color': '#666', 'linestyle': '-', 'label': algo})
        if not plotted_gt:
            plt.plot(gt[:, 0], gt[:, 1], '--', linewidth=2, color="#C00F0F", label='Ground Truth')
            plt.scatter(gt[0, 0],  gt[0, 1],  color="#C00F0F", marker='o', s=100, label='Start')
            plt.scatter(gt[-1, 0], gt[-1, 1], color="#C00F0F", marker='X', s=100, label='End')
            plotted_gt = True
        plt.plot(est[:, 0], est[:, 1], linewidth=2,
                 linestyle=style['linestyle'], color=style['color'],
                 label=f"{algo} ({particles}p)")
    plt.title(f"Best Paths per Algorithm — {scenario}")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.axis("equal"); plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig(best_path, dpi=200); plt.close()

    # ── Yaw ──────────────────────────────────────────────────────────────
    yaw_path = best_path.replace(".png", "_yaw.png")
    plt.figure(figsize=(8, 6))
    plotted_gt = False
    for algo, (particles, rmse, best_run) in best_per_algo.items():
        if best_run is None:
            continue
        est, gt = best_run["est"], best_run["gt"]
        style = STYLES.get(algo, {'color': '#666', 'linestyle': '-', 'label': algo})
        if not plotted_gt:
            plt.plot(np.degrees(gt[:, 2]), '--', linewidth=2, color="#C00F0F", label='Ground Truth')
            plotted_gt = True
        plt.plot(np.degrees(est[:, 2]), linewidth=2,
                 linestyle=style['linestyle'], color=style['color'],
                 label=f"{algo} ({particles}p)")
    plt.title(f"Best Yaw per Algorithm — {scenario}")
    plt.xlabel("Timestep"); plt.ylabel("Yaw (deg)")
    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig(yaw_path, dpi=200); plt.close()

    # ── ATE ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    for algo, (particles, rmse, best_run) in best_per_algo.items():
        if best_run is None:
            continue
        est, gt = best_run["est"], best_run["gt"]
        error = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
        style = STYLES.get(algo, {'color': '#666', 'linestyle': '-'})
        plt.semilogy(error, label=f"{algo} ({particles}p)",
                     linestyle=style['linestyle'], color=style['color'])
    plt.title(f"ATE Comparison — {scenario}")
    plt.xlabel("Timestep"); plt.ylabel("Position ATE (m)")
    plt.grid(True, linestyle='--', alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig(ate_path, dpi=200); plt.close()

    print(f"  Paths/ATE salvos: {best_path}")


# ─── lógica principal por diretório ─────────────────────────────────────────

def process_results_dir(results_dir, scenario_label=""):
    """
    Processa um único diretório de resultados (ex: results/C/house/).
    Gera todos os plots e retorna os dados agregados para o HTML.
    """
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # data[scenario][algo][particles] = {pos:[], yaw:[], T1:[], T2:[], T3:[], FR:[]}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "pos": [], "yaw": [],
        "T1": [], "T2": [], "T3": [],
        "failure_rate": []
    })))
    trajectories = {}

    for filename in os.listdir(results_dir):
        if not filename.endswith(".txt"):
            continue

        if filename.startswith("poses_"):
            algo      = extract_algorithm(filename)
            particles = extract_particles(filename)
            if algo and particles:
                est, gt = load_trajectory(os.path.join(results_dir, filename))
                if est.size > 0:
                    key = filename.replace("poses_", "")
                    trajectories[key] = {"est": est, "gt": gt}

        elif "summary" not in filename:
            algo      = extract_algorithm(filename)
            particles = extract_particles(filename)
            scenario  = extract_scenario(filename)
            if algo and particles:
                m = extract_rmse(os.path.join(results_dir, filename))
                if m["rmse_pos"] is not None and m["rmse_yaw"] is not None:
                    data[scenario][algo][particles]["pos"].append(m["rmse_pos"])
                    data[scenario][algo][particles]["yaw"].append(np.degrees(m["rmse_yaw"]))
                    if m["recall_T1"] is not None:
                        data[scenario][algo][particles]["T1"].append(m["recall_T1"])
                        data[scenario][algo][particles]["T2"].append(m["recall_T2"])
                        data[scenario][algo][particles]["T3"].append(m["recall_T3"])
                    if m["failure_rate"] is not None:
                        data[scenario][algo][particles]["failure_rate"].append(m["failure_rate"])

    if not data:
        print(f"  [plot] Sem dados em {results_dir}")
        return {}

    # ── Agregação ─────────────────────────────────────────────────────────
    for scenario, scenario_data in data.items():
        avg_data = {}
        for algo, p_dict in scenario_data.items():
            avg_data[algo] = {}
            for p in sorted(p_dict.keys()):
                d = p_dict[p]
                avg_data[algo][p] = {
                    "pos_mean":          np.mean(d["pos"]) if d["pos"] else None,
                    "pos_std":           np.std(d["pos"])  if d["pos"] else None,
                    "yaw_mean":          np.mean(d["yaw"]) if d["yaw"] else None,
                    "yaw_std":           np.std(d["yaw"])  if d["yaw"] else None,
                    "recall_T1_mean":    np.mean(d["T1"])  if d["T1"]  else None,
                    "recall_T2_mean":    np.mean(d["T2"])  if d["T2"]  else None,
                    "recall_T3_mean":    np.mean(d["T3"])  if d["T3"]  else None,
                    "failure_rate_mean": np.mean(d["failure_rate"]) if d["failure_rate"] else None,
                }

        tag = f"{scenario_label}_{scenario}" if scenario_label else scenario

        # RMSE plots
        plot_metric_vs_particles(avg_data, tag,
            os.path.join(plots_dir, f"{tag}_rmse_pos_mean.png"),
            "pos_mean", "Position RMSE (m)", "Position RMSE Mean")

        plot_metric_vs_particles(avg_data, tag,
            os.path.join(plots_dir, f"{tag}_rmse_pos_std.png"),
            "pos_std", "Position RMSE Std (m)", "Position RMSE Std Dev")

        plot_metric_vs_particles(avg_data, tag,
            os.path.join(plots_dir, f"{tag}_rmse_yaw_mean.png"),
            "yaw_mean", "Yaw RMSE (deg)", "Yaw RMSE Mean")

        plot_metric_vs_particles(avg_data, tag,
            os.path.join(plots_dir, f"{tag}_rmse_yaw_std.png"),
            "yaw_std", "Yaw RMSE Std (deg)", "Yaw RMSE Std Dev")

        # Recall T1/T2/T3
        plot_recall_vs_particles(avg_data, tag, plots_dir)

        # Failure Rate
        plot_metric_vs_particles(avg_data, tag,
            os.path.join(plots_dir, f"{tag}_failure_rate.png"),
            "failure_rate_mean", "Failure Rate (falhas/km)", "Failure Rate")

        # Best paths / ATE
        summary_path = os.path.join(results_dir, "summary_results.txt")
        best_per_algo = unpack_best_per_algo(summary_path, trajectories, scenario)
        if best_per_algo:
            plot_best_paths_all_algos(
                tag,
                best_per_algo,
                os.path.join(plots_dir, f"{tag}_best_paths_all.png"),
                os.path.join(plots_dir, f"{tag}_ate_all.png"),
            )

    return data


# ─── HTML report ─────────────────────────────────────────────────────────────

def generate_html_report(all_scenario_data, output_dir):
    """
    all_scenario_data: dict[scenario_label] → data dict retornado por process_results_dir
    """
    html_path = os.path.join(output_dir, 'particle_sweep_report.html')

    html = """<html><head><title>Particle Sweep Report</title>
    <style>
    body {font-family: Arial; margin:40px;}
    h1 {color:#2c3e50;} h2 {margin-top:40px;color:#2980b9;} h3{color:#555;}
    table{border-collapse:collapse;margin-top:15px;}
    th,td{border:1px solid #ccc;padding:6px 12px;text-align:center;}
    th{background:#f2f2f2;} .best{background:#c8f7c5;font-weight:bold;}
    img{max-width:700px;margin-top:10px;}
    .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}
    </style></head><body>
    <h1>Particle Sweep Results</h1>
    """

    for scenario_label, data in all_scenario_data.items():
        html += f"<h2>Cenário: {scenario_label}</h2>"

        for scenario, scenario_data in data.items():
            tag = f"{scenario_label}_{scenario}" if scenario_label else scenario
            html += f"<h3>{scenario}</h3><div class='grid'>"

            for imgname in [
                f"{tag}_rmse_pos_mean.png",
                f"{tag}_rmse_pos_std.png",
                f"{tag}_recall_vs_particles.png",
                f"{tag}_rmse_yaw_mean.png",
                f"{tag}_rmse_yaw_std.png",
                f"{tag}_failure_rate.png",
                f"{tag}_best_paths_all.png",
                f"{tag}_best_paths_all_yaw.png",
                f"{tag}_ate_all.png",
            ]:
                html += f"<img src='plots/{imgname}' onerror=\"this.style.display='none'\">"

            html += "</div>"

            # Tabela de métricas
            particles = sorted({p for algo in scenario_data for p in scenario_data[algo]})
            algorithms = sorted(scenario_data.keys())

            html += "<table><tr><th>Particles</th>"
            for algo in algorithms:
                html += f"<th>{algo}</th>"
            html += "<th>Best RMSE</th></tr>"

            for p in particles:
                row_vals = {}
                for algo in algorithms:
                    if p in scenario_data[algo]:
                        vals = scenario_data[algo][p]["pos"]
                        if vals:
                            row_vals[algo] = np.mean(vals)
                best_algo = min(row_vals, key=row_vals.get) if row_vals else None

                html += f"<tr><td>{p}</td>"
                for algo in algorithms:
                    if p in scenario_data[algo] and scenario_data[algo][p]["pos"]:
                        d = scenario_data[algo][p]
                        pm = np.mean(d["pos"]); ps = np.std(d["pos"])
                        ym = np.mean(d["yaw"]) if d["yaw"] else None
                        t1 = np.mean(d["T1"]) if d["T1"] else None
                        t2 = np.mean(d["T2"]) if d["T2"] else None
                        t3 = np.mean(d["T3"]) if d["T3"] else None
                        fr = np.mean(d["failure_rate"]) if d["failure_rate"] else None
                        cls = "best" if algo == best_algo else ""
                        html += f'<td class="{cls}">'
                        html += f'RMSE: {pm:.3f}±{ps:.3f}m<br>'
                        if ym is not None:
                            html += f'Yaw: {ym:.2f}°<br>'
                        if t1 is not None:
                            html += f'T1:{t1:.2f} T2:{t2:.2f} T3:{t3:.2f}<br>'
                        if fr is not None:
                            html += f'FR:{fr:.2f}f/km'
                        html += '</td>'
                    else:
                        html += "<td>—</td>"
                html += f"<td><b>{best_algo}</b></td></tr>"

            html += "</table>"

    html += "</body></html>"

    with open(html_path, "w") as f:
        f.write(html)
    print(f"HTML report: {html_path}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    override = os.environ.get("RESULTS_OVERRIDE")

    if override:
        # Invocado pelo shell script para um diretório específico
        label = os.path.basename(override)
        data  = process_results_dir(override, scenario_label=label)
        if data:
            plots_dir = os.path.join(override, "plots")
            generate_html_report({label: data}, plots_dir)
        return

    # Invocado manualmente: varre C/, M/, A/
    base_results = os.path.join(os.path.dirname(__file__), '../results')
    all_data = {}

    for scenario_folder in sorted(os.listdir(base_results)):
        scenario_path = os.path.join(base_results, scenario_folder)
        if not os.path.isdir(scenario_path):
            continue
        if scenario_folder not in ("C", "M", "A"):
            continue

        for bag_folder in sorted(os.listdir(scenario_path)):
            bag_path = os.path.join(scenario_path, bag_folder)
            if not os.path.isdir(bag_path):
                continue
            label = f"{scenario_folder}/{bag_folder}"
            print(f"\n=== Processando {label} ===")
            data = process_results_dir(bag_path, scenario_label=f"{scenario_folder}_{bag_folder}")
            if data:
                all_data[label] = data

    if all_data:
        # Report consolidado na raiz de results/
        generate_html_report(all_data, base_results)
    else:
        print("Nenhum dado encontrado.")


if __name__ == "__main__":
    main()
