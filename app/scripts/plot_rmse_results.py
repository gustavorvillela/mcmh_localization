#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

GT_X_OFFSET = float(os.environ.get("MCMH_GT_X_OFFSET", "0.7"))

def load_error_data(filepath):
    """Load temporal data and final RMSE from file"""
    times = []
    errors = []
    final_rmse = None
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('time,error'):
                    continue
                elif line.startswith('RMSE final:') or line.startswith('RMSE position:'):
                    final_rmse = float(line.split(':')[1].strip())
                elif ',' in line:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        times.append(float(parts[0]))
                        errors.append(float(parts[1]))
                    
        return np.array(times), np.array(errors), final_rmse
    except Exception as e:
        print(f"Error reading {filepath}: {str(e)}")
        return None, None, None

def load_trajectory_data(filepath):
    """Load trajectory data from poses_*.txt file"""
    data = {
        'time': [],
        'est_x': [],
        'est_y': [],
        'est_yaw': [],
        'gt_x': [],
        'gt_y': [],
        'gt_yaw': []
    }
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('time,est_x'):
                    continue
                elif ',' in line:
                    parts = line.strip().split(',')
                    if len(parts) >= 7:
                        data['time'].append(float(parts[0]))
                        data['est_x'].append(float(parts[1]))
                        data['est_y'].append(float(parts[2]))
                        data['est_yaw'].append(float(parts[3]))
                        data['gt_x'].append(float(parts[4]) + GT_X_OFFSET)
                        data['gt_y'].append(float(parts[5]))
                        data['gt_yaw'].append(float(parts[6]))
        
        # Convert to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
            
        return data
    except Exception as e:
        print(f"Error reading trajectory {filepath}: {str(e)}")
        return None

def discover_result_dirs(results_root):
    result_dirs = []

    for current_dir, dirnames, filenames in os.walk(results_root):
        dirnames[:] = [d for d in dirnames if d != "plots"]
        has_result_files = any(
            filename.endswith(".txt")
            and not filename.startswith("poses_")
            and filename != "summary_results.txt"
            and "p_run" not in filename
            for filename in filenames
        )
        if has_result_files:
            result_dirs.append(current_dir)

    return sorted(set(result_dirs))


def process_results_dir(results_dir, results_root):
    if not os.path.exists(results_dir):
        print(f"Folder {results_dir} not found.")
        return

    relative_dir = os.path.relpath(results_dir, results_root)
    report_label = "root" if relative_dir == "." else relative_dir

    # Estrutura para armazenar todos os dados
    all_data = defaultdict(dict)
    
    # Processa cada arquivo
    for filename in os.listdir(results_dir):
        if (
            filename.endswith('.txt')
            and not filename.startswith('poses_')
            and filename != 'summary_results.txt'
            and 'p_run' not in filename  # ← ignora arquivos do particle sweep
        ):
            parts = filename.replace('.txt','').split('_')
            if parts[-1].startswith("run"):
                run_id = parts[-1]
                algorithm = parts[-2]
                test_name = '_'.join(parts[:-2])
            else:
                run_id = None
                algorithm = parts[-1]
                test_name = '_'.join(parts[:-1])
            
            # Load error data
            filepath = os.path.join(results_dir, filename)
            times, errors, final_rmse = load_error_data(filepath)
            
            # Load trajectory data if available
            traj_filepath = os.path.join(results_dir, f'poses_{filename}')
            trajectory_data = None
            if os.path.exists(traj_filepath):
                trajectory_data = load_trajectory_data(traj_filepath)
            
            if times is not None and errors is not None:
                if algorithm not in all_data[test_name]:
                    all_data[test_name][algorithm] = {
                        'runs': [],
                        'rmses': []
                    }
                all_data[test_name][algorithm]['runs'].append({
                    'times': times,
                    'errors': errors,
                    'rmse': final_rmse,
                    'trajectory': trajectory_data
                })
                if final_rmse is not None:
                    all_data[test_name][algorithm]['rmses'].append(final_rmse)
                rmse_text = f"{final_rmse:.4f}" if final_rmse is not None else "N/A"
                print(f"Processed: {report_label}/{filename} | Points: {len(times)} | RMSE: {rmse_text}")
        elif 'p_run' in filename:
            print(f"Ignored (particle sweep): {filename}")

    if not all_data:
        print(f"No valid data found in {results_dir}.")
        return

    # Post-processing: compute mean, std and best run
    for test_name, algos in all_data.items():
        for algo, data in algos.items():
            if data['rmses']:
                data['mean_rmse'] = np.mean(data['rmses'])
                data['std_rmse'] = np.std(data['rmses'])
                best_idx = np.argmin(data['rmses'])
                data['best_run'] = data['runs'][best_idx]
            else:
                data['mean_rmse'] = None
                data['std_rmse'] = None
                data['best_run'] = None

    # Create directory for plots if it doesn't exist
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    colors = {'MCL': '#ff7f0e', 'AMCL': '#1f77b4', 'MHMCL': "#b4331f", 'MHAMCL': '#2ca02c', 'AMHMCL': "#4C2F67", 'AMHAMCL': '#8c564b'}

    # Generate plots for each test
    for test_name, algorithms in all_data.items():
        if len(algorithms) < 1:
            continue
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Plot 1: Error evolution (best run only)
        for algo, data in algorithms.items():
            best_run = data.get('best_run')
            if best_run and best_run['times'] is not None:
                ax1.plot(best_run['times'], best_run['errors'], 
                        label=f'{algo} (best RMSE: {best_run["rmse"]:.3f})',
                        color=colors.get(algo, '#666666'),
                        linewidth=2,
                        alpha=0.9)
        
        ax1.set_title(f'Error Evolution - {test_name.replace("_", " ").title()}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Error (m)')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Plot 2: Trajectory comparison (best run only)
        for algo, data in algorithms.items():
            best_run = data.get('best_run')
            if best_run and best_run['trajectory'] is not None:
                traj = best_run['trajectory']
                ax2.plot(traj['gt_x'], traj['gt_y'], 
                         color='#333333', linestyle='--', 
                         label='Ground Truth' if algo == list(algorithms.keys())[0] else '', 
                         linewidth=2)
                ax2.plot(traj['est_x'], traj['est_y'],
                         color=colors.get(algo, '#666666'),
                         label=f"{algo} (best RMSE {best_run['rmse']:.3f})",
                         linewidth=1.5,
                         alpha=0.9)
                
                # Plot start and end markers
                ax2.scatter(traj['gt_x'][0], traj['gt_y'][0], 
                            color='green', marker='o', s=50, 
                            label='Start' if algo == list(algorithms.keys())[0] else '')
                ax2.scatter(traj['gt_x'][-1], traj['gt_y'][-1], 
                            color='red', marker='x', s=50,
                            label='End' if algo == list(algorithms.keys())[0] else '')
        
        ax2.set_title(f'Trajectory Comparison - {test_name.replace("_", " ").title()}')
        ax2.set_xlabel('Position X (m)')
        ax2.set_ylabel('Position Y (m)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.axis('equal')
        
        # Save combined plot
        plot_path = os.path.join(plots_dir, f'{test_name}_combined.png')
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Combined plot saved: {plot_path}")

        # Comparative bar chart (mean ± std)
        plt.figure(figsize=(8, 5))
        
        sorted_algs = sorted(algorithms.items(), 
                           key=lambda x: x[1]['mean_rmse'] if x[1]['mean_rmse'] is not None else float('inf'))
        
        for i, (algo, data) in enumerate(sorted_algs):
            if data['mean_rmse'] is not None:
                plt.bar(i, data['mean_rmse'], 
                       yerr=data['std_rmse'],
                       capsize=5,
                       color=colors.get(algo, '#666666'),
                       label=algo,
                       width=0.6)
                plt.text(i, data['mean_rmse']/2, 
                        f'{data["mean_rmse"]:.1f}±{data["std_rmse"]:.1f}',
                        ha='center', va='center',
                        color='white',
                        fontweight='bold',
                        fontsize=8)

        plt.xticks(range(len(sorted_algs)), [x[0] for x in sorted_algs])
        plt.title(f'Final RMSE (mean ± std) - {test_name.replace("_", " ").title()}')
        plt.ylabel('RMSE (m)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        bar_plot_path = os.path.join(plots_dir, f'{test_name}_rmse_comparison.png')
        plt.savefig(bar_plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"RMSE plot saved: {bar_plot_path}")

    # Gera tabela resumo HTML
    generate_html_summary(all_data, results_dir)


def main():
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
    if not os.path.exists(results_root):
        print(f"Folder {results_root} not found.")
        return

    result_dirs = discover_result_dirs(results_root)
    if not result_dirs:
        print("No valid data found.")
        return

    for results_dir in result_dirs:
        print(f"\nProcessing RMSE plots in: {results_dir}")
        process_results_dir(results_dir, results_root)

def generate_html_summary(data, output_dir):
    """Generate HTML report with all results"""
    html_content = """
    <html>
    <head>
        <title>Performance Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .plot-container { display: flex; margin-bottom: 30px; }
            .plot { margin: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .plot img { max-width: 100%; height: auto; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .best { background-color: #e8f5e9; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Performance Report - Localization Algorithms</h1>
    """
    
    # Summary table
    html_content += "<h2>Comparative Summary (mean ± std)</h2><table>"
    html_content += "<tr><th>Test</th><th>MCL</th><th>AMCL</th><th>MHMCL</th><th>MHAMCL</th><th>AMHMCL</th><th>AMHAMCL</th></tr>"
    
    for test_name in sorted(data.keys()):
        html_content += f"<tr><td>{test_name.replace('_', ' ').title()}</td>"
        all_rmses = [v['mean_rmse'] for v in data[test_name].values() if v['mean_rmse'] is not None]
        best_rmse = min(all_rmses) if all_rmses else None

        for algo in ['MCL', 'AMCL', 'MHMCL','MHAMCL', 'AMHMCL', 'AMHAMCL']:
            if algo in data[test_name] and data[test_name][algo]['mean_rmse'] is not None:
                rmse = data[test_name][algo]['mean_rmse']
                std = data[test_name][algo]['std_rmse']
                cell_class = "best" if rmse == best_rmse else ""
                html_content += f'<td class="{cell_class}">{rmse:.4f} ± {std:.4f}</td>'
            else:
                html_content += "<td>N/A</td>"
        html_content += "</tr>"
    
    html_content += "</table>"
    
    # Plots section
    html_content += "<h2>Detailed Plots</h2>"
    for test_name in sorted(data.keys()):
        html_content += f"""
        <div class="plot-container">
            <div class="plot">
                <h3>{test_name.replace('_', ' ').title()} - Full Analysis</h3>
                <img src="plots/{test_name}_combined.png" alt="Full analysis">
            </div>
            <div class="plot">
                <h3>{test_name.replace('_', ' ').title()} - Final RMSE (mean ± std)</h3>
                <img src="plots/{test_name}_rmse_comparison.png" alt="RMSE comparison">
            </div>
        </div>
        """
    
    html_content += "</body></html>"
    
    report_path = os.path.join(output_dir, 'performance_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nFull report generated: {report_path}")

if __name__ == '__main__':
    main()
