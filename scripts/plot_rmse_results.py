#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def load_error_data(filepath):
    """Carrega dados temporais e RMSE final do arquivo"""
    times = []
    errors = []
    final_rmse = None
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('time,error'):
                    continue
                elif line.startswith('RMSE final:'):
                    final_rmse = float(line.split(':')[1].strip())
                elif ',' in line:
                    time, error = line.strip().split(',')
                    times.append(float(time))
                    errors.append(float(error))
                    
        return np.array(times), np.array(errors), final_rmse
    except Exception as e:
        print(f"Erro ao ler {filepath}: {str(e)}")
        return None, None, None

def main():
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    if not os.path.exists(results_dir):
        print(f"Pasta {results_dir} não encontrada.")
        return

    # Estrutura para armazenar todos os dados
    all_data = defaultdict(dict)
    
    # Processa cada arquivo
    for filename in os.listdir(results_dir):
        if filename.endswith('.txt') and '_' in filename:
            parts = filename.split('_')
            algorithm = parts[-1].replace('.txt', '')
            test_name = '_'.join(parts[:-1])
            
            filepath = os.path.join(results_dir, filename)
            times, errors, final_rmse = load_error_data(filepath)
            
            if times is not None and errors is not None:
                all_data[test_name][algorithm] = {
                    'times': times,
                    'errors': errors,
                    'rmse': final_rmse
                }
                print(f"Processado: {filename} | Pontos: {len(times)} | RMSE: {final_rmse:.4f}")

    if not all_data:
        print("Nenhum dado válido encontrado.")
        return

    # Cria diretório para os gráficos se não existir
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Gera gráficos para cada teste
    for test_name, algorithms in all_data.items():
        if len(algorithms) < 1:
            continue
            
        # Gráfico de evolução temporal
        plt.figure(figsize=(12, 6))
        colors = {'MCL': '#ff7f0e','AMCL': '#1f77b4','MHMCL': "#b4331f" , 'MHAMCL': '#2ca02c'}
        
        for algo, data in algorithms.items():
            if data['times'] is not None:
                plt.plot(data['times'], data['errors'], 
                        label=f'{algo} (RMSE: {data["rmse"]:.3f})',
                        color=colors.get(algo, '#666666'),
                        linewidth=2,
                        alpha=0.8)
        
        plt.title(f'Evolução do Erro - {test_name.replace("_", " ").title()}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Erro (m)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        time_plot_path = os.path.join(plots_dir, f'{test_name}_evolution.png')
        plt.savefig(time_plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Gráfico temporal salvo: {time_plot_path}")

        # Gráfico de barras comparativo (RMSE final)
        plt.figure(figsize=(8, 5))
        
        sorted_algs = sorted(algorithms.items(), 
                           key=lambda x: x[1]['rmse'] if x[1]['rmse'] is not None else float('inf'))
        
        for i, (algo, data) in enumerate(sorted_algs):
            if data['rmse'] is not None:
                plt.bar(i, data['rmse'], 
                       color=colors.get(algo, '#666666'),
                       label=algo,
                       width=0.6)
                plt.text(i, data['rmse']/2, 
                        f'{data["rmse"]:.3f}',
                        ha='center', va='center',
                        color='white',
                        fontweight='bold')
        
        plt.xticks(range(len(sorted_algs)), [x[0] for x in sorted_algs])
        plt.title(f'RMSE Final - {test_name.replace("_", " ").title()}')
        plt.ylabel('RMSE (m)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        bar_plot_path = os.path.join(plots_dir, f'{test_name}_rmse_comparison.png')
        plt.savefig(bar_plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Gráfico de RMSE salvo: {bar_plot_path}")

    # Gera tabela resumo HTML
    generate_html_summary(all_data, results_dir)

def generate_html_summary(data, output_dir):
    """Gera relatório HTML com todos os resultados"""
    html_content = """
    <html>
    <head>
        <title>Relatório de Desempenho</title>
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
        <h1>Relatório de Desempenho - Algoritmos de Localização</h1>
    """
    
    # Tabela resumo
    html_content += "<h2>Resumo Comparativo</h2><table>"
    html_content += "<tr><th>Teste</th><th>MCL</th><th>AMCL</th><th>MHMCL</th><th>MHAMCL</th></tr>"
    
    for test_name in sorted(data.keys()):
        html_content += f"<tr><td>{test_name.replace('_', ' ').title()}</td>"
        best_rmse = min([v['rmse'] for v in data[test_name].values() if v['rmse'] is not None], default=None)
        
        for algo in ['MCL', 'AMCL', 'MHMCL','MHAMCL']:
            if algo in data[test_name] and data[test_name][algo]['rmse'] is not None:
                rmse = data[test_name][algo]['rmse']
                cell_class = "best" if rmse == best_rmse else ""
                html_content += f'<td class="{cell_class}">{rmse:.4f}</td>'
            else:
                html_content += "<td>N/A</td>"
        html_content += "</tr>"
    
    html_content += "</table>"
    
    # Seção de gráficos
    html_content += "<h2>Gráficos Detalhados</h2>"
    for test_name in sorted(data.keys()):
        html_content += f"""
        <div class="plot-container">
            <div class="plot">
                <h3>{test_name.replace('_', ' ').title()} - Evolução</h3>
                <img src="plots/{test_name}_evolution.png" alt="Evolução do erro">
            </div>
            <div class="plot">
                <h3>{test_name.replace('_', ' ').title()} - RMSE Final</h3>
                <img src="plots/{test_name}_rmse_comparison.png" alt="Comparação RMSE">
            </div>
        </div>
        """
    
    html_content += "</body></html>"
    
    report_path = os.path.join(output_dir, 'performance_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nRelatório completo gerado: {report_path}")

if __name__ == '__main__':
    main()