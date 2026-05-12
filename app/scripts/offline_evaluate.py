#!/usr/bin/env python3
"""
offline_evaluate.py
-------------------
Reconstrói métricas completas a partir dos arquivos poses_*.txt:
  - RMSE posição e yaw
  - Recall rate T1/T2/T3 (como em Sattler et al.)
  - Failure Rate (falhas de localização por km percorrido)
  - Gera summary_results.txt com todas as métricas
"""

import os
import numpy as np


# ─── Thresholds Recall (Sattler et al.) ─────────────────────────────────────
RECALL_THRESHOLDS = [
    (0.25, np.radians(2.0),  "T1"),   # <0.25m, <2°
    (0.50, np.radians(5.0),  "T2"),   # <0.50m, <5°
    (5.00, np.radians(10.0), "T3"),   # <5.00m, <10°
]

# ─── Parâmetros de Failure Rate ──────────────────────────────────────────────
FAILURE_THRESHOLD = 1.0   # metros
FAILURE_WINDOW    = 10    # timesteps consecutivos


def rebuild_error_file_from_pose(pose_path, results_dir):
    """
    Lê poses_*.txt e gera:
      - arquivo de erros por timestep  (nome sem 'poses_')
      - métricas escalares: rmse_pos, rmse_yaw, recall T1/T2/T3, failure_rate
    Retorna dict com todas as métricas ou None em caso de erro.
    """
    base      = os.path.basename(pose_path).replace("poses_", "")
    error_path = os.path.join(results_dir, base)

    pos_errors  = []
    yaw_errors  = []
    yaw_raws    = []   # yaw estimado e gt para recall
    timestamps  = []

    gt_positions = []  # para calcular distância percorrida

    # ── Leitura ──────────────────────────────────────────────────────────
    try:
        with open(pose_path) as f:
            header = next(f).strip().split(",")
            # Suporta poses com e sem coluna pos_error (compatibilidade)
            has_error_col = len(header) >= 8

            for line in f:
                vals = line.strip().split(",")
                if len(vals) < 7:
                    continue
                t      = float(vals[0])
                est_x  = float(vals[1]);  est_y  = float(vals[2]);  est_yaw = float(vals[3])
                gt_x   = float(vals[4]);  gt_y   = float(vals[5]);  gt_yaw  = float(vals[6])

                pos_err  = np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2)
                yaw_diff = np.arctan2(
                    np.sin(est_yaw - gt_yaw),
                    np.cos(est_yaw - gt_yaw)
                )

                pos_errors.append(pos_err)
                yaw_errors.append(yaw_diff)
                yaw_raws.append((abs(yaw_diff), abs(yaw_diff)))  # (est_err, gt_err) já é diff
                timestamps.append(t)
                gt_positions.append((gt_x, gt_y))

    except Exception as e:
        print(f"[offline_evaluate] Erro lendo {pose_path}: {e}")
        return None

    if not pos_errors:
        print(f"[offline_evaluate] Sem dados válidos: {pose_path}")
        return None

    pos_errors = np.array(pos_errors)
    yaw_errors = np.array(yaw_errors)

    # ── RMSE ─────────────────────────────────────────────────────────────
    rmse_pos = float(np.sqrt(np.mean(pos_errors**2)))
    rmse_yaw = float(np.sqrt(np.mean(yaw_errors**2)))

    # ── Recall T1/T2/T3 ──────────────────────────────────────────────────
    # Recall = % de poses com pos_err < pos_thresh E |yaw_err| < yaw_thresh
    recall = {}
    abs_yaw = np.abs(yaw_errors)
    for (pos_thr, yaw_thr, label) in RECALL_THRESHOLDS:
        mask = (pos_errors < pos_thr) & (abs_yaw < yaw_thr)
        recall[label] = float(np.mean(mask))   # proporção 0–1

    # ── Failure Rate ──────────────────────────────────────────────────────
    # Janela deslizante: FAILURE_WINDOW timesteps consecutivos > FAILURE_THRESHOLD
    consecutive = 0
    in_failure  = False
    n_failures  = 0
    failure_times = []

    for i, err in enumerate(pos_errors):
        if err > FAILURE_THRESHOLD:
            consecutive += 1
        else:
            consecutive = 0
            in_failure  = False

        if consecutive >= FAILURE_WINDOW and not in_failure:
            in_failure = True
            n_failures += 1
            failure_times.append(timestamps[i])

    # Distância total percorrida (GT odometry)
    gt_arr = np.array(gt_positions)
    if len(gt_arr) > 1:
        diffs = np.linalg.norm(np.diff(gt_arr, axis=0), axis=1)
        total_km = float(np.sum(diffs)) / 1000.0
    else:
        total_km = 0.0

    failure_rate = n_failures / total_km if total_km > 0 else float('nan')

    # ── Salva arquivo de erros por timestep ──────────────────────────────
    with open(error_path, "w") as f:
        f.write("time,error_pos,error_yaw\n")

        with open(pose_path) as pf:
            next(pf)
            for line in pf:
                vals = line.strip().split(",")
                if len(vals) < 7:
                    continue
                t      = float(vals[0])
                est_x  = float(vals[1]);  est_y  = float(vals[2]);  est_yaw = float(vals[3])
                gt_x   = float(vals[4]);  gt_y   = float(vals[5]);  gt_yaw  = float(vals[6])

                pos_err  = np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2)
                yaw_diff = np.arctan2(
                    np.sin(est_yaw - gt_yaw),
                    np.cos(est_yaw - gt_yaw)
                )
                f.write(f"{t:.3f},{pos_err:.4f},{yaw_diff:.6f}\n")

        f.write(f"\nRMSE position: {rmse_pos:.4f}\n")
        f.write(f"RMSE yaw (rad): {rmse_yaw:.6f}\n")
        f.write(f"Recall T1: {recall['T1']:.4f}\n")
        f.write(f"Recall T2: {recall['T2']:.4f}\n")
        f.write(f"Recall T3: {recall['T3']:.4f}\n")
        f.write(f"Failure Rate (falhas/km): {failure_rate:.4f}\n")
        f.write(f"Failures: {n_failures}\n")
        f.write(f"Distance (km): {total_km:.4f}\n")

    print(
        f"{base}\n"
        f"  RMSE pos={rmse_pos:.4f}m  yaw={rmse_yaw:.4f}rad\n"
        f"  Recall  T1={recall['T1']:.3f}  T2={recall['T2']:.3f}  T3={recall['T3']:.3f}\n"
        f"  Failures={n_failures}  dist={total_km:.3f}km  FR={failure_rate:.3f} f/km"
    )

    return {
        "rmse_pos":     rmse_pos,
        "rmse_yaw":     rmse_yaw,
        "recall_T1":    recall["T1"],
        "recall_T2":    recall["T2"],
        "recall_T3":    recall["T3"],
        "failure_rate": failure_rate,
        "n_failures":   n_failures,
        "total_km":     total_km,
    }


def main():
    results_dir  = os.path.join(os.path.dirname(__file__), '../results')
    summary_path = os.path.join(results_dir, "summary_results.txt")

    if not os.path.exists(results_dir):
        print("[offline_evaluate] Pasta results/ não encontrada.")
        return

    summary_lines = []

    for filename in sorted(os.listdir(results_dir)):
        if filename.startswith("poses_") and filename.endswith(".txt"):
            pose_path = os.path.join(results_dir, filename)
            base      = filename.replace("poses_", "")

            metrics = rebuild_error_file_from_pose(pose_path, results_dir)

            if metrics:
                summary_lines.append(
                    f"{base},"
                    f"{metrics['rmse_pos']:.4f},"
                    f"{metrics['rmse_yaw']:.4f},"
                    f"{metrics['recall_T1']:.4f},"
                    f"{metrics['recall_T2']:.4f},"
                    f"{metrics['recall_T3']:.4f},"
                    f"{metrics['failure_rate']:.4f},"
                    f"{metrics['n_failures']},"
                    f"{metrics['total_km']:.4f}"
                )

    # ── Cabeçalho do summary ──────────────────────────────────────────────
    with open(summary_path, "w") as f:
        f.write("# filename,rmse_pos,rmse_yaw,recall_T1,recall_T2,recall_T3,failure_rate,n_failures,total_km\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\n[offline_evaluate] Summary salvo: {summary_path}")


if __name__ == "__main__":
    main()
