#!/usr/bin/env python3
import os
import numpy as np

GT_X_OFFSET = float(os.environ.get("MCMH_GT_X_OFFSET", "0.7"))

RECALL_THRESHOLDS = {
    "T1": (0.25, np.deg2rad(2.0)),
    "T2": (0.50, np.deg2rad(5.0)),
    "T3": (5.00, np.deg2rad(10.0)),
}

FAILURE_POS_THRESHOLD = 5.0
FAILURE_YAW_THRESHOLD = np.deg2rad(10.0)


def normalize_yaw(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def trajectory_length_xy(points):
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def load_pose_rows(pose_path):
    rows = []
    with open(pose_path) as f:
        next(f)  # skip header
        for line in f:
            vals = line.strip().split(",")
            if len(vals) != 8:
                continue

            _, est_x, est_y, est_yaw, gt_x, gt_y, gt_yaw, mh_rate = map(float, vals)
            rows.append(
                (
                    est_x,
                    est_y,
                    est_yaw,
                    gt_x + GT_X_OFFSET,
                    gt_y,
                    gt_yaw,
                    mh_rate,
                )
            )

    return rows


def calculate_metrics(rows):
    est = np.array([(r[0], r[1], r[2]) for r in rows], dtype=float)
    gt = np.array([(r[3], r[4], r[5]) for r in rows], dtype=float)

    pos_errors = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
    yaw_errors = normalize_yaw(est[:, 2] - gt[:, 2])
    yaw_abs = np.abs(yaw_errors)

    gt_path_m = trajectory_length_xy(gt[:, :2])
    est_path_m = trajectory_length_xy(est[:, :2])
    gt_path_km = gt_path_m / 1000.0

    failure_mask = (pos_errors > FAILURE_POS_THRESHOLD) | (
        yaw_abs > FAILURE_YAW_THRESHOLD
    )
    failure_events = int(failure_mask[0]) if len(failure_mask) else 0
    if len(failure_mask) > 1:
        failure_events += int(np.sum(failure_mask[1:] & ~failure_mask[:-1]))

    success = 1.0 if failure_events == 0 else 0.0
    spl = success * gt_path_m / max(est_path_m, gt_path_m, 1e-9)
    failure_rate = (
        failure_events / gt_path_km
        if gt_path_km > 0.0
        else float("nan")
    )

    recalls = {
        key: float(np.mean((pos_errors < pos_thr) & (yaw_abs < yaw_thr)))
        for key, (pos_thr, yaw_thr) in RECALL_THRESHOLDS.items()
    }

    return {
        "pos_errors": pos_errors,
        "yaw_errors": yaw_errors,
        "rmse_pos": float(np.sqrt(np.mean(np.square(pos_errors)))),
        "rmse_yaw": float(np.sqrt(np.mean(np.square(yaw_errors)))),
        "success": success,
        "spl": float(spl),
        "recall_t1": recalls["T1"],
        "recall_t2": recalls["T2"],
        "recall_t3": recalls["T3"],
        "failure_events": failure_events,
        "path_length_km": gt_path_km,
        "failure_rate": float(failure_rate),
    }


def format_metric(value, digits=4):
    if np.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def rebuild_error_file_from_pose(pose_path, results_dir):
    """
    Reconstructs error file and benchmark metrics from a poses file.
    """
    base = os.path.basename(pose_path).replace("poses_", "")
    error_path = os.path.join(results_dir, base)

    rows = load_pose_rows(pose_path)
    if not rows:
        print(f"No valid data in {pose_path}")
        return None

    metrics = calculate_metrics(rows)
    rmse_pos = metrics["rmse_pos"]
    rmse_yaw = metrics["rmse_yaw"]

    # Save detailed error file
    with open(error_path, "w") as f:
        f.write("time,error_pos,error_yaw\n")

        with open(pose_path) as pf:
            next(pf)
            for line in pf:
                vals = line.strip().split(",")
                if len(vals) != 8:
                    continue

                t, est_x, est_y, est_yaw, gt_x, gt_y, gt_yaw, mh_rate = map(float, vals)
                gt_x += GT_X_OFFSET

                pos_error = np.sqrt((est_x - gt_x) ** 2 + (est_y - gt_y) ** 2)
                yaw_diff = normalize_yaw(est_yaw - gt_yaw)

                f.write(f"{t:.3f},{pos_error:.4f},{yaw_diff:.6f}\n")

        f.write(f"\nRMSE position: {rmse_pos:.4f}\n")
        f.write(f"RMSE yaw (rad): {rmse_yaw:.6f}\n")
        f.write(f"Success: {metrics['success']:.0f}\n")
        f.write(f"SPL: {metrics['spl']:.4f}\n")
        f.write(f"Recall T1 (<0.25m,2deg): {metrics['recall_t1']:.4f}\n")
        f.write(f"Recall T2 (<0.50m,5deg): {metrics['recall_t2']:.4f}\n")
        f.write(f"Recall T3 (<5.00m,10deg): {metrics['recall_t3']:.4f}\n")
        f.write(f"Failure events: {metrics['failure_events']}\n")
        f.write(f"Path length (km): {metrics['path_length_km']:.6f}\n")
        f.write(
            "Failure rate (events/km): "
            f"{format_metric(metrics['failure_rate'], 6)}\n"
        )

    print(
        f"{base} -> Pos RMSE={rmse_pos:.4f} | Yaw RMSE={rmse_yaw:.4f} | "
        f"SR={metrics['success']:.0f} | SPL={metrics['spl']:.3f} | "
        f"R(T1/T2/T3)={metrics['recall_t1']:.2f}/"
        f"{metrics['recall_t2']:.2f}/{metrics['recall_t3']:.2f} | "
        f"F={format_metric(metrics['failure_rate'], 3)} ev/km"
    )

    return metrics


def discover_result_dirs(results_root):
    result_dirs = []

    for current_dir, dirnames, filenames in os.walk(results_root):
        dirnames[:] = [d for d in dirnames if d != "plots"]
        has_pose_files = any(
            filename.startswith("poses_") and filename.endswith(".txt")
            for filename in filenames
        )
        if has_pose_files:
            result_dirs.append(current_dir)

    return sorted(result_dirs)


def process_results_dir(results_dir):
    summary_path = os.path.join(results_dir, "summary_results.txt")

    summary_lines = []

    for filename in sorted(os.listdir(results_dir)):

        if filename.startswith("poses_") and filename.endswith(".txt"):

            pose_path = os.path.join(results_dir, filename)

            base = filename.replace("poses_", "")

            # Always recompute (recommended)
            metrics = rebuild_error_file_from_pose(pose_path, results_dir)

            if metrics is not None:
                summary_lines.append(
                    ",".join(
                        [
                            base,
                            f"{metrics['rmse_pos']:.4f}",
                            f"{metrics['rmse_yaw']:.6f}",
                            f"{metrics['success']:.0f}",
                            f"{metrics['spl']:.4f}",
                            f"{metrics['recall_t1']:.4f}",
                            f"{metrics['recall_t2']:.4f}",
                            f"{metrics['recall_t3']:.4f}",
                            str(metrics["failure_events"]),
                            f"{metrics['path_length_km']:.6f}",
                            format_metric(metrics["failure_rate"], 6),
                        ]
                    )
                )

    # Write summary file
    with open(summary_path, "w") as f:
        f.write(
            "file,rmse_pos,rmse_yaw_rad,success,spl,recall_t1,"
            "recall_t2,recall_t3,failure_events,path_length_km,"
            "failure_rate_events_per_km\n"
        )
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nSummary saved to: {summary_path}")


def main():
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))

    if not os.path.exists(results_root):
        print("Results directory not found.")
        return

    result_dirs = discover_result_dirs(results_root)
    if not result_dirs:
        print("No pose files found.")
        return

    for results_dir in result_dirs:
        print(f"\nProcessing results in: {results_dir}")
        process_results_dir(results_dir)


if __name__ == "__main__":
    main()
