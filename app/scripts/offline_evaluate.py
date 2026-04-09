#!/usr/bin/env python3
import os
import numpy as np

def rebuild_error_file_from_pose(pose_path, results_dir):
    """
    Reconstructs error file (position + yaw RMSE) from poses file.
    """
    base = os.path.basename(pose_path).replace("poses_", "")
    error_path = os.path.join(results_dir, base)

    pos_errors = []
    yaw_errors = []

    with open(pose_path) as f:
        next(f)  # skip header
        for line in f:
            vals = line.strip().split(",")
            if len(vals) != 7:
                continue

            _, est_x, est_y, est_yaw, gt_x, gt_y, gt_yaw = map(float, vals)

            # Position error
            pos_error = np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2)

            # Yaw error (correct wrap)
            yaw_diff = np.arctan2(
                np.sin(est_yaw - gt_yaw),
                np.cos(est_yaw - gt_yaw)
            )

            pos_errors.append(pos_error)
            yaw_errors.append(yaw_diff)

    if not pos_errors:
        print(f"No valid data in {pose_path}")
        return None, None

    rmse_pos = np.sqrt(np.mean(np.square(pos_errors)))
    rmse_yaw = np.sqrt(np.mean(np.square(yaw_errors)))

    # Save detailed error file
    with open(error_path, "w") as f:
        f.write("time,error_pos,error_yaw\n")

        with open(pose_path) as pf:
            next(pf)
            for line in pf:
                vals = line.strip().split(",")
                if len(vals) != 7:
                    continue

                t, est_x, est_y, est_yaw, gt_x, gt_y, gt_yaw = map(float, vals)

                pos_error = np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2)
                yaw_diff = np.arctan2(
                    np.sin(est_yaw - gt_yaw),
                    np.cos(est_yaw - gt_yaw)
                )

                f.write(f"{t:.3f},{pos_error:.4f},{yaw_diff:.6f}\n")

        f.write(f"\nRMSE position: {rmse_pos:.4f}\n")
        f.write(f"RMSE yaw (rad): {rmse_yaw:.6f}\n")

    print(f"{base} → Pos RMSE={rmse_pos:.4f} | Yaw RMSE={rmse_yaw:.4f}")

    return rmse_pos, rmse_yaw


def main():
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    summary_path = os.path.join(results_dir, "summary_results.txt")

    if not os.path.exists(results_dir):
        print("Results directory not found.")
        return

    summary_lines = []

    for filename in sorted(os.listdir(results_dir)):

        if filename.startswith("poses_") and filename.endswith(".txt"):

            pose_path = os.path.join(results_dir, filename)

            base = filename.replace("poses_", "")
            error_path = os.path.join(results_dir, base)

            # Always recompute (recommended)
            rmse_pos, rmse_yaw = rebuild_error_file_from_pose(pose_path, results_dir)

            if rmse_pos is not None and rmse_yaw is not None:
                summary_lines.append(
                    f"{base},{rmse_pos:.4f},{rmse_yaw:.4f}"
                )

    # Write summary file
    with open(summary_path, "w") as f:
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()