#!/usr/bin/env python3
"""
evaluate_localization.py
------------------------
Grava poses (estimada vs ground truth) durante a execução do localizador.
Detecta FALHAS de localização em tempo real usando janela deslizante:
  - Falha = erro de posição > FAILURE_THRESHOLD por >= FAILURE_WINDOW timesteps consecutivos
  - Ao detectar falha, registra o evento e "reinicializa" (reset do contador)
  - Calcula Failure Rate = reinicializações / km percorrido (como no paper)
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelStates
import os
from tf.transformations import euler_from_quaternion


# ─── Parâmetros de detecção de falha ────────────────────────────────────────
FAILURE_THRESHOLD  = 1.0   # metros — erro acima disso é candidato a falha
FAILURE_WINDOW     = 10    # timesteps consecutivos acima do threshold = falha confirmada
# ─────────────────────────────────────────────────────────────────────────────


class Evaluator:
    def __init__(self):
        self.est_topic  = rospy.get_param("~est_topic",  "/estimated_pose")
        self.gt_topic   = rospy.get_param("~gt_topic",   "/gazebo/model_states")
        self.robot_name = rospy.get_param("~robot_name", "turtlebot3_waffle")

        result_param = rospy.get_param("~result_name", "eval")
        result_name  = os.path.basename(result_param).replace(".txt", "")

        results_dir = os.path.join(os.path.dirname(__file__), "../results")
        os.makedirs(results_dir, exist_ok=True)

        self.poses_file   = os.path.join(results_dir, f"poses_{result_name}.txt")
        self.failure_file = os.path.join(results_dir, f"failures_{result_name}.txt")

        self.gt_pose = None

        # ── Histórico de poses ──────────────────────────────────────────────
        self.pose_history = []   # lista de tuplas (t, ex, ey, eyaw, gx, gy, gyaw)

        # ── Estado da janela deslizante de falha ────────────────────────────
        self.consecutive_failures = 0   # quantos timesteps consecutivos acima do threshold
        self.failure_events       = []  # lista de (timestamp, pos_error) de cada falha confirmada
        self.in_failure           = False  # evita contar a mesma falha várias vezes

        # ── Odometria acumulada (para Failure Rate em km) ───────────────────
        self.prev_gt_pos   = None   # (x, y) do timestep anterior
        self.total_distance_m = 0.0

        rospy.Subscriber(self.est_topic, PoseWithCovarianceStamped, self.estimated_callback)
        rospy.Subscriber(self.gt_topic,  ModelStates,               self.gt_callback)

    # ────────────────────────────────────────────────────────────────────────
    def get_yaw(self, pose):
        q = [pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w]
        _, _, yaw = euler_from_quaternion(q)
        return yaw

    # ────────────────────────────────────────────────────────────────────────
    def gt_callback(self, msg):
        if self.robot_name not in msg.name:
            return
        idx = msg.name.index(self.robot_name)
        self.gt_pose = msg.pose[idx]

    # ────────────────────────────────────────────────────────────────────────
    def estimated_callback(self, msg):
        if self.gt_pose is None:
            return

        timestamp = msg.header.stamp.to_sec()

        est = msg.pose.pose
        est_x   = est.position.x
        est_y   = est.position.y
        est_yaw = self.get_yaw(est)

        gt_x   = self.gt_pose.position.x
        gt_y   = self.gt_pose.position.y
        gt_yaw = self.get_yaw(self.gt_pose)

        # ── Acumula distância percorrida ─────────────────────────────────
        cur_gt = np.array([gt_x, gt_y])
        if self.prev_gt_pos is not None:
            self.total_distance_m += float(np.linalg.norm(cur_gt - self.prev_gt_pos))
        self.prev_gt_pos = cur_gt

        # ── Erro de posição ──────────────────────────────────────────────
        pos_error = float(np.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2))

        # ── Lógica de janela deslizante ──────────────────────────────────
        if pos_error > FAILURE_THRESHOLD:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
            self.in_failure = False   # saiu da falha → pronto para detectar nova

        if self.consecutive_failures >= FAILURE_WINDOW and not self.in_failure:
            self.in_failure = True
            self.failure_events.append((timestamp, pos_error))
            rospy.logwarn(
                f"[Evaluator] FALHA detectada t={timestamp:.2f}s "
                f"pos_error={pos_error:.3f}m "
                f"(falhas totais={len(self.failure_events)})"
            )

        # ── Registra pose ────────────────────────────────────────────────
        self.pose_history.append((
            timestamp,
            est_x, est_y, est_yaw,
            gt_x,  gt_y,  gt_yaw,
            pos_error
        ))

    # ────────────────────────────────────────────────────────────────────────
    def run(self):
        rospy.loginfo("[Evaluator] Gravando poses + detectando falhas...")
        rospy.spin()

    # ────────────────────────────────────────────────────────────────────────
    def save_results(self):
        if not self.pose_history:
            rospy.logwarn("[Evaluator] Nenhuma pose gravada.")
            return

        # ── Salva poses ──────────────────────────────────────────────────
        with open(self.poses_file, "w") as f:
            f.write("time,est_x,est_y,est_yaw,gt_x,gt_y,gt_yaw,pos_error\n")
            for d in self.pose_history:
                f.write(
                    f"{d[0]:.6f},{d[1]:.4f},{d[2]:.4f},{d[3]:.6f},"
                    f"{d[4]:.4f},{d[5]:.4f},{d[6]:.6f},{d[7]:.4f}\n"
                )
        rospy.loginfo(f"[Evaluator] Poses salvas: {self.poses_file}")

        # ── Salva eventos de falha ───────────────────────────────────────
        total_km = self.total_distance_m / 1000.0
        n_failures = len(self.failure_events)
        failure_rate = n_failures / total_km if total_km > 0 else float('nan')

        with open(self.failure_file, "w") as f:
            f.write(f"# Parâmetros: threshold={FAILURE_THRESHOLD}m, janela={FAILURE_WINDOW} timesteps\n")
            f.write(f"# Total de falhas detectadas: {n_failures}\n")
            f.write(f"# Distância percorrida: {self.total_distance_m:.2f} m ({total_km:.4f} km)\n")
            f.write(f"# Failure Rate (falhas/km): {failure_rate:.4f}\n")
            f.write("timestamp,pos_error\n")
            for (t, err) in self.failure_events:
                f.write(f"{t:.6f},{err:.4f}\n")

        rospy.loginfo(
            f"[Evaluator] Falhas: {n_failures} | "
            f"Distância: {self.total_distance_m:.1f}m | "
            f"Failure Rate: {failure_rate:.3f} falhas/km"
        )
        rospy.loginfo(f"[Evaluator] Arquivo de falhas: {self.failure_file}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rospy.init_node("evaluate_localization")
    evaluator = Evaluator()
    try:
        evaluator.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        evaluator.save_results()
