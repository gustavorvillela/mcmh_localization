#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose
from gazebo_msgs.msg import ModelStates
import os
import time
from tf.transformations import euler_from_quaternion

class Evaluator:
    def __init__(self):
        self.est_topic = rospy.get_param("~est_topic", "/estimated_pose")
        self.gt_topic = rospy.get_param("~gt_topic", "/gazebo/model_states")
        self.robot_name = rospy.get_param("~robot_name", "turtlebot3_waffle")

        self.sim_end_time = rospy.get_param("~sim_end_time", 120)
        self.start_time = None
        self.eval_start_time = None
        
         # Get just the base filename, not full path
        result_param = rospy.get_param("~result_name", "eval")
        result_name = os.path.basename(result_param).replace(".txt", "")
        
        # Create results directory
        results_dir = os.path.join(os.path.dirname(__file__), "../results")
        pose_dir = os.path.join(os.path.dirname(__file__), "../results")
        os.makedirs(results_dir, exist_ok=True)

        
        # Create file paths
        self.output_file = os.path.join(results_dir, f"{result_name}.txt")
        
        self.poses_file = os.path.join(pose_dir, f"poses_{result_name}.txt")

        self.gt_pose = None
        self.est_pose = None  # Nova variável para armazenar pose estimada
        self.errors = []
        self.timestamps = []
        self.error_history = []
        self.pose_history = []  # Novo: armazena histórico completo de poses

        rospy.Subscriber(self.est_topic, PoseWithCovarianceStamped, self.estimated_callback)
        rospy.Subscriber(self.gt_topic, ModelStates, self.gt_callback)

    def estimated_callback(self, msg):
        if self.gt_pose is None:
            return
            
        if self.eval_start_time is None:
            self.eval_start_time = rospy.Time.now()
            
        est_position = msg.pose.pose.position
        est_orientation = msg.pose.pose.orientation
        self.est_pose = msg.pose.pose  # Armazena a pose estimada completa
        
        # Calcula erro de posição
        pos_error = np.linalg.norm(np.array([
            est_position.x - self.gt_pose.position.x,
            est_position.y - self.gt_pose.position.y,
        ]))
        
        # Calcula erro de orientação (yaw)
        gt_yaw = self.get_yaw_from_pose(self.gt_pose)
        est_yaw = self.get_yaw_from_pose(self.est_pose)
        yaw_error = abs(gt_yaw - est_yaw)
        
        # Armazena dados
        elapsed_time = (rospy.Time.now() - self.eval_start_time).to_sec()
        self.timestamps.append(elapsed_time)
        self.errors.append(pos_error)
        self.error_history.append((elapsed_time, pos_error))
        
        # Novo: armazena poses completas
        self.pose_history.append((
            elapsed_time,
            est_position.x, est_position.y, est_yaw,
            self.gt_pose.position.x, self.gt_pose.position.y, gt_yaw
        ))

    def get_yaw_from_pose(self, pose):
        """Extrai o ângulo yaw de uma mensagem Pose"""
        orientation = pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(quat)
        return yaw

    def gt_callback(self, msg):
        if self.robot_name not in msg.name:
            return
        idx = msg.name.index(self.robot_name)
        self.gt_pose = msg.pose[idx]

    def run(self):
        rospy.loginfo("Avaliação iniciada...")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            
            if self.start_time is None and now.to_sec() > 0:
                self.start_time = now

            if self.sim_end_time and self.start_time:
                elapsed = (now - self.start_time).to_sec()
                if elapsed >= self.sim_end_time:
                    rospy.loginfo("Tempo de simulação alcançado. Encerrando avaliação.")
                    break

            rate.sleep()

        self.save_results()

    def save_results(self):
        if not self.errors:
            rospy.logwarn("Nenhum erro registrado. Verifique os topicos.")
            return
            
        rmse = np.sqrt(np.mean(np.square(self.errors)))
        
        # --- Salva erros detalhados dessa execução ---
        with open(self.output_file, "w") as f:
            f.write("time,error\n")
            for timestamp, error in self.error_history:
                f.write(f"{timestamp:.3f},{error:.4f}\n")
            f.write(f"\nRMSE final: {rmse:.4f}\n")
        
        with open(self.poses_file, "w") as f:
            f.write("time,est_x,est_y,est_yaw,gt_x,gt_y,gt_yaw\n")
            for data in self.pose_history:
                f.write(f"{data[0]:.3f},{data[1]:.4f},{data[2]:.4f},{data[3]:.4f},"
                        f"{data[4]:.4f},{data[5]:.4f},{data[6]:.4f}\n")

        # --- NOVO: salva RMSE em um arquivo acumulado ---
        summary_file = os.path.join(os.path.dirname(__file__), "../results/summary_results.txt")
        with open(summary_file, "a") as f:
            f.write(f"{os.path.basename(self.output_file)},{rmse:.4f}\n")

        rospy.loginfo(f"Resultados salvos:")
        rospy.loginfo(f"- Dados de erro: {self.output_file}")
        rospy.loginfo(f"- Dados de poses: {self.poses_file}")
        rospy.loginfo(f"RMSE final: {rmse:.4f}")



if __name__ == "__main__":
    rospy.init_node("evaluate_localization")
    evaluator = Evaluator()
    try:
        evaluator.run()
    except rospy.ROSInterruptException:
        pass