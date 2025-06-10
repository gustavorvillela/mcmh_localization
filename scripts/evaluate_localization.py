#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelStates
import os
import time

class Evaluator:
    def __init__(self):
        self.est_topic = rospy.get_param("~est_topic", "/estimated_pose")
        self.gt_topic = rospy.get_param("~gt_topic", "/gazebo/model_states")
        self.robot_name = rospy.get_param("~robot_name", "turtlebot3_waffle")

        self.sim_end_time = rospy.get_param("~sim_end_time", 120)
        self.start_time = None
        self.eval_start_time = None
        
        # Corrigido: Removida a extensão .txt duplicada
        result_name = rospy.get_param("~result_name", "eval")
        results_dir = os.path.join(os.path.dirname(__file__), "../results")
        os.makedirs(results_dir, exist_ok=True)

        # Garante que o nome do arquivo termine com .txt (sem duplicação)
        if not result_name.endswith('.txt'):
            result_name += '.txt'
        
        self.output_file = os.path.join(results_dir, result_name)

        self.gt_pose = None
        self.errors = []
        self.timestamps = []  # Novo: armazena os timestamps dos erros
        self.error_history = []  # Novo: armazena histórico completo

        rospy.Subscriber(self.est_topic, PoseWithCovarianceStamped, self.estimated_callback)
        rospy.Subscriber(self.gt_topic, ModelStates, self.gt_callback)

    def estimated_callback(self, msg):
        if self.gt_pose is None:
            return
            
        if self.eval_start_time is None:
            self.eval_start_time = rospy.Time.now()
            
        est_position = msg.pose.pose.position
        error = np.linalg.norm(np.array([
            est_position.x - self.gt_pose.position.x,
            est_position.y - self.gt_pose.position.y,
        ]))
        
        # Novo: armazena timestamp e erro
        elapsed_time = (rospy.Time.now() - self.eval_start_time).to_sec()
        self.timestamps.append(elapsed_time)
        self.errors.append(error)
        self.error_history.append((elapsed_time, error))  # Armazena tuplas (tempo, erro)

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
        
        # Novo: salva histórico completo e RMSE final
        with open(self.output_file, "w") as f:
            # Escreve cabeçalho
            f.write("time,error\n")
            
            # Escreve todos os dados históricos
            for timestamp, error in self.error_history:
                f.write(f"{timestamp:.3f},{error:.4f}\n")
                
            # Escreve RMSE no final
            f.write(f"\nRMSE final: {rmse:.4f}\n")
            
        rospy.loginfo(f"Resultados salvos em: {self.output_file}")
        rospy.loginfo(f"RMSE final: {rmse:.4f}")


if __name__ == "__main__":
    rospy.init_node("evaluate_localization")
    evaluator = Evaluator()
    try:
        evaluator.run()
    except rospy.ROSInterruptException:
        pass