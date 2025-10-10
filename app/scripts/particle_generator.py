#!/usr/bin/env python3

import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class ParticleMarkerPublisher:
    def __init__(self):
        rospy.init_node('particle_generator', anonymous=True)
        
        # Parâmetros
        self.num_particles = rospy.get_param('~num_particles', 1000)
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)
        self.particle_scale = rospy.get_param('~particle_scale', 0.1)  # Tamanho das partículas
        self.particle_color = rospy.get_param('~particle_color', [1.0, 0.0, 0.0, 1.0])  # RGBA
        
        # Variáveis do mapa
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None
        
        # Subscribers
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        
        # Publishers
        self.marker_pub = rospy.Publisher('/particle_markers', MarkerArray, queue_size=10)
        
        # Timer
        rospy.Timer(rospy.Duration(1.0/self.publish_rate), self.publish_markers)
        
    def map_callback(self, msg):
        """Callback para receber dados do mapa"""
        self.map_data = msg.data
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        rospy.loginfo("Mapa recebido. Resolução: %f, Dimensões: %d x %d", 
                     self.map_resolution, self.map_width, self.map_height)
    
    def is_valid_position(self, x, y):
        """Verifica se a posição está no espaço livre do mapa"""
        if not self.map_data:
            return False
            
        # Converter coordenadas do mundo para coordenadas do mapa
        map_x = int((x - self.map_origin.x) / self.map_resolution)
        map_y = int((y - self.map_origin.y) / self.map_resolution)
        
        # Verificar se está dentro dos limites do mapa
        if map_x < 0 or map_x >= self.map_width or map_y < 0 or map_y >= self.map_height:
            return False
            
        # Verificar se é espaço livre (0 no occupancy grid)
        index = map_y * self.map_width + map_x
        return self.map_data[index] == 0
    
    def generate_particle_markers(self):
        """Gera markers para partículas aleatórias dentro do espaço livre do mapa"""
        marker_array = MarkerArray()
        
        if not self.map_data:
            rospy.logwarn("Mapa ainda não recebido. Não é possível gerar partículas.")
            return marker_array
            
        # Criar um marker para todas as partículas (mais eficiente que um marker por partícula)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "particles"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        # Configuração do marker
        marker.scale.x = self.particle_scale  # Largura dos pontos
        marker.scale.y = self.particle_scale  # Altura dos pontos
        marker.color.r = self.particle_color[0]
        marker.color.g = self.particle_color[1]
        marker.color.b = self.particle_color[2]
        marker.color.a = self.particle_color[3]
        
        count = 0
        while count < self.num_particles:
            # Gerar coordenadas aleatórias
            x = np.random.uniform(self.map_origin.x, 
                                self.map_origin.x + self.map_width * self.map_resolution)
            y = np.random.uniform(self.map_origin.y, 
                                self.map_origin.y + self.map_height * self.map_resolution)
            
            if self.is_valid_position(x, y):
                point = Point()
                point.x = x
                point.y = y
                point.z = 0  # No plano 2D
                marker.points.append(point)
                count += 1
        
        marker_array.markers.append(marker)
        
        # Adicionar markers para orientação (opcional)
        if True:  # Mudar para True se quiser setas mostrando orientação
            for i in range(len(marker.points)):
                arrow_marker = Marker()
                arrow_marker.header = marker.header
                arrow_marker.ns = "particle_arrows"
                arrow_marker.id = i + 1
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                
                # Posição e orientação aleatória
                yaw = np.random.uniform(0, 2*np.pi)
                quat = quaternion_from_euler(0, 0, yaw)
                
                arrow_marker.pose.position = marker.points[i]
                arrow_marker.pose.orientation.x = quat[0]
                arrow_marker.pose.orientation.y = quat[1]
                arrow_marker.pose.orientation.z = quat[2]
                arrow_marker.pose.orientation.w = quat[3]
                
                arrow_marker.scale.x = self.particle_scale * 2  # Comprimento
                arrow_marker.scale.y = self.particle_scale * 0.5  # Largura
                arrow_marker.scale.z = self.particle_scale * 0.5  # Altura
                arrow_marker.color.r = 0.0
                arrow_marker.color.g = 1.0
                arrow_marker.color.b = 0.0
                arrow_marker.color.a = 1.0
                
                marker_array.markers.append(arrow_marker)
        
        return marker_array
    
    def publish_markers(self, event):
        """Publica os markers das partículas"""
        marker_array = self.generate_particle_markers()
        self.marker_pub.publish(marker_array)
        rospy.loginfo("Publicados %d markers de partículas", self.num_particles)

if __name__ == '__main__':
    try:
        pmp = ParticleMarkerPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass