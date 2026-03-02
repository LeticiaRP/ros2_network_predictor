#!/home/leticia/.pyenv/versions/ros2-predictor/bin/python3
import rclpy
from rclpy.node import Node
import torch
import os

from ros2_network_predictor.forecaster.latency_forecaster import LatencyForecaster 


import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import torch
import os


def main(args=None):
    rclpy.init(args=args)
    
    node = LatencyForecaster()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()