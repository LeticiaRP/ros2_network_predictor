import os
import collections
import numpy as np
import importlib  # <--- Added
from rclpy.signals import SignalHandlerOptions

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray



# Topology mapping
PLAT_MAP = {
    'h2h': 0.0,      'rtt_h2h': 0.1,
    'h2m': 0.2,      'rtt_h2m': 0.3,
    'm2h': 0.4,      'rtt_m2h': 0.5,
    'm2m': 0.6,      'rtt_m2m': 0.7
}



# Format: { 'scenario': (safety_threshold, failure_threshold) }
THRESHOLD_DEFAULTS = {
    'h2h': (0.8, 2.5),
    'rtt_h2h': (1.5, 5.0),
    'h2m': (6.0, 15.0),
    'rtt_h2m': (18.0, 22.0), # Tighter range for RTT sensitivity
    'm2h': (6.0, 15.0),
    'rtt_m2h': (12.0, 18.0),
    'm2m': (15.0, 35.0),
    'rtt_m2m': (20.0, 45.0)
}



MODEL_TRAINED_PATH = "/home/leticia/ros2_ws/src/ros2_network_predictor/ros2_network_predictor/models/saved_models/universal_tcn.pth"



# TCN ARCHITECTURE COMPONENTS --------------------------------------------------------------------

class Chomp1d(nn.Module):


    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size



    def forward(self, x): 
        return x[:, :, :-self.chomp_size].contiguous()




class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()




    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)





class UniversalQuantileTCN(nn.Module):

    def __init__(self, input_dim, output_len, num_quantiles=3, num_channels=[32, 64, 128, 64], kernel_size=3):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=0.0)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_len * num_quantiles)
        self.num_quantiles = num_quantiles


    def forward(self, x):
        x = x.permute(0, 2, 1) 
        y = self.network(x)
        out = self.linear(y[:, :, -1])
        return out.view(out.shape[0], -1, self.num_quantiles)




# ROS 2 NODE ------------------------------------------------------------------------------------------------------------------

class LatencyForecaster(Node):

    def __init__(self):
        super().__init__('latency_forecaster_node')
        
        # params 
        self.declare_parameter('topic', '/benchmark/latency/h2h')
        self.declare_parameter('msg_pkg', 'std_msgs')
        self.declare_parameter('msg_type', 'Float32')
        self.declare_parameter('scenario', 'h2h')
        self.declare_parameter('frequency_hz', 20.0)
        self.declare_parameter('qos_reliable', True)
        self.declare_parameter('calibration_steps', 100)



        pkg = self.get_parameter('msg_pkg').value
        m_type = self.get_parameter('msg_type').value
        topic_name = self.get_parameter('topic').value
        scenario_str = self.get_parameter('scenario').value.lower()
        

        self.platform_id = PLAT_MAP.get(scenario_str, 0.0)
        self.freq_val = self.get_parameter('frequency_hz').value
        self.qos_val = 1.0 if self.get_parameter('qos_reliable').value else 0.0
        self.cal_limit = self.get_parameter('calibration_steps').value


        try:
            module = importlib.import_module(f"{pkg}.msg")
            msg_class = getattr(module, m_type)
            self.sub = self.create_subscription(msg_class, topic_name, self.universal_callback, 10)
            self.get_logger().info(f"Attached to {topic_name} | Scenario: {scenario_str}")
        except Exception as e:
            self.get_logger().error(f"Import Failed: {e}")
            return


        self.last_arrival_time = None
        self.is_calibrated = False
        self.calibration_data = []
        self.buffer = collections.deque(maxlen=50)
        self.tau = 0.0
        self.f_limit = 0.0


        self.history_actual = []
        self.history_p95 = []


        # model 
        self.model = UniversalQuantileTCN(input_dim=4, output_len=1).to('cpu')

        try:
            self.model.load_state_dict(torch.load(MODEL_TRAINED_PATH, map_location='cpu'))
            self.model.eval()

        except Exception as e:
            self.get_logger().error(f"Model Load Error: {e}")

        self.pub_risk = self.create_publisher(Float32MultiArray, '/network/risk_profile', 10)




    def universal_callback(self, msg):
        now = self.get_clock().now()
        
        if self.last_arrival_time is None:
            self.last_arrival_time = now
            return
        
        latency_ms = (now.nanoseconds - self.last_arrival_time.nanoseconds) / 1e6
        self.last_arrival_time = now

        
        if not self.is_calibrated:
            self.calibration_data.append(latency_ms)
            if len(self.calibration_data) >= self.cal_limit:
                self.tau = np.percentile(self.calibration_data, 95)
                self.f_limit = np.percentile(self.calibration_data, 99) * 1.2
                self.is_calibrated = True
                self.get_logger().info(f"✅ CALIBRATED: τ={self.tau:.2f}ms, F={self.f_limit:.2f}ms")
            return


        # inference
        freq_feat = np.clip(self.freq_val / 200.0, 0.0, 1.0)
        self.buffer.append([latency_ms, self.qos_val, self.platform_id, freq_feat])
        

        if len(self.buffer) < 50: return


        input_t = torch.tensor(np.array(self.buffer)).float().unsqueeze(0)
        with torch.no_grad():
            preds = self.model(input_t).numpy()
            p50, p95 = preds[0, 0, 1], preds[0, 0, 2]

        
        # risk index equation 
        risk_index = np.clip((p95 - self.tau) / (self.f_limit - self.tau), 0.0, 1.0)

        self.history_actual.append(latency_ms)
        self.history_p95.append(p95)


        
        if len(self.history_actual) % 20 == 0:
            status = "OK" if risk_index < 0.4 else "WARN" if risk_index < 0.8 else "FAIL"
            self.get_logger().info(
                f"[{status}] Lat|P95: {latency_ms:5.2f}|{p95:5.2f} ms "
                f"| Bounds [τ:{self.tau:4.1f} F:{self.f_limit:4.1f}] "
                f"| RI: {risk_index:6.1%}"
            )


        # Publish results
        msg_out = Float32MultiArray()
        msg_out.data = [
            float(p50),          # tube center
            float(p95),          # tube upper boundary
            float(risk_index),   # control signal
            float(latency_ms),   # ground truth (Actual)
            float(self.tau),     # dynamic floor
            float(self.f_limit)  # dynamic ceiling
        ]





def main(args=None):
    rclpy.init(args = None, signal_handler_options = SignalHandlerOptions.NO)
    node = LatencyForecaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:

        pass

    
    rclpy.shutdown()