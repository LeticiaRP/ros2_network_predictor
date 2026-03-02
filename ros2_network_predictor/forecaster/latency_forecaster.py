import os
import collections
import numpy as np

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



MODEL_TRAINED_PATH = "../models/saved_models/universal_tcn.pth"



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
        
        
        self.declare_parameter('scenario', 'h2h')
        self.declare_parameter('frequency_hz', 20.0)
        self.declare_parameter('qos_reliable', True)
        self.declare_parameter('safety_threshold', -1.0) 
        self.declare_parameter('failure_threshold', -1.0)
        self.declare_parameter('alpha', 0.6) 

        
        scenario_str = self.get_parameter('scenario').value.lower()
        self.platform_id = PLAT_MAP.get(scenario_str, 0.0)
        self.freq_val = self.get_parameter('frequency_hz').value
        self.qos_val = 1.0 if self.get_parameter('qos_reliable').value else 0.0
        self.alpha = self.get_parameter('alpha').value
        safety_param = self.get_parameter('safety_threshold').value
        failure_param = self.get_parameter('failure_threshold').value
        

        # if safety_param < 0: # If not set manually by user
        self.safety_threshold, self.failure_threshold = THRESHOLD_DEFAULTS.get(scenario_str, (1.0, 5.0))
        # else:
        #     self.safety_threshold, self.failure_threshold = safety_param, failure_param


        device = 'cpu'
        self.model = UniversalQuantileTCN(input_dim=4, output_len=1, num_quantiles=3).to(device)

        try:
            self.model.load_state_dict(torch.load(MODEL_TRAINED_PATH, map_location=device))
            self.model.eval()
            self.get_logger().info(f"Model Loaded. Using thresholds for {scenario_str}: Safety={self.safety_threshold}, Failure={self.failure_threshold}")
        except Exception as e:
            self.get_logger().error(f"Failed to load weights: {e}")
            return


        self.buffer = collections.deque(maxlen=50)
        self.smoothed_p95 = self.safety_threshold
        self.warning_active = False
        self.t_warning = 0.0
        self.total_samples = 0
        self.reliability_hits = 0


        self.sub = self.create_subscription(Float32, f'/benchmark/latency/{scenario_str}', self.callback, 10)
        self.pub_risk = self.create_publisher(Float32MultiArray, '/network/risk_profile', 10)






    def callback(self, msg):
        actual_latency = msg.data
        current_time = self.get_clock().now().nanoseconds / 1e9
        freq_feature = np.clip(self.freq_val / 200.0, 0.0, 1.0)


        self.buffer.append([actual_latency, self.qos_val, self.platform_id, freq_feature])
        if len(self.buffer) < 50: return

        # Inference
        input_tensor = torch.from_numpy(np.array(self.buffer)).float().unsqueeze(0)
        with torch.no_grad():
            preds = self.model(input_tensor)


        median = preds[0, 0, 1].item()
        p95_raw = preds[0, 0, 2].item()


        # Sanity check: ensure P95 is logically >= median
        p95_validated = max(p95_raw, median + 0.1)



        # Exponential smoothing
        self.smoothed_p95 = (self.alpha * p95_validated) + ((1.0 - self.alpha) * self.smoothed_p95)



        # risk calculation (sensitivity adjustment)
        linear_risk = np.clip((self.smoothed_p95 - self.safety_threshold) / 
                             (self.failure_threshold - self.safety_threshold), 0.0, 1.0)
        

        # power scaling makes the risk index more reactive at high frequency
        risk_index = np.power(linear_risk, 0.5) 


        # reliability
        self.total_samples += 1
        if actual_latency <= self.smoothed_p95: self.reliability_hits += 1
        reliability = self.reliability_hits / self.total_samples



        # pub
        msg_out = Float32MultiArray()
        msg_out.data = [median, self.smoothed_p95, self.smoothed_p95 - median, risk_index, reliability]
        self.pub_risk.publish(msg_out)



        # log warnings
        if risk_index > 0.4 and not self.warning_active:
            self.t_warning = current_time
            self.warning_active = True
            self.get_logger().warn(f"RISK DETECTED: {risk_index:.2%} | Anticipated P95: {self.smoothed_p95:.2f}ms")
        


        if actual_latency > (self.failure_threshold * 0.8) and self.warning_active:
            lead_time = (current_time - self.t_warning) * 1000.0
            self.get_logger().info(f"VALIDATION | Lead-Time: {lead_time:.1f}ms")
            self.warning_active = False





def main(args=None):
    rclpy.init(args=args)
    node = LatencyForecaster()  
    try:
        if hasattr(node, 'model'):
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()