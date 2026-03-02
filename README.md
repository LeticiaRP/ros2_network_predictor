# ROS 2 Network Latency Predictor 

An AI-driven framework for anticipating communication risks in ROS 2 and micro-ROS systems. This project uses a Universal Quantile Temporal Convolutional Network (TCN) to predict latency envelopes and provide proactive Risk Index (RI) signals for Autonomous Mobile Robots (AMRs) and aerial platforms.



## Key Features
- Quantile TCN: Predicts the $P_{05}$, $P_{50}$, and $P_{95}$ latency quantiles to create a Safety Tube.
- Localized Risk Index: Anticipates network failure based on frequency-specific Safety ($\tau$) and Failure ($F$) thresholds.
- Multi-Topology Support: Pre-configured for Host-to-Host (H2H), Micro-to-Micro (M2M), and Round-Trip Time (RTT) analysis.
- Stress Testing: Built-in scenarios for simulating Cyclic Jitter and Buffer Saturation.



## Project Structure
- ``data/raw/``: CSV datasets organized by Topology, QoS, and Frequency.
- ``ros2_network_predictor/models/``: Core training (tcn_train.py) and evaluation (evaluate_scenarios.py) logic.
- ``ros2_network_predictor/utils/``: Implementation of the UniversalQuantileTCN and calibration tools.
- ``config/`: Model weights. 



## Installation

1. Put the project into your ROS 2 workspace

2. Install dependencies
```bash
pip install torch numpy pandas matplotlib pyyaml
```

3. Build the package:
```bash
cd ~/ros2_ws
colcon build --packages-select ros2_network_predictor
source install/setup.bash
``` 

## Usage
1. Evaluate Scenarios

Analyze the performance of the TCN across all collected data. This script generates the Safety Tube and Risk Index plots for every topology.

```bash
python3 ros2_network_predictor/models/evaluate_scenarios.py
```
Outputs are saved in ``ros2_network_predictor/models/model_results/``.

2. Run Stress Test Simulation

Test the model's ability to anticipate Buffer Saturation and Cyclic Jitter using a synthetic stress environment.

```bash
python3 ros2_network_predictor/models/tcn_test.py
```

3. Launch as a ROS 2 Node
Integrate the predictor into your live robot control loop to monitor network health in real-time.
```bash
ros2 launch ros2_network_predictor predictor.launch.py
```


# Results Visualization
The evaluation script generates comprehensive subplots comparing:
- Best Effort vs. Reliable QoS across 10Hz, 50Hz, and 200Hz.
- Actual Latency vs. Predicted Envelopes.
- Real-time Risk Index spikes corresponding to network anomalies.