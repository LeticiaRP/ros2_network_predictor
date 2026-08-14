import os
import sys
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import numpy as np
import matplotlib.pyplot as plt

MODEL_TRAINED_PATH = "saved_models/universal_tcn.pth"
SAVE_DIR = "model_results/"

class Chomp1d(nn.Module):


    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size


    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()




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
                                     dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=0.0)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_len * num_quantiles)
        self.num_quantiles = num_quantiles




    def forward(self, x):
        x = x.permute(0, 2, 1) 
        y = self.network(x)
        out = self.linear(y[:, :, -1])
        return out.view(out.shape[0], -1, self.num_quantiles)





def generate_stress_data():

    length = 400
    t = np.linspace(0, 40, length)
    latency = 200000.0 + 50000.0 * np.sin(t) + np.random.normal(0, 20000.0, length)
    qos, plat, freq = np.ones(length), np.zeros(length), np.full(length, 0.3)
    
    # Events
    freq[50:100] = 0.1 
    latency[50:100] += np.random.normal(0, 50000.0, 50)
    qos[150:200] = 0.0 
    latency[150:200] += np.random.choice([0, 300000.0], size=50, p=[0.9, 0.1])
    plat[250:300] = 1.0 
    latency[250:300] = 200000.0 + (np.arange(50) % 5) * 50000.0 
    freq[350:] = 1.0; qos[350:] = 0.0; plat[350:] = 1.0
    latency[350:] += np.linspace(0, 600000.0, 50) 
    
    return np.stack([latency, qos, plat, freq], axis=1), latency




def run_stress_test():
    device = "cpu"
    model = UniversalQuantileTCN(input_dim=4, output_len=1, num_quantiles=3).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_TRAINED_PATH, map_location=device))
    except Exception as e:
        print(f"Error: {e}"); return

    model.eval()
    raw_data, ground_truth = generate_stress_data()
    
    input_len = 50
    calibration_window = ground_truth[:input_len]
    

    tau = np.percentile(calibration_window, 95)
    f_limit = np.percentile(calibration_window, 99) * 1.2

    print(f"Scenario Calibrated:")
    print(f"  > Safety Threshold (tau): {tau/1e6:.4f} ms")
    print(f"  > Failure Threshold (F):   {f_limit/1e6:.4f} ms")

    preds_p95, risk_indices = [], []

    for i in range(len(raw_data) - input_len):
        window = raw_data[i : i+input_len]
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor).numpy()
            p95 = out[0, 0, 2] 
        
        preds_p95.append(p95)
        
        ri = (p95 - tau) / (f_limit - tau)
        risk_indices.append(np.clip(ri, 0, 1))

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })


    plot_x = range(input_len, len(raw_data))
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Axis 1: Latency
    ax1.plot(plot_x, ground_truth[input_len:] / 1e6, 'k', label='Actual Latency', alpha=0.4)
    ax1.plot(plot_x, np.array(preds_p95) / 1e6, 'b', label='P95 Prediction')
    ax1.axhline(y=f_limit/1e6, color='darkred', linestyle=':', label='Failure Limit (F)')
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xlabel("Step")


    # Axis 2: Risk Index
    ax2 = ax1.twinx()
    ax2.plot(plot_x, risk_indices, 'r',  linestyle='--', linewidth=1, label='Risk Index (RI)')
    ax2.set_ylabel("Risk Index [0, 1]", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(-0.1, 1.1)


    # Labeling Events
    events = [(50, "Freq Drop"), (150, "Best Effort"), (250, "Cyclic Jitter"), (350, "Buffer Saturation")]
    for x, txt in events:
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
        ax1.text(x+2, 0.0, txt, fontweight='bold')


    plt.title("Stress Test: Latency vs. Risk Index Anticipation")
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(SAVE_DIR, f"stress_test.pdf"), bbox_inches='tight', dpi=300)
    print("Plot saved as stress_test.png")



if __name__ == "__main__":
    run_stress_test()

