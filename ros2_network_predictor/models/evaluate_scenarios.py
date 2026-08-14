import os
import glob
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- GLOBAL CONFIGURATION ---
INPUT_LEN = 50
OUTPUT_LEN = 1
DEVICE = "cpu"  
DATA_PATH = '../../data/test'
MODEL_TRAINED_PATH = "saved_models/universal_tcn.pth"
SAVE_DIR = "model_results/"

# Mapping for the model's internal representation
PLAT_MAP = {
    'h2h': 0.0,      'rtt_h2m': 0.3,
    'h2m': 0.2,      'rtt_m2h': 0.5,
    'm2h': 0.4,      'm2m': 0.6,      
}



# ARCHITECTURE CLASSES -------------------------------------------------------------------------------------------------------------------

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
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
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
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=0.0)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_len * num_quantiles)
        self.num_quantiles = num_quantiles

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        y = self.network(x)
        out = self.linear(y[:, :, -1])
        return out.view(out.shape[0], -1, self.num_quantiles)




def parse_metadata(fname):
    basename = os.path.basename(fname)
    parts = basename.split('_')
    freq_str = next((p for p in parts if 'Hz' in p), "0Hz")
    return float(freq_str.replace('Hz', '')), freq_str



def load_data(filepath):
    df = pd.read_csv(filepath)
    target_col = next((c for c in df.columns if ('latency' in c or 'rtt' in c) and 'ns' in c), None)
    if not target_col: return None, None
    lat_ms = pd.to_numeric(df[target_col], errors='coerce').dropna().values.astype(np.float32) / 1e6
    parts = os.path.basename(filepath).split('_')
    qos = 1.0 if 'reliable' in parts else 0.0
    topo = next((k for k in PLAT_MAP.keys() if k in os.path.basename(filepath)), 'h2h')
    f_val, _ = parse_metadata(filepath)
    data = np.stack([lat_ms, np.full_like(lat_ms, qos), np.full_like(lat_ms, PLAT_MAP[topo]), np.full_like(lat_ms, f_val/200.0)], axis=1)
    X = [data[i:i+INPUT_LEN] for i in range(len(data) - INPUT_LEN)]
    return torch.tensor(np.array(X)), lat_ms[INPUT_LEN:]








# STATISTICAL ANALYSIS -----------------------------------------------------------------------------------------------------

def plot_statistical_sweep(model, all_files):

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })
        
    summary = []
    for f_path in all_files:
        if 'jitter' in f_path: continue
        f_val, f_str = parse_metadata(f_path)
        qos = 'reliable' if 'reliable' in f_path else 'best_effort'
        topo = next((k for k in PLAT_MAP.keys() if k in f_path), 'h2h')
        
        X_val, actual = load_data(f_path)
        if X_val is None: continue
        with torch.no_grad():
            preds = model(X_val.to(DEVICE)).cpu().numpy()
        
        coverage = np.mean(actual <= preds[:, 0, 2]) * 100
        summary.append({'Frequency': f_val, 'Coverage': coverage, 'Topology': topo, 'QoS': qos})
    

    df = pd.DataFrame(summary).sort_values('Frequency')

    plt.figure(figsize=(12, 7), dpi=300)
    sns.lineplot(data=df, x='Frequency', y='Coverage', hue='Topology', style='QoS', markers=True, markersize=8)
    plt.axhline(y=95, color='red', linestyle='--', label='95% Target', linewidth=2)
    plt.title("P95 Coverage Stability Across Frequencies", fontsize=14)
    plt.ylabel("Coverage Rate (%)")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True, alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(SAVE_DIR, "_statistical_sweep.png"), bbox_inches='tight')
    plt.close()




def plot_shaded_safety_panels(model, all_files, topo):

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

    freqs = ['10.0Hz', '50.0Hz', '200.0Hz']
    qos_types = ['best_effort', 'reliable']
    
    # Create the multi-panel figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    
    for row, f_str in enumerate(freqs):
        for col, qos in enumerate(qos_types):
            f_list = [f for f in all_files if topo in f and f_str in f and qos in f]
            if not f_list: continue
            
            X_val, actual = load_data(f_list[0])
            with torch.no_grad():
                preds = model(X_val.to(DEVICE)).cpu().numpy()
            
            # 0: P05 (Lower), 1: P50 (Median/Prediction), 2: P95 (Upper)
            p05 = preds[:, 0, 0]
            p50 = preds[:, 0, 1]
            p95 = preds[:, 0, 2]
            
            ax = axes[row, col]
            steps = np.arange(len(p50))[:300]
            
            # Plot the raw latency
            ax.plot(steps, actual[:300], color='black', alpha=0.6, linewidth=0.8, label='Actual')

            # Plot the median prediction 
            ax.plot(steps, p50[:300], color='blue', linewidth=1.2, label='Prediction (P50)')

            # 3. Create the shaded safety tube (P05 to P95)
            ax.fill_between(steps, p05[:300], p95[:300], color='blue', alpha=0.2, label='Safety Tube ($P_{05}$-$P_{95}$)')

            ax.set_title(f"{f_str} | {qos.upper().replace('_', ' ')}")
            if col == 0: ax.set_ylabel("Latency (ms)")
            if row == 2: ax.set_xlabel("Packet Step")
            ax.grid(True, alpha=0.15)

    # add a unified legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"shaded_tube_{topo}.png"), bbox_inches='tight', dpi=300)
    plt.close()





def plot_shaded_with_risk(model, all_files, topo):

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })

    freqs = ['10.0Hz', '50.0Hz', '200.0Hz']
    qos_types = ['best_effort', 'reliable']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    
    for row, f_str in enumerate(freqs):
        for col, qos in enumerate(qos_types):
            f_list = [f for f in all_files if topo in f and f_str in f and qos in f]
            if not f_list: continue
            
            X_val, actual = load_data(f_list[0])
            with torch.no_grad():
                preds = model(X_val.to(DEVICE)).cpu().numpy()
            
            p05, p50, p95 = preds[:, 0, 0], preds[:, 0, 1], preds[:, 0, 2]
        
            tau = np.percentile(actual, 95)
            failure_F = np.percentile(actual, 99) * 1.1
            
            # Calculate risk index (RI)
            ri = np.clip((p95 - tau) / (failure_F - tau), 0, 1)
                        
            ax1 = axes[row, col]
            steps = np.arange(len(p50))[:300]
            

            ax1.plot(steps, actual[:300], color='black', alpha=0.6, lw=0.8, label='Actual')
            ax1.plot(steps, p50[:300], color='blue', linewidth=1.3, label=r'Pred ($P_{50}$)')
            ax1.fill_between(steps, p05[:300], p95[:300], color='blue', alpha=0.15, label='Safety Tube')
            ax1.axhline(y=tau, color='green', linestyle=':', linewidth=1.2,  alpha=0.8, label=r'$\tau$ (Local)')
            ax1.axhline(y=failure_F, color='m', linestyle=':', linewidth=1.2, alpha=0.8, label=r'$F$ (Local)')
            ax1.set_ylabel("Latency (ms)")
            
            # Secondary Axis: Risk Index
            ax2 = ax1.twinx()
            ax2.plot(steps, ri[:300], color='red', linewidth=1.3, linestyle='--', alpha=0.6, label='Risk Index ($RI$)')
            ax2.set_ylabel("Risk Index", color='red')
            ax2.set_ylim(0, 1.1)
            ax2.tick_params(axis='y', labelcolor='red')


            ax1.set_title(f"{f_str} | {qos.upper()}")
            if row == 2: ax1.set_xlabel("Packet Step")
            ax1.grid(True, alpha=0.05)



    lines_1, labels_1 = axes[0, 0].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), ncol=6, fontsize='small')
    

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"risk_index_{topo}.png"), bbox_inches='tight', dpi=300)
    plt.close()




if __name__ == "__main__":


    try:

        os.makedirs(SAVE_DIR, exist_ok=True)
        model = UniversalQuantileTCN(input_dim=4, output_len=OUTPUT_LEN).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_TRAINED_PATH, map_location=DEVICE))
        model.eval()

        files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
        
        scenarios = ['m2m', 'h2h', 'rtt_m2h', 'rtt_h2m']

        plot_statistical_sweep(model, files)

        for scenario in scenarios:
            plot_shaded_with_risk(model, files, scenario)
            
        print(f"Final analysis complete. Check {SAVE_DIR} for results.")



    except KeyboardInterrupt:
        pass
