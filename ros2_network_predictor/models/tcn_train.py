import os
import sys
import glob
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
from pathlib import Path


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import ROS2DataLoader
from preprocessor import ROS2Preprocessor



# configuration 
INPUT_LEN = 50
OUTPUT_LEN = 1
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTILES = [0.05, 0.5, 0.95]
DATA_PATH = '/home/leticia/ros2_ws/src/ros2_network_predictor/data/raw'




# topology mapping 
PLAT_MAP = {
    'h2h': 0.0,      'rtt_h2m': 0.3,
    'h2m': 0.2,      'rtt_m2h': 0.5,
    'm2h': 0.4,      
    'm2m': 0.6,      
    
}



print(f"Using device: {DEVICE}")





# model architecture 
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
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=0.1)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_len * num_quantiles)
        self.num_quantiles = num_quantiles


    def forward(self, x):
        x = x.permute(0, 2, 1) 
        y = self.network(x)
        out = self.linear(y[:, :, -1])
        return out.view(out.shape[0], -1, self.num_quantiles)




class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
    def forward(self, preds, target):
        loss = 0
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i].unsqueeze(-1)
            loss += torch.max((q-1) * errors, q * errors).mean()
        return loss
    






# getting dataset 
print("1. Auto-Discovering Datasets...")



# 1. FIND FILES (OUTSIDE LOOP)
files_lat = glob.glob(os.path.join(DATA_PATH, "latency_*.csv"))
files_rtt = glob.glob(os.path.join(DATA_PATH, "rtt_*.csv"))
files = files_lat + files_rtt



if len(files) == 0:
    print("No files found in ../data/raw. Check path.")
    exit()



all_X = []
all_y = []



def parse_filename(fname):

    basename = os.path.basename(fname)
    parts = basename.split('_')
    

    # check if RTT
    is_rtt = 'rtt' in basename
    

    # heuristics: Topology is usually index 1 (e.g., latency_h2m...)
    topo = parts[1]



    # build Key
    if is_rtt:
        plat_key = f"rtt_{topo}"
    else:
        plat_key = topo



    # map to float ID
    plat_val = PLAT_MAP.get(plat_key, 0.0)

    

    # QoS
    qos = 1.0 if 'reliable' in basename else 0.0
        


    # frequency
    freq_val = 0.3
    for p in parts:
        if 'Hz' in p:
            try:
                f_str = p.replace('Hz', '')
                freq_val = float(f_str) / 200.0 # normalize 200Hz -> 1.0
                break
            except: pass
            
    return qos, plat_val, freq_val







def create_windows(data, il, ol, step=10): 
    data_t = torch.tensor(data, dtype=torch.float32)
    
    X = [data_t[i:i+il] for i in range(0, len(data) - il - ol + 1, step)]
    y = [data_t[i+il:i+il+ol, 0:1] for i in range(0, len(data) - il - ol + 1, step)]

    return torch.stack(X), torch.stack(y)





for f in tqdm(files, desc="Parsing Files"):

    try:
        df = pd.read_csv(f)
        


        # column search
        target_col = next((c for c in df.columns if ('latency' in c or 'rtt' in c) and 'ns' in c), None)        
        


        if target_col is None: continue 
            


        
        # force numeric, turn garbage to NaN
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        # drop bad rows
        df = df.dropna(subset=[target_col])

        # fill missing
        df = df.ffill().fillna(0)
        


        # convert to milliseconds
        lat_ms = df[target_col].values.reshape(-1, 1).astype(np.float32) / 1e6
        


        # parse context
        qos, plat, freq = parse_filename(f)
        


        # create the feature tensors
        c_qos = np.full((len(lat_ms), 1), qos, dtype=np.float32)
        c_plat = np.full((len(lat_ms), 1), plat, dtype=np.float32)
        c_freq = np.full((len(lat_ms), 1), freq, dtype=np.float32)
        



        # combine: [latency, QoS, platform, freq]
        combined_data = np.hstack([lat_ms, c_qos, c_plat, c_freq])
        



        # windowing
        X_chunk, y_chunk = create_windows(combined_data, INPUT_LEN, OUTPUT_LEN, step=10)
        



        if len(X_chunk) > 0:
            all_X.append(X_chunk)
            all_y.append(y_chunk)
            

    except Exception as e:
        
        tqdm.write(f"Error parsing {os.path.basename(f)}: {e}")
    
    
    except KeyboardInterrupt:
        pass
    



if not all_X:
    print("No valid data loaded.")
    exit()



print(f"Successfully loaded {len(all_X)} experiment chunks.")




# merge
X_train = torch.cat(all_X)
y_train = torch.cat(all_y)



# shuffle
idx = torch.randperm(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]



# split
val_split = int(len(X_train) * 0.9)
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]




train_loader = DataLoader(
    TensorDataset(X_train, y_train), 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=12,          
    prefetch_factor=2,       
    persistent_workers=True, # Keep workers alive between epochs (don't kill/respawn)
    pin_memory=True          
)




 
# training
print("2. Initializing model...")

model = UniversalQuantileTCN(input_dim=4, output_len=OUTPUT_LEN, num_quantiles=3).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = QuantileLoss(QUANTILES)



print(" Starting training...")
try: 
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []


        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {np.mean(batch_losses):.5f}")


    # save the model 
    torch.save(model.state_dict(), "universal_tcn.pth")
    print("Model saved to universal_tcn.pth")

except KeyboardInterrupt:

    pass


