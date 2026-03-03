import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add this to your existing imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import ROS2DataLoader
from preprocessor import ROS2Preprocessor

# Configuration (same as before)
INPUT_LEN = 50
OUTPUT_LEN = 1
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTILES = [0.05, 0.5, 0.95]
DATA_PATH = '../../data/raw'

# Topology mapping (same as before)
PLAT_MAP = {
    'h2h': 0.0,      'rtt_h2m': 0.3,
    'h2m': 0.2,      'rtt_m2h': 0.5,
    'm2h': 0.4,      
    'm2m': 0.6,      
}

print(f"Using device: {DEVICE}")



# TCN model definitions --------------------------------------------------------------------------

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





# data loading ------------------------------------------------------------------------------------------------
print("1. Auto-Discovering Datasets...")

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
    is_rtt = 'rtt' in basename
    topo = parts[1]
    
    if is_rtt:
        plat_key = f"rtt_{topo}"
    else:
        plat_key = topo
    
    plat_val = PLAT_MAP.get(plat_key, 0.0)
    qos = 1.0 if 'reliable' in basename else 0.0
    
    freq_val = 0.3
    for p in parts:
        if 'Hz' in p:
            try:
                f_str = p.replace('Hz', '')
                freq_val = float(f_str) / 200.0
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
        target_col = next((c for c in df.columns if ('latency' in c or 'rtt' in c) and 'ns' in c), None)
        
        if target_col is None: continue 
        
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col])
        df = df.ffill().fillna(0)
        
        lat_ms = df[target_col].values.reshape(-1, 1).astype(np.float32) / 1e6
        
        qos, plat, freq = parse_filename(f)
        
        c_qos = np.full((len(lat_ms), 1), qos, dtype=np.float32)
        c_plat = np.full((len(lat_ms), 1), plat, dtype=np.float32)
        c_freq = np.full((len(lat_ms), 1), freq, dtype=np.float32)
        
        combined_data = np.hstack([lat_ms, c_qos, c_plat, c_freq])
        
        X_chunk, y_chunk = create_windows(combined_data, INPUT_LEN, OUTPUT_LEN, step=10)
        
        if len(X_chunk) > 0:
            all_X.append(X_chunk)
            all_y.append(y_chunk)
            
    except Exception as e:
        tqdm.write(f"Error parsing {os.path.basename(f)}: {e}")

if not all_X:
    print("No valid data loaded.")
    exit()

print(f"Successfully loaded {len(all_X)} experiment chunks.")

# merge and prepare data
X_train = torch.cat(all_X)
y_train = torch.cat(all_y)


# shuffle
idx = torch.randperm(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]


# split
val_split = int(len(X_train) * 0.9)
X_val, y_val = X_train[val_split:], y_train[val_split:]
X_train, y_train = X_train[:val_split], y_train[:val_split]





# alternative models ---------------------------------------------------------

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



# LSTM Model
class QuantileLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_len=1, num_quantiles=3, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_len * num_quantiles)
        self.num_quantiles = num_quantiles
        self.output_len = output_len
        

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Take the last output
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return out.view(out.shape[0], self.output_len, self.num_quantiles)




# CNN-only Model
class QuantileCNN(nn.Module):
    def __init__(self, input_dim, output_len=1, num_quantiles=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, output_len * num_quantiles)
        self.dropout = nn.Dropout(0.2)
        self.num_quantiles = num_quantiles
        self.output_len = output_len
        

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> permute for conv1d
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # (batch, channels)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out.view(out.shape[0], self.output_len, self.num_quantiles)




# CNN-GRU Model
class QuantileCNNGRU(nn.Module):
    def __init__(self, input_dim, output_len=1, num_quantiles=3):
        super().__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # GRU layer
        self.gru = nn.GRU(64, 128, batch_first=True, dropout=0.2, num_layers=2)
        
        # Output layers
        self.fc = nn.Linear(128, output_len * num_quantiles)
        self.dropout = nn.Dropout(0.2)
        self.num_quantiles = num_quantiles
        self.output_len = output_len
        


    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        
        # Back to (batch, seq_len, features) for GRU
        x = x.permute(0, 2, 1)
        
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return out.view(out.shape[0], self.output_len, self.num_quantiles)






# Res-CNN-LSTM Model (Residual CNN + LSTM)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class QuantileResCNN_LSTM(nn.Module):
    def __init__(self, input_dim, output_len=1, num_quantiles=3):
        super().__init__()
        # Initial CNN
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        
        # LSTM
        self.lstm = nn.LSTM(64, 128, batch_first=True, dropout=0.2, num_layers=2)
        
        # Output
        self.fc = nn.Linear(128, output_len * num_quantiles)
        self.dropout = nn.Dropout(0.2)
        self.num_quantiles = num_quantiles
        self.output_len = output_len
        

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Apply residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Back to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        out = self.fc(last_out)
        return out.view(out.shape[0], self.output_len, self.num_quantiles)




# 5. XGBoost Wrapper (for compatibility)
class XGBoostQuantile:
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        self.quantiles = quantiles
        self.models = {}
        

    def fit(self, X_train, y_train, X_val, y_val):
        # Reshape data for XGBoost: (samples, features)
        # For time series, we flatten the sequence
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        y_train_flat = y_train.squeeze()
        y_val_flat = y_val.squeeze()
        

        for q in self.quantiles:
            print(f"Training XGBoost for quantile {q}...")
            self.models[q] = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=q,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            self.models[q].fit(
                X_train_flat, y_train_flat,
                eval_set=[(X_val_flat, y_val_flat)],
                verbose=False
            )
    

    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1)
        predictions = []
        for q in self.quantiles:
            pred = self.models[q].predict(X_flat)
            predictions.append(pred)
        return np.stack(predictions, axis=-1)





# Training function ------------------------------------------------------------------------------------------------------------

def train_model(model, train_loader, val_data, epochs=EPOCHS, lr=LR, model_name="model"):
    """Generic training function for PyTorch models"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = QuantileLoss(QUANTILES)
    
    X_val, y_val = val_data
    X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
    
    train_losses = []
    val_losses = []
    
    print(f"\nTraining {model_name}...")
    for epoch in range(epochs):
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
        
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()
            val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}")
    
    return train_losses, val_losses



def evaluate_model(model, X_test, y_test, model_name="model"):
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test.to(DEVICE)
        y_test_tensor = y_test.to(DEVICE)
        
        preds = model(X_test_tensor)
        
        # Calculate metrics for each quantile
        results = {}
        for i, q in enumerate(QUANTILES):
            pred_q = preds[:, :, i].cpu().numpy().flatten()
            true_q = y_test_tensor[:, :, 0].cpu().numpy().flatten()
            
            mae = mean_absolute_error(true_q, pred_q)
            rmse = np.sqrt(mean_squared_error(true_q, pred_q))
            
            # Pinball loss
            errors = true_q - pred_q
            pinball = np.mean(np.maximum((q-1) * errors, q * errors))
            
            results[f'q{q}_mae'] = mae
            results[f'q{q}_rmse'] = rmse
            results[f'q{q}_pinball'] = pinball
        
        return results



# data loaders ------------------------------------------------------------------------------------------------

train_loader = DataLoader(
    TensorDataset(X_train, y_train), 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=0,  # Set to 0 for CPU to avoid multiprocessing issues
    pin_memory=True if DEVICE == "cuda" else False
)

val_data = (X_val, y_val)




# train models ---------------------------------------------------------------------------------------------------

models_to_train = {
    'LSTM': QuantileLSTM(input_dim=4, hidden_dim=128, num_layers=2).to(DEVICE),
    'CNN-only': QuantileCNN(input_dim=4).to(DEVICE),
    'CNN-GRU': QuantileCNNGRU(input_dim=4).to(DEVICE),
    'Res-CNN-LSTM': QuantileResCNN_LSTM(input_dim=4).to(DEVICE),
    'TCN': UniversalQuantileTCN(input_dim=4, output_len=OUTPUT_LEN, num_quantiles=3).to(DEVICE)
}

results = {}

# Train PyTorch models
for model_name, model in models_to_train.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print('='*50)
    
    train_losses, val_losses = train_model(
        model, train_loader, val_data, 
        epochs=EPOCHS, lr=LR, 
        model_name=model_name
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_val, y_val, model_name)
    results[model_name] = metrics
    
    # Save model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), f"saved_models/{model_name.lower().replace('-', '_')}.pth")
    print(f"✓ {model_name} saved to saved_models/")

# Train XGBoost (special case)
print(f"\n{'='*50}")
print("Training XGBoost")
print('='*50)

xgb_model = XGBoostQuantile(quantiles=QUANTILES)
xgb_model.fit(
    X_train.numpy(), y_train.numpy(),
    X_val.numpy(), y_val.numpy()
)

# Evaluate XGBoost
xgb_preds = xgb_model.predict(X_val.numpy())
xgb_metrics = {}
for i, q in enumerate(QUANTILES):
    true_vals = y_val.numpy()[:, :, 0].flatten()
    pred_vals = xgb_preds[:, i]
    
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    errors = true_vals - pred_vals
    pinball = np.mean(np.maximum((q-1) * errors, q * errors))
    
    xgb_metrics[f'q{q}_mae'] = mae
    xgb_metrics[f'q{q}_rmse'] = rmse
    xgb_metrics[f'q{q}_pinball'] = pinball

results['XGBoost'] = xgb_metrics



# Comparison table -------------------------------------------------------------------------------------------------

print("\n" + "="*80)
print("FINAL COMPARISON RESULTS")
print("="*80)

# Create comparison table
comparison_data = []
for model_name, metrics in results.items():
    row = [
        model_name,
        f"{metrics.get('q0.05_mae', 0):.8f}",
        f"{metrics.get('q0.5_mae', 0):.8f}",
        f"{metrics.get('q0.95_mae', 0):.8f}",
        f"{metrics.get('q0.05_rmse', 0):.8f}",
        f"{metrics.get('q0.5_rmse', 0):.8f}",
        f"{metrics.get('q0.95_rmse', 0):.8f}",
        f"{metrics.get('q0.05_pinball', 0):.8f}",
        f"{metrics.get('q0.5_pinball', 0):.8f}",
        f"{metrics.get('q0.95_pinball', 0):.8f}"
    ]
    comparison_data.append(row)

# create DataFrame for better display
columns = ['Model', 'MAE_5%', 'MAE_50%', 'MAE_95%', 'RMSE_5%', 'RMSE_50%', 'RMSE_95%', 
           'Pinball_5%', 'Pinball_50%', 'Pinball_95%']
df_results = pd.DataFrame(comparison_data, columns=columns)


# Sort by median MAE (q0.5_mae) for comparison
df_results['MAE_50%'] = pd.to_numeric(df_results['MAE_50%'])
df_results = df_results.sort_values('MAE_50%')

print("\n" + df_results.to_string(index=False))



df_results.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Results saved to model_comparison_results.csv")


# visualization --------------------------------------------------------------------------------------------------------


# Plot comparison bar charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_to_plot = ['MAE_50%', 'RMSE_50%', 'Pinball_50%']
titles = ['Median MAE Comparison', 'Median RMSE Comparison', 'Median Pinball Loss']

for ax, metric, title in zip(axes, metrics_to_plot, titles):
    models = df_results['Model'].values
    values = pd.to_numeric(df_results[metric]).values
    
    bars = ax.bar(range(len(models)), values)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylabel('Error')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.5f}', ha='center', va='bottom', fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Comparison plot saved to model_comparison.png")