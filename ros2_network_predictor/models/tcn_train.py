import os
import glob
import torch
import torch.nn as nn
import logging
import pandas as pd 
import numpy as np 
from tqdm import tqdm # progress bar 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.parametrizations import weight_norm


TOPOLOGY_MAP = {
    "h2h" : 0.0,
    "m2m" : 0.2,
    "h2m" : 0.3, 
    "m2h" : 0.5,  
    "rtt_h2m" : 0.4, 
    "rtt_m2h" : 0.6, 
}

FREQUENCY_MAP = {
    "5.0Hz": 5, 
    "10.0Hz": 10,
    "20.0hZ": 20, 
    "50.0Hz": 50, 
    "100.0Hz": 100,
    "200.0Hz": 200, 
}

QOS = {
    "best_effort":0.0, 
    "reliable":1.0,
}

# ---------- Message-observable variables (raw signal) ---------------------------
DATA_FOLDER = "../../data/raw"          # folder with the raw data 
REQUIRED_DATA = ["seq", "tx", "rx"]     # csv structure for train the network
FREQ_MAX = max(FREQUENCY_MAP.values())     # max analyzed frequency 


# --------- Configuration ---------------------------
WINDOW_SIZE = 50                       # analyzed window (W)



# -------- Horizons (model's output) ------------------
HORIZONS = [1, 5, 10, 20]
HORIZON_MAX = max(HORIZONS)



# -------- Quantiles (also model's output) --------------------------------
QUANTILES = [0.05, 0.50, 0.95]


# ---------- TCN training parameters -------------------
N_INPUTS = 6
N_HORIZONS = len(HORIZONS)
N_QUANTILES = len(QUANTILES)
N_OUTPUTS = N_HORIZONS * N_QUANTILES
N_CHANNELS = [32, 64, 128, 64]
KERNEL_SIZE = 3
DROPOUT = 0.1 
STRIDE = 1

# dilatation: 1, 2, 4, 8
# padding = 2, 4, 8, 16


# ---------- Loss functions
BETA = 0.5 


logger = logging.getLogger("[TCN TRAIN]")
logging.basicConfig(level=logging.DEBUG)

#TODO: improve error handler 
def metadata_from_filename(filename): 
    """
    Extracts QoS, topology, and frequency from the filename

    Argument: 
        - filename: file's path 

    Returns: 
        - tuple: (topology, qos, freq_hz)
    """

    basename = os.path.basename(filename)
    parts = basename.split("_")


    measure_type = parts[0]      # latency (one-way-path) / rtt /(round time trip)
    scenario = parts[1]


    try: 

        # ------------------- identify topology ------------------------
        if measure_type == "rtt": 
            topology_key = f"rtt_{scenario}"
        elif measure_type == "latency": 
            topology_key = scenario
        else: 
            raise ValueError(f"Unknown measure_type '{measure_type}' in  {basename}")



        # -------------------- identify qos and frequency --------------------
        if parts[2] == "best" and parts[3] == "effort": 
            qos_policy = "best_effort"
            frequency = parts[4]

        elif parts[2] == "reliable": 
            qos_policy = parts[2]
            frequency = parts[3]
        
        else: 
            raise ValueError(f"Unknown qos policy {parts[2]} in {basename}")



        # -------------- get value from the dictionary ---------------------
        qos = float(QOS.get(qos_policy))
        freq_hz = float(FREQUENCY_MAP.get(frequency))
        topology = float(TOPOLOGY_MAP.get(topology_key))


        # ---------- debug -------------
        logger.debug(f" File {basename} is: Qos: {qos} | Freq: {freq_hz} | Topology: {topology}\n")



        return qos, freq_hz, topology

    
    except Exception as e: 
        
        print(f"Error {e}")
        raise 




def main(folder_path): 
    """
    Iteracts throught all CSVs in a folder, extracts the data and returns a list of numpy arrays

    Arguments: 
        - folder_path: folder that contains the data for train, test and validation of the tcn
    
    Returns: 
        - 
    """

    try: 

        # -------- get all the csv files for a gien folder 
        all_files = glob.glob(os.path.join(folder_path, "*.csv"))

        if not all_files: 
            raise ValueError(f"Could not find CSV files in {folder_path}!")

        logger.info(f"Found {len(all_files)} in the {folder_path} folder")

        # -------- process the data of each file presented at folder 
        for filepath in tqdm(all_files): 

            qos, freq_hz, topology = metadata_from_filename(filepath)

            msg_seq_raw, msg_tx_raw, msg_rx_raw = load_raw_data(filepath)

            features = compute_features(msg_seq_raw, msg_rx_raw, msg_tx_raw, qos, topology, freq_hz)

            X_windows = compute_sliding_window(features, window_size=WINDOW_SIZE)

            latency_array = features["l_t"].values

            dataset = MultiHorizonDataset(X_windows=X_windows,latency_series=latency_array)

            logging.debug(dataset)


            
            
    except Exception as e: 
        print(f"Error {e}")
        raise 



def load_raw_data(filepath):

    """
    Returns raw data from a given CSV file

    Arguments: 
        - filepath: filepath for a given CSV 
    
    Returns: 
        - seq_raw: Message seq ID
        - tx_raw: ROS 2 timestamp from when the transmission happened  
        - rx_raw: ROS 2 timestamp from when the reception happened 

    """

    try:


        df = pd.read_csv(filepath)

        # ---------- verify if the csv data is valid  ---------
        
        for col in REQUIRED_DATA: 
            if col not in df.columns: 
                raise ValueError(f"Missing column '{col}' in {filepath}! Available: {df.columns.tolist()}")
            

        # ------------ extract data from the csv --------------

        msg_seq_raw = pd.to_numeric(df[REQUIRED_DATA[0]], errors="coerce")
        msg_tx_raw = pd.to_numeric(df[REQUIRED_DATA[1]], errors="coerce")
        msg_rx_raw = pd.to_numeric(df[REQUIRED_DATA[2]], errors="coerce")

        return msg_seq_raw, msg_tx_raw, msg_rx_raw


    
    except Exception as e: 
        logger.error(f"Error {e}")
        raise




def compute_features(msg_seq, msg_rx, msg_tx, qos, topology, freq):
    """
    Returns the features for trainning the TCN

    Arguments: 
        - msg_seq: raw data of the messages seq id
        - msg_rx: raw data of the message reception time 
        - msg_tx: raw data of the message transmission time 
        - qos: id for the selected qos policy 
        - topology: id for the selected topology 
        - freq: raw value of the selected frequency 

    
    Return: 
        - features: structure of data to train the neural network
    
    """


    try: 

        # ------------ calculate the metric(s -----------------
        # -- latency 
        l_t = msg_rx - msg_tx


        # -- inter-arrival time (a(t))
        a_t = msg_rx - msg_rx.shift(1)
        a_t[0] = msg_rx[0]

        # -- sequence delta
        max_seq_seen = -1.0
        delta_sigma = []

        for seq in msg_seq: 

            if max_seq_seen == -1: 
                delta = 0
                max_seq_seen = seq

            elif seq > max_seq_seen: 
                gap = seq - max_seq_seen
                delta = gap - 1 
                max_seq_seen = seq

            else: 
                delta = 1

            delta_sigma.append(delta)

            
        # -- normalized frequence 
        freq_norm = freq/FREQ_MAX

        # -- qos 
        q_col = np.full(len(msg_seq), qos)

        # -- topology 
        c_col = np.full(len(msg_seq), topology)


        # ----------- build feature DataFrame 
        features = pd.DataFrame({
            "l_t": l_t,             # latency 
            "a_t": a_t,                 # Inter-arrival time
            "delta_seq": delta_sigma,     # Packet loss 
            "f_norm": freq_norm,        # Normalized frequency 
            "q_t": q_col,               # QoS policy 
            "c_t": c_col                # Scenario Topology 
        })


        # --------- debug 

        logging.debug(f"Features: \n{features}\n")


        if features.isna().values.any(): 
            logger.warning(f"The DataFrame has a NaN value at {features.isna().any()}")

        elif features is None: 
            raise ValueError(f"The DataFrame is empty!")

        return features

    except Exception as e: 
        logger.error(f"Error: {e}")
        raise




def compute_sliding_window(features, window_size):

    """
    Transform the matrix of features (N,6) in sliding window 

    Arguments: 
        - features: matrix of data
        - window_size: value of the window size 
    
    Returns: 
        - X: sliding window
    """

    try: 
        data = features.values

        windows = np.lib.stride_tricks.sliding_window_view(
                data, 
                window_shape=window_size, 
                axis=0
            )

        X = windows.transpose(0,2,1) # transpose to (sample, feature, tempo)


        # --------- debug ---------
        logger.debug(X)

        return X

    except Exception as e:
        logger.error(f"Error {e}")




class MultiHorizonDataset(Dataset):

    def __init__(self, X_windows, latency_series, horizons = HORIZONS, max_horizon = HORIZON_MAX):
        """
        Creates a Pytorch Dataset that pairs sliding windows of feature (X) with their corresponding future latency targets (Y) across multiple horizons 

        Arguments: 
            - X_windows: matrix of sliding windows (N_samples, window_size, n_features)
            - latency_series: 1D array containing the raw latency values 
            - horizons: list of step-ahead forecast horizons 
            - max_horizons: maximum value in horizons, used to discard windows that don't have a valid future target 


        Returns:
            - X: tensor of shape (N_samples, window_size, n_features)
            - Y: tensor of shape (N_samples, len(horizons)) representing the true latency at t+h for each horizon h

        """

        self.horizons = horizons
        self.max_horizon = max_horizon

        
        # ----------- create the tensor for X -----------------
        X_windows = X_windows[:-max_horizon]
        self.X = torch.tensor(X_windows, dtype=torch.float32)
        

        # ---------- create the tensor for Y ------------------
        self.Y = []

        for i in range(len(self.X)): 

            window_end_idx = i + 49         # totalizing 50 samples 

            target = [latency_series[window_end_idx+h] for h in horizons]
            self.Y.append(target)


        self.Y = torch.tensor(self.Y, dtype=torch.float32)  



    def __len__(self):
        """
        Returns the total number of valid samples (sliding windows) in the dataset 
    
        Returns: 
            - int: number of samples available (size of the X tensor)
        """
        return len(self.X)



    def __getitem__(self, index):
        """
        Retrieves a single sample and its corresponding target at the given index

        Arguments: 
            - index: index of the sample to retrieve

        Returns: 
            - tuple: (X_sample, Y_target) where:
                - X_sample: tensor of shape (window_size, 6) with the historical features
                - Y_target: tensor of shape (len(horizons),) with the future latency values
        """
        return self.X[index], self.Y[index]




class Chomp1d(nn.Module):
    """
    Removes trailing elements (padding) from the end of a 1D temporal sequence to maintain
    casuality in convulutional networks. 
    """


    def __init__(self, chomp_size): 
        """
        Initializes the Chomp1d module

        Arguments: 
            - chomp_size: int, number of elements to remove from the end of the sequence
        """

        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size



    def forward(self, x): 
        """
        Slices off the trailing padding along the sequence dimension 

        Arguments: 
            - x: tensor of shape (batch_size, channels, seq_len) representing the padded sequence

        Returns: 
            - tensor: contiguous tensor of shape (batch size, channels, seq_len - chomp_size) 
        """

        return x[:, :, : - self.chomp_size].contiguous()
    




class TemporalBlock(nn.Module): 
    """
    Residual building block for a Temporal Convolutional Network (TCN) consisting of two dilated 
    causal convolutions and a skip connection 
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, 
                 stride, dropout, dilation, padding):
        """
        Initializes the TemporalBlock with two causal convolution layers and optimal 1x1 
        convolution for matching residual dimension

        Arguments: 
            - n_inputs: number of input channels
            - n_outputs: number of output channels 
            - kernel_size: size of the convolving kernel 
            - stride: stride of the convolution
            - dilation: spacing between kernel elements
            - padding: zero-padding added to both sides of the input before chomping 
            - dropout: dropout probability
        """

        super(TemporalBlock, self).__init__()

        # Passes the input through two identical causal convolution stages -------------

        self.conv1 = weight_norm(                                               # Applies a dilated 1D convolution with weight normalization 
            nn.Conv1d(                                                          # which decouples the lenght of weight vectors from their direction to         
                n_inputs, n_outputs, kernel_size,                               # stabilize and speed um traning 
                stride=stride, padding=padding, dilation=dilation               
            )
        )
        self.chomp1 = Chomp1d(padding)                                          # Slices off the extra right-side padding added by Conv1d so future
                                                                                # time steps never leak into the past 
        
        self.relu1 = nn.ReLU()                                                  # Adds non-linearity
        self.dropout1 = nn.Dropout(dropout)                                     # Adds regularization 


        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size, 
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)



        self.net = nn.Sequential(                                               
            self.conv1, 
            self.chomp1, 
            self.relu1, 
            self.dropout1, 

            self.conv2, 
            self.chomp2, 
            self.relu2, 
            self.dropout2
        )


        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()


    def forward(self, x): 
        """
        Passes the input through the dilated causal convolutions and adds the residual connection

        Arguments: 
            - x: torch. Tensor of shape (batch_size, n_inputs, seq_len)
        
        Returns
            - torch.Tensor: output tensor of shape (batch_size, n_outputs, seq_len)
        """

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)




class TCN(nn.Module): 
    """
    Temporal Convolutional Network (TCN) for multi-horizon probabilistic time-series 
    forecast using quantile regression 
    """

    def __init__(self, input_dim = N_INPUTS, num_horizons = N_HORIZONS, num_quantiles = N_QUANTILES, 
                 num_channels = N_CHANNELS, kernel_size = KERNEL_SIZE, dropout = DROPOUT, stride = STRIDE): 
        """
        Initializes the TCN with exponentially dilated TemporalBlocks and a multi-quantile projection ahead 

        Arguments: 
            - input_dim = number of input features per time step 
            - num_horizons = number of future time steps to forecast 
            - num_quantiles = number of quantiles predicted per horizons 
            - num_channels = hidden channel sizes for each TCN level
            - kernel_size = size of convolving kernel in each TemporalBlock 
            - dropout = dropout probability inside each TemporalBlock 
        """

        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i 
            in_channels = input_dim if i==0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    n_inputs = in_channels, 
                    n_outputs = out_channels, 
                    kernel_size = kernel_size, 
                    stride = stride, 
                    dilation = dilation_size, 
                    padding = (kernel_size - 1) * dilation_size, 
                    dropout = dropout 
                )
            )

        self.network = nn.Sequential(*layers)

        self.num_horizons = num_horizons
        self.num_quantiles = num_quantiles
        self.linear = nn.Linear(num_channels[-1], self.num_horizons * self.num_quantiles)



    def forward(self, x): 
        """
        Generates multi-horizon quantile forecasts from an input historical window 

        Arguments: 
            - x: tensor of shape (batch_size, seq_len, input_dim) containing historical features 

        Returns: 
            - torch.Tensor: forecasts tensor of shape (batch_size, num_horizons, num_quantiles)
        """

        # rearrange to 1D convulations 
        x = x.permute(0, 2, 1)
        y = self.network(x)

        # extract the final causal time step (batch_size, num_channels[-1]) and project
        out = self.linear(y[:, :, -1])

        # reshape to (batch_size, num_horizons, num_quantiles)
        return out.view(out.shape[0], self.num_horizons, self.num_quantiles)





class LossFunction(nn.Module):
    """
    Horizon-weighted Pinball (Quantile) Loss for multi-horizon probabilistic forecasting. 
    Penalizes prediction errors asymmetrically per quantile and scales each horizon by inverse power weight h^(-beta)
    """


    def __init__(self, quantiles, horizons, beta): 
        """
        Initializes the Loss Function module and precomputes normalized horizon weights

        Arguments: 
            - quantiles: target quantile
            - horizons: forecast step offsets used to compute weights
            - beta: decay exponent controlling how strongly distant horizons are down-weightd 
        """

        super().__init__()

        self.quantiles = quantiles

        # compute inverse power weights w_h = h^(-beta) and normalize to sum to len(horizons)
        weights = torch.tensor([h ** (-beta) for h in horizons ], dtype=torch.float32)
        weights = weights / weights.sum() * len(horizons)
        self.register_buffer("horizon_weights", weights)




    def forward(self, preds, target): 
        """
        Computes the horizon-weighted quantile loss between predicted quantiles and ground truth

        Arguents: 
            - preds: torch.Tensor of shape
            - target: torch.Tensor of shape
        
        Return: 
            - torch.Tensor: scalar loss avareged across batch and horizons, summed over quantiles
        """

        loss = 0.0 
        target = target.unsqueeze(-1)

        for i, q in enumerate(self.quantiles): 
            preds_q = preds[:, :, i]    
            errors = target - preds_q.unsqueeze(-1)

            # assymetric pinball loss 
            pinball = torch.max((q-1)*errors, q*errors).squeeze(-1)

            # weight by horizon importance and average across batch and horizons
            loss += (pinball * self.horizon_weights).mean()

        return loss 




    

if __name__ == '__main__': 

    main(DATA_FOLDER)


