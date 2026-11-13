import os
import glob
import torch
import logging
import pandas as pd 
import numpy as np 
from tqdm import tqdm # progress bar 
from torch.utils.data import Dataset, DataLoader


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


DATA_FOLDER = "../../data/raw"          # folder with the raw data 

REQUIRED_DATA = ["seq", "tx", "rx"]     # csv structure for train the network

WINDOW_SIZE = 50                       # analyzed window (W)


# LIMITS
FREQ_MAX = max(FREQUENCY_MAP.values())     # max analyzed frequency 
DELTA_SIGMA_MAX = 10

HORIZONS = [1, 5, 10, 20]
HORIZON_MAX = max(HORIZONS)


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

            print(dataset)
            
            
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


if __name__ == '__main__': 

    main(DATA_FOLDER)


