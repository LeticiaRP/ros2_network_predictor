import os
import glob
import logging
import pandas as pd 
import numpy as np 
from tqdm import tqdm # progress bar 


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


DATA_FOLDER = "../../data/raw"      # folder with the raw data 

REQUIRED_DATA = ["seq", "latency_ns", "tx", "rx"]     # csv structure for train the network

W = 50                              # analyzed window (W)

FREQ_MAX = max(FREQUENCY_MAP.values())     # max analyzed frequency 




logger = logging.getLogger("[TCN TRAIN]")
logging.basicConfig(level=logging.DEBUG)

#TODO: improve error handler 
def parse_matadata(filename): 
    """
    Extracts QoS, topology, and frequency from the filename

    Argument: 
        - filename: file's path 

    Returns: 
        - tuple: (topology, qos, freq_hz)
    """

    measure_type = None 
    scenario = None
    qos_policy = None
    frequency = None

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
        qos = QOS.get(qos_policy)
        freq_hz = FREQUENCY_MAP.get(frequency)
        topology = TOPOLOGY_MAP.get(topology_key)

        return qos, freq_hz, topology

    
    except Exception as e: 
        
        print(f"Error {e}")
        pass 


def load_folder_path(folder_path): 
    """
    Iteracts throught all CSVs in a folder, extracts the data and returns a list of numpy arrays

    Arguments: 
        - folder_path: folder that contains the data for train, test and validation of the tcn
    
    Returns: 
        - 
    """

    try: 

        all_files = glob.glob(os.path.join(folder_path, "*.csv"))

        if not all_files: 
            raise ValueError(f"Could not find CSV files in {folder_path}!")

        logger.info(f"Found {len(all_files)} in the {folder_path} folder")

        for filepath in tqdm(all_files): 

            qos, freq_hz, topolocy = parse_matadata(filepath)

            
            
    except Exception as e: 
        print(f"Error {e}")
        pass 



    def extract_features(filepath, qos, freq_hz, topology):

        try:

            df = pd.read_csv(filepath)


            # ---------- verify if the csv data is valid  ---------
            
            for col in REQUIRED_DATA: 
                if col not in df.coluns: 
                    raise ValueError(f"Data frame must contain {col}! The available columns are {df.columns.tolist()}")
                

            # ------------ extract data from the csv --------------

            seq_raw = pd.to_numeric(df[REQUIRED_DATA[0]], errors="coerce")
            lat_raw = pd.to_numeric(df[REQUIRED_DATA[1]], errors="coerce")
            tx_raw = pd.to_numeric(df[REQUIRED_DATA[2]], errors="coerce")
            rx_raw = pd.to_numeric(df[REQUIRED_DATA[3]], errors="coerce")


            # ------------ calculate the metric(s -----------------
        
            # -- inter-arrival time (a(t))
            a_t = rx_raw - rx_raw.shift(1)

            # -- sequence delta
            delta_seq = seq_raw - seq_raw.shift(1)   #TODO: ve se precisa normalizar 

            # -- deadline 
            deadline = 1/freq_hz

            # -- normalized frequence 
            freq_norm = freq_hz/FREQ_MAX

            # -- qos 
            q_col = np.full(len(df), qos)

            # -- topology 
            c_col = np.full(len(df), topology)


            # ----------- build feature DataFrame 
            features = pd.DataFrame({
                "l_t": lat_raw,             # latency 
                "a_t": a_t,                 # Inter-arrival time
                "delta_seq": delta_seq,     # Packet loss 
                "f_norm": freq_norm,        # Normalized frequency 
                "q_t": q_col,               # QoS policy 
                "c_t": c_col                # Scenario Topology 
            })


        except Exception as e: 
            logger.error(f"Error {e}")
            pass



if __name__ == '__main__': 

    load_folder_path(DATA_FOLDER)


