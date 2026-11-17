import os
import glob
import tqdm
import torch
import logging
import numpy as np

from data_pipeline import metadata_from_filename, load_raw_data, compute_features, compute_sliding_window
from data_pipeline import WINDOW_SIZE, HORIZON_MAX
from model_tcn import TCN, N_INPUTS, N_HORIZONS, N_QUANTILES

logger = logging.getLogger("[EVALUATION]")
logging.basicConfig(level=logging.DEBUG)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_interface(model, test_dir): 
    """
    Loads test CSV files, extracts features, run model interface across sliding windows, 
    and collects predictions alongside ground truth values 

    Arguments: 
        - model: Training PyTorch model instance set of latency forecasting
        - test_dir: Path to the directory containing test .csv files 

    Returns: 
        - list[dict]: List of dictionaries per processed file containing file metadata, 
        model predictions, real latency values and computed features 
    """

    # locate all csv files in the target directory
    all_files = glob.glob(os.path.join(test_dir, "*.csv"))

    if not all_files: 
        raise ValueError(f"Could not find CSV files in {test_dir}!")
    logger.info(f"Found {len(all_files)} files in {test_dir}")


    # switch model to evaluation mode 
    model.eval()
    results = []


    # disable gradient tracking for faster inference
    with torch.no_grad(): 
        for filepath in tqdm.tqdm(all_files, desc="Running inference"): 
            try: 
                # extract network parameters encoded in each file 
                qos, freq_hz, topology = metadata_from_filename(filename=filepath)
    
                # load raw data
                msg_seq_raw, msg_tx_raw, msg_rx_raw = load_raw_data(filepath=filepath)
    
                # compute engineered features
                features = compute_features(msg_seq=msg_seq_raw, msg_rx=msg_rx_raw, msg_tx=msg_tx_raw, 
                                            qos=qos, topology=topology, freq=freq_hz)
    
                # segment features into fixed-lenght sliding windows for time-series modeling
                X_windows = compute_sliding_window(features=features, window_size=WINDOW_SIZE)

                # trim the trailing windows that lack future ground truth up to HORIZON_MAX
                X_valid = X_windows[:-HORIZON_MAX]

                # convert valid window to a float32 Pytorch tensor and move to target device (GPU/CPU)
                X_tensor = torch.tensor(X_valid, dtype=torch.float32).to(DEVICE)

                # predictions 
                preds = model(X_tensor).cpu().numpy()

                # ground truth - real latency data 
                latency_real = features["l_t"].values 


                # store the predictions 
                results.append({
                    "filepath": filepath,
                    "freq_hz": freq_hz,
                    "topology": topology,
                    "preds": preds,                  # Shape: (N, 4, 3)
                    "latency_real": latency_real,    # Shape: (N_total,)
                    "features": features
                })
            except Exception as e:
                logging.error(f"Error processing file {filepath}: {e}")


    return results



def compute_deadline(fred_hz): 
    """
    Eq. 4: Application deadline

    Computes the application latency deadline from the transmission frequency 
    
    Arguments: 
        - freq_hz: Packet transmission frequency in Hz
    
    Returns: 
        - float: Maximum allowable latency deadline in ms
    """

    return 1000.0/fred_hz




def compute_safety_threshold(deadline): 
    """

    Eq. 11: Safety threshold 

    Computes the safe operational latency threshold as 80% of the application deadline 

    Arguments: 
        - deadline: Maximum allowable latency deadline in ms 
    
    Returns: 
        - float: Safety threshold (tau) in ms
    """

    return 0.8 * deadline




def compute_failure_threshold(deadline): 
    """
    Eq. 12: Failure threshold

    Computes the critical latency failure thrshold equal to 100% of the application deadline 

    Arguments: 
        - dealine: Maximum allowable latency deadline in ms 
    
    Returns: 
        - float: Failure threshold (F) in ms 
    """

    return deadline


def compute_effective_frequency(a_t_ms, W): 
    """
    Eq. 13: Effective publishing rate 

    Computes the effective publishing frequency in Hz over a sliding window of W messages

    Arguments: 
        - a_t_ms: Array of message inter-arrival times in ms
        - W: Sliding window size 
    
    Returns: 
        - float: Effective publishing frequency in Hz
    """

    try: 
        if len(a_t_ms) == 0: 
            raise ValueError(f"The a_t_ms vector is empty!")

        # select the window
        window_ms = a_t_ms[-W:]
        window_size = len(window_ms)


        elapsed_time = (np.sum(a_t_ms)) / 1000.0        # conversion from ms to s

        if elapsed_time <=0: 
            raise ValueError(f"Invalid value of elapsed time -> {elapsed_time}")

        return window_size/elapsed_time


    except Exception as e: 
        raise ValueError(f"Error processing the effective frequency: {e}")






def compute_frequency_risk (f_nom, f_eff, tau_f=0.1):
    """
    Eq. 14: Frequency risk 

    Computes the normalized frequency risk score r_F in [0,1] based on the effective publishing rate deficit 

    Arguments: 
        - f_nom: Nominal publishing frequency in Hz
        - f_eff: Measured effective publishing frequency in Hz 
        - tau_f: Relative tolerance band below nominal frequency 
    
    Returns: 
        - np.array: Normalized latenc risk values clipped between 0.0 (safe) and 1.0 (failure)
    """

    try: 

        if f_nom <= 0.0 or f_eff<=0.0: 
            raise ValueError(f"Invalid frequency value! f_nom: {f_nom} | f_eff: {f_eff}")

        # normalize frequency drop relative to the allowable tolerance band 
        r_F = (f_nom - f_eff) / (tau_f * f_nom)

        return np.clip(r_F, 0.0, 1.0)

    
    except Exception as e:
        raise ValueError(f"Error processing the frequency risk: {e}")

    



def compute_latency_risk(latency_p95_preds, tau, F): 
    """
    Eq. 16: Latency risk 

    Computes the normalized latency risk score r_L in [0,1] based on the P95 latency prediction 

    Arguments: 
        - latency_p95_preds: Array of scalar of 95th percentile (P95) latency prediction in ms 
        - tau: Safe operational latency threshold in ms
        - F: Critical latency failure threshold in ms
    
    Returns: 
        - np.array: Normalized latency risk values clipped between 0.0 (safe) and 1.0 (failure)
    """

    try: 
        if (F - tau) == 0: 
            raise ValueError(f"Invalid division by 0!")

        # linear interpolation between tau and F 
        ri = (latency_p95_preds - tau) / (F - tau)

        return np.clip(ri, 0.0, 1.0)

    except Exception as e: 
        raise ValueError(f"Error processing the latency frequency: {e}")



def main(): 
    """
    Loads trained TCN weights, runs test set inference and performs a sanity check on output 
    shapes and sample predictions  
    """

    TEST_DIR = "../../data/test"
    MODEL_PATH = "saved_models/best_tcn_model.pth"


    # initalize model architecure and load trained weights onto target device 
    model = TCN(
        input_dim=N_INPUTS, 
        num_horizons= N_HORIZONS, 
        num_quantiles= N_QUANTILES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # execute the inference
    logger.info("Starting evaluation pipeline .... ")
    test_results = run_interface(model, TEST_DIR)

    # 3. Sanity check: inspect output tensor shapes and sample median (P50) predictions
    if test_results:
        sample = test_results[0]
        print("\n--- Sample Prediction Check ---")
        print(f"File: {os.path.basename(sample['filepath'])}")
        print(f"Predictions shape: {sample['preds'].shape} (Expected: [N, 4, 3])")
        # Index [0, :, 1] -> 1st sliding window, all 4 horizons, 2nd quantile (P50 median)
        print(f"Sample P50 predictions (median): {sample['preds'][0, :, 1]}")
        print(f"Sample real latency sequence (first 5): {sample['latency_real'][:5]}")
    else:
        logger.error("No results generated. Check test directory.")


if __name__=='__main__': 
    main()