import os
import glob
import tqdm
import torch
import logging
import numpy as np
import pandas as pd

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
            raise ValueError(f"The a_t_ms array is empty!")

        # select the window
        window_ms = a_t_ms[-W:]
        window_size = len(window_ms)


        elapsed_time = (np.sum(a_t_ms)) / 1000.0        # conversion from ms to s

        if elapsed_time <=0: 
            raise ValueError(f"Invalid value of elapsed time -> {elapsed_time}")

        return window_size/elapsed_time


    except Exception as e: 
        raise ValueError(f"Error processing the effective frequency: {e}")



def packet_loss_rate(delta_sigma_array, W): 
    """
    Eq. 14: Measured packet loss rate 

    Computes the observed packet loss rate p(t) over a sliding window of W received packets using 
    sequence deltas (delta_sigma), treating out-of-order packets as lost. 

    Arguments: 
        - delta_sigma_array: Array of sequence delta for lost packets
        - W: sliding window size 

    Returns: 
        - p_loss: estimated packet loss rate 
    """

    try: 

        if len(delta_sigma_array) == 0: 
            raise ValueError(f"The delta_sigma array is empty!")

        # sequence number of the current packet (t) and the packet W steps earlier (t-W)
        sigma_t = delta_sigma_array[-1]
        sigma_t_w = delta_sigma_array[-W -1]

        # total expected packages 
        total_expected = sigma_t - sigma_t_w

        if total_expected <= 0: 
            raise ValueError(f"Total messages expected are invalid! The interval is 0 or negative!")

        total_losses = np.sum(delta_sigma_array)

        p_loss = total_losses / total_expected

        return p_loss

    except Exception as e: 
        raise ValueError(f"Error processing the packet loss rate: {e}")



def compute_frequency_risk(f_nom, f_eff, tau_f=0.1):
    """
    Eq. 15: Frequency risk 

    Computes the normalized frequency risk score r_F in [0,1] based on the effective publishing rate deficit 

    Arguments: 
        - f_nom: Nominal publishing frequency in Hz
        - f_eff: Measured effective publishing frequency in Hz 
        - tau_f: Relative tolerance band below nominal frequency 
    
    Returns: 
        - r_F: Normalized latenc risk values clipped between 0.0 (safe) and 1.0 (failure)
    """

    try: 

        if f_nom <= 0.0 or f_eff<=0.0: 
            raise ValueError(f"Invalid frequency value! f_nom: {f_nom} | f_eff: {f_eff}")

        # normalize frequency drop relative to the allowable tolerance band 
        r_F = (f_nom - f_eff) / (tau_f * f_nom)
        r_F = np.clip(r_F, 0.0, 1.0)

        return r_F

    
    except Exception as e:
        raise ValueError(f"Error processing the frequency risk: {e}")



def compute_packet_loss_risk(p_loss, p_fail):
    """
    Eq. 16: Packet loss risk

    Normalizes the mesured packet loss rate p(t) against the critical failure threshold p_fail 
    and clips the resulting r_P to the [0,1] interval.

    Arguments: 
        - p_loss: Measured packet loss rate
        - p_fail: Critical packet loss threshold representing maximum tolerable loss 

    Returns: 
        - r_P: Normalized packet-loss risk bounded within [0.0, 1.0]
    """

    try: 
        if p_fail <= 0: 
            raise ValueError(f"Invalid p_fail value!")

        # normalize observed loss rate relative to the failure threshold
        r_P = p_loss/p_fail
        r_P = np.clip(r_P, 0.0, 1.0)

        return r_P


    except Exception as e: 
        raise ValueError(f"Error processing the packet loss risk: {e}")



def compute_latency_risk(latency_p95_preds, tau, F): 
    """
    Eq. 17: Latency risk 

    Computes the normalized latency risk score r_L in [0,1] based on the P95 latency prediction 
np.clip(ri, 0.0, 1.0)
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
        r_L = (latency_p95_preds - tau) / (F - tau)
        r_L = np.clip(r_L, 0.0, 1.0)

        return r_L

    except Exception as e: 
        raise ValueError(f"Error processing the latency frequency: {e}")



def compute_composite_risk_index(r_L, r_F, r_P, w_F = 0.2, w_P = 0.2): 
    """
    Eq. 18: Composite risk index 

    Aggregates latency, frequency, packet-loss risks into a unified Risk Index using a weighted 
    noisy-OR reliability formulation

    Arguments: 
        - r_L: latency risk bounded in [0.0, 1.0]
        - r_F: frequency risk bounded in [0.0, 1.0] 
        - r_P: packet-loss risk bounded in [0.0, 1.0]
        - w_F: weight factor applied to the frequency risk 
        - w_P: weight factor applied to the packet-loss risk 
    
    Returns: 
        - RI: aggregated Composite Risk Index bounded within [0.0, 1.0].
    """

    RI = 1 - (1 - r_L)*(1 - w_F*r_F)*(1 - w_P*r_P)
    RI = np.clip(RI, 0.0, 1.0)

    return RI



def compute_true_failures(latency, D_ms): 
    """
    Ground-truth failure identification 

    Determines the extact time steps where observed end-to-end latency violates 
    (exceeds) the maximum allowable deadline D

    Arguments: 
        - latency: Array of measured packet latencies in ms
        - D: maximum allowable latency deadline in ms 
    
    Returns: 
        - Boolean array where True indicates a deadline violation 
    """

    return latency >= D_ms



def compute_TRF(RI, theta_eval = 0.7): 
    """
    Eq. 19: Time-in-Risk Fraction

    Computes the proportion of total observation steps during which the Composite Risk Index (RI)
    meets or exceeds the evaluation thresholds theta_eval 

    Arguments: 
        - RI: Risk Index Composite, with values in [0.0, 1.0]
        - theta_eval: Risk threshold defining an active high-risk state 
    
    Returns: 
        - TRF: Fraction of total time spend in a high-risk state
    """

    try: 
        if (len(RI)) == 0: 
            raise ValueError(f"The RI array is empty!")

        # identify the steps where the risk index exceeds the evaluation threshold
        risk_exceeded = RI >= theta_eval

        # compute the fraction of samples in risk over total observation length T
        TRF = np.sum(risk_exceeded) / len(RI)

        return TRF

    except Exception as e: 
        raise ValueError(f"Error processing the TRF: {e}")



def compute_RER(RI, true_failures, h, theta_eval=0.7):
    """
    Eq. 20: Risk exceedance rate 

    Computes the false-alarm fraction, that is, the proportion of risk alerts (RI >= theta_eval) that 
    were NOT followed by an atual deadline failure within the look-ahead horizon 

    Arguments: 
        - RI: Risk Index Composite with values in [0.0, 1.0]
        - true_failures: Boolean array indicating actual deadline violations 
        - h: Prediction horizon in time steps defining the forward window size
        - theta_eval: Risk threshold triggering an alert 
    
    Returns: 
        - RER: False-alarm rate
    """

    try: 

        # find all indices where the risk index triggers an alert
        risk_exceeded = RI >= theta_eval
        alarm_indices = np.where(risk_exceeded)[0]

        # if no alarms were triggered, the false alarm rate is zero
        if len(alarm_indices) == 0: 
            RER = 0.0
            return RER

        false_alarms = 0
        N = len(RI)

        # evaluate each alert against the subsequent look-ahead window
        for t in alarm_indices: 

            end_window = min(t + h + 1, N)
            future_window = true_failures[t + 1 : end_window]

            # count as a false alarms if no true failure materializes within the horizon
            if not np.any(future_window): 
                false_alarms += 1


        # returns the ratio of uncorfirmed alerts
        RER = false_alarms / len(alarm_indices)
        return RER

    except Exception as e: 
        raise ValueError(f"Error processing RER: {e}")



def compute_fail_coverage(RI, true_failures, h, theta_eval = 0.7): 
    """
    Eq. 21: Failure coverage 

    Computes the faction of actual deadline failures that were preceded by at least one risk alert (RI >= theta_eval)
    within the loop-back prediction 

    Arguments: 
        - RI: Risk Index Composite with values in [0.0, 1.0]
        - true_failures: Boolean array indicating actual deadline violations 
        - h: Prediction horizon in time steps defining the forward window size
        - theta_eval: Risk threshold triggering an alert 
    
    Returns: 
        - coverage: fraction of true values antecipated by an alert
    """

    try: 

        # find all indices where the risk index triggers an failure
        risk_exceeded = RI >= theta_eval
        failure_indices = np.where(true_failures)[0]

        if len(failure_indices): 
            return 1.0          # if there is any failures the cover is 100%

        covered_failures = 0 

        for t in failure_indices: 
            start_window = max(0, t-h)
            past_window = risk_exceeded[start_window : t]

            if np.any(past_window): 
                covered_failures += 1

            coverage = covered_failures / len(failure_indices)

    except Exception as e: 
        raise ValueError(f"Error processing the fail_coverage: {e}")


def compute_warning_reliability(RI, true_failures, theta_eval, h): 
    """
    Eq. 22: Warning reliability 

    Computes predictive precision: the fraction of the risk alerts (RI >= theta_eval) that are followed by at 
    one true deadline failure within the strictly future look-ahead horizon 

    Arguments: 
        - RI: Risk Index Composite with values in [0.0, 1.0]
        - true_failures: Boolean array indicating actual deadline violations 
        - theta_eval: Risk threshold triggering an alert 
        - h: Prediction horizon in time steps defining the forward window size

    Returns: 
        - reliability: Fraction of alerts confirmed by a subsequent failure 
    """

    try: 

        # find all indices where the risk index triggers an alert
        risk_exceeded = RI >= theta_eval
        alarm_indices = np.where(risk_exceeded)[0]


        # perfect reliability when no alarms were raised
        if len(alarm_indices) == 0: 
            return 1.0

        true_alarms = 0
        N = len(RI)

        # evaluate each alarm against the strictly future window 
        for t in alarm_indices: 

            # clamp the upper bound to array lenght N
            end_window = min(t + h + 1, N)
            future_window = true_failures[t +1 : end_window]

            # count as true warning if at least one failure occurs within the horizon 
            if np.any(future_window): 
                true_alarms += 1

        reliability = true_alarms / len(alarm_indices)
        return reliability


    except Exception as e: 
        raise ValueError(f"Error processing the warning reliability: {e}")



def compute_mean_detection_delay(RI, true_failures, h, theta_eval = 0.7): 
    """
    Eq. 23: Mean detection delay (lead time)

    Computes the average advance warning time (in time steps) between the earliest risk index
    within the loop-back window and the actual deadline failure occured at step t

    Arguments: 
        - RI: Risk Index Composite with values in [0.0, 1.0]
        - true_failures: Boolean array indicating actual deadline violations 
        - theta_eval: Risk threshold triggering an alert 
        - h: Prediction horizon in time steps defining the forward window size

    Returns: 
        - mean_delay = Mean advance warning lead time in steps 
    """

    try: 
        
        # find all indices where the risk index triggers an failure
        risk_exceeded = RI >= theta_eval
        failure_indices = np.where(true_failures)[0]

        # return 0.0 if no true failures occured
        if len(failure_indices) == 0: 
            logging.info(f"Any failures have occured!")
            return 0

        detection_delays = []


        # measure advance warning lead-time for each true failure
        for t in failure_indices: 

            # define look-back window
            start_window = max(0, t-h)
            past_window = risk_exceeded[start_window : t]

            # if at least one alert occured in the window, find the earlist one 
            if np.any(past_window): 
                fist_alarm_idx = np.where(past_window)[0][0]
                t_alarm = start_window + fist_alarm_idx

                # record the lead time 
                detection_delays.append(t - t_alarm)

        mean_delay = np.mean(detection_delays) if detection_delays else 0.0

        
        return mean_delay


    except Exception as e: 
        raise ValueError(f"Error processing the warning mean detection delay {e}")



def compute_msis(y_true, y_pred_lower, y_pred_upper, alpha=0.10): 
    """
    Eq. 22: Mean scaled interval score (MSIS)

    Evaluates the quality of probabilistic prediction intervals by balancing interval
    sharpness (width) against coverage panalties when observations fall outside 
    the prediction bounds, normalized by the in-sample naive forecaster

    Arguments: 
        - y_true: Array of observed ground-truth latency values 
        - y_pred_lower: Array of predicted lower quantile bounds (P05)
        - y_pred_upper: Array od predicted upper quantile bounds (P95)
        - alpha: Significance level defining the (1-alpha) prediction interval 
    """

    N = len(y_true)

    # requires at least 2 observations
    if N < 2: 
        return 0.0

    # measure the width of each predicted confidence interval 
    interval_width = y_pred_upper - y_pred_lower

    # apply vectorized indicator penalties whenever Y_i falls outside 
    penalty_lower = np.where(y_true < y_pred_lower, (2.0/alpha) * (y_pred_lower - y_true), 0.0)
    penalty_upper = np.where(y_true > y_pred_upper, (2.0/alpha) * (y_true - y_pred_upper), 0.0)

    # average the combined interval scores across all N times steps 
    score_num = np.sum(interval_width + penalty_lower + penalty_upper)
    mean_score = score_num / N 

    # mean absolute error 
    naive_error = np.mean(np.abs(np.diff(y_true)))

    if naive_error == 0: 
        return mean_score

    # normalized interval 
    msis = mean_score/naive_error
    return msis 



def compute_horizon_scaling_law(msis_results, horizons): 
    """
    Eq. 23: Horizon-scaling law

    Fits a linear regression in log-log space to estimate the power-law degradation 
    exponent (gamma) of the Mean Scaled Interval Score (MSIS) as the prediction 
    horizon h increases: 

    Arguments: 
        - msis_results: Computed MSIS value for each h horizon
        - horizons: predicted horizon
    
    Returns: 
        - gamma: slop of the log-log representing the accuracy degradation rate
        - log_msis_1: intercept of the log-log fit, representating ln(MSIS) at horizon 
    """

    try: 

        # extract and align value strictly matching the order of horizons
        msis_values = np.array([msis_results[horizons] for h in horizons])
        h_values = np.array(horizons)

        # guard against non-positive values
        if np.any(msis_results <= 0): 
            raise ValueError("MSIS contains values <= 0; cannot compute the ln!")

        log_msis = np.log(msis_values)
        log_h = np.log(h_values)

        # fit a first-degree polynomial (y = ax + b) where a = gamma and b = ln 
        coefficients = np.polyfit(log_h, log_msis, 1)
        gamma = coefficients[0]
        log_msis_1 = coefficients[1]

        return gamma, log_msis_1

    except Exception as e:    
        raise ValueError(f"Error processing scaling law: {e}")



def compute_lead_time(h, freq_hz):       # nominal lead time
    """
    Eq. 26: Nominal physical lead time

    Calculates the theoretical physical lead time for horizon h, assuming the network 
    operates exactly at the configured frequency (no jitter)

    Arguments: 
        - h: forecast horizon (steps)
        - freq_hz: nominal freq_hz in Hz
    
    Returns: 
        - lead_time_nom: nominal lead-time in ms
    """

    try: 
        if freq_hz <= 0: 
            raise ValueError(f"Invalid frequency value!")

        delta_ms = 1000 / freq_hz

        lead_time_nom = delta_ms * h

        return lead_time_nom

    except Exception as e: 
        raise ValueError(f"Error processing nominal lead time: {e}")



def compute_realized_lead_time(a_t_array, h): 
    """
    Eq. 27: Realized lead-time

    Calculates the sum of inter-arrival timer over the next h steps to obtain the 
    actual physical time elapsed, accounting for network jitter. 

    Arguments;
        - a_t_array: array with the inter-arrival times in ms 
        - h: forecast horizon (steps)

    Returns: 
        - lead_time_exec = executed/realized lead time in ms  
    """

    try: 
        if len(a_t_array) == 0: 
            raise ValueError(f"The `a_t_array` is empty!")

        # convert to panda series facillitate the operations 
        a_t_series = pd.Series(a_t_array)

        # the sum is represented as the rolling sum of the future 
        lead_time_exec = a_t_series.iloc[::-1].rolling(window=h, min_periods=1).sum().iloc[::-1]

        # shift by 1 so that index t represents the sum from t+1 to t+h
        lead_time_exec = lead_time_exec.shift(-1)

        return lead_time_exec.values

    except Exception as e: 
        raise ValueError(f"Error processing realized lead time: {e}")



def compute_trustworthiness_score(c_fail, r_warn):
    """
    Eq. 28: Trustworthiness score

    Quantifies the reliability of the forecast at horizon h 

    Arguments: 
        - c_fail: failure coverage at horizon h
        - r_warn: warning coverage at horizon h
    
    Returns: 
        - u_trust: trade-off quantification between the anticipation and reliability
    """

    try: 
        if c_fail == 0 and r_warn == 0: 
            return 0

        u_trust = 2 * (c_fail * r_warn)/(c_fail + r_warn)
        return u_trust


    except Exception as e: 
        raise ValueError(f"Error processing the trustworthiness score: {e}")


def compute_pareto(trust_dict, lead_time_dict): 
    """
    Eq. 29: Pareto set 

    Identifies the set of horizons that offer the best trade-off between 
    trustworthiness U_trust (higher is better) and lead-time (higher is better).

    Arguments: 
        - u_trust: trustworthiness_score
        - lead_time: lead-time of each horizon 

    Returns:
        - pareto_set
    """

    try: 

        pareto_set = []
        horizons = list(trust_dict.keys())

        for h in horizons:
            is_dominated = False
            
            for other_h in horizons:
                if h == other_h:
                    continue
                    
                # A point is dominated if another point has:
                # - Higher or equal trustworthiness (better accuracy)
                # - Higher or equal lead-time (more time)
                # - And is strictly better in at least one dimension.
                if (trust_dict[other_h] >= trust_dict[h] and 
                    lead_time_dict[other_h] >= lead_time_dict[h] and 
                    (trust_dict[other_h] > trust_dict[h] or lead_time_dict[other_h] > lead_time_dict[h])):
                    
                    is_dominated = True
                    break
                    
            if not is_dominated:
                pareto_set.append(h)
                
        return sorted(pareto_set)

    except Exception as e: 
        raise ValueError(f"Error processing pareto set: {e}")



def lead_time_efficiency(trust_h, trust_hmax, lead_time_h):
    """
    Eq. 30: lead-time efficiency 

    Quantifies the loss of warning realibility per unit of physical lead time  

    Arguments: 
        - trust_h: Trustworthiness score U_trust(h) for the current horizon.
        - trust_hmax: Trustworthiness score U_trust(h_max) for the maximum horizon.
        - lead_time_h: Nominal physical lead time for horizon h (in ms).
    
    Returns: 
        - eta: lead time efficiency 
    """ 

    try: 

        if lead_time_h <= 0: 
            raise ValueError(f"Invalid lead-time! Dividing by zero!")

        eta = (trust_hmax - trust_h)/lead_time_h

        return eta

    except Exception as e: 
        raise ValueError(f"Error processig lead-time efficiency: {e}")


def compute_weighted_utility():
    """
    Eq. 31 Weighted utility

    Calculates a utility score for a given horizon based on the application's 
    priority (alpha) between accuracy and lead time
    """