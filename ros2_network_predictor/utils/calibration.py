import pandas as pd
import numpy as np
from pathlib import Path
import json
import os

def parse_filename(filename):

    parts = filename.replace('.csv', '').split('_')
    

    try:
        if filename.startswith('rtt_'):
            # parts[0]=rtt, parts[1]=h2m, parts[2]=reliable, parts[3]=50.0Hz
            scenario = f"rtt_{parts[1]}"
            qos = parts[2]
            freq_str = parts[3].replace('Hz', '')
            return {'scenario': scenario, 'qos': qos, 'frequency_hz': float(freq_str)}
            

        elif filename.startswith('latency_'):
            # For 'latency_h2h_reliable_50.0Hz_...'
            scenario = parts[1]
            qos = parts[2]
            freq_str = parts[3].replace('Hz', '')
            return {'scenario': scenario, 'qos': qos, 'frequency_hz': float(freq_str)}
    
    except (IndexError, ValueError):
        return None
    
    return None




def calibrate_thresholds():

    # Path configuration
    data_dir = Path.home() / "ros2_ws" / "src" / "ros2_network_predictor" / "data" / "raw"
    

    print(f"Searching for CSVs in: {data_dir}")
    if not data_dir.exists():
        print(f"ERROR: Directory does not exist: {data_dir}")
        return
    

    all_files = list(data_dir.glob("*.csv"))
    scenario_data = {}
    processed_count = 0


    for csv_file in all_files:
        filename = csv_file.name
        file_info = parse_filename(filename)
        
        if not file_info:
            continue
            
        scenario = file_info['scenario']
        

        # SKIP only direct H2M/M2H (rtt_h2m and rtt_m2h will pass)
        if scenario in ['h2m', 'm2h']:
            continue


        try:
            df = pd.read_csv(csv_file)
            
            # Find the correct latency column (ignoring 'seq')
            latency_col = None
            for col in df.columns:
                c_low = col.lower()
                if any(k in c_low for k in ['latency', 'rtt', 'data', 'value']) and 'seq' not in c_low:
                    latency_col = col
                    break
            

            if latency_col:

                # Convert nanoseconds to milliseconds
                raw_data = pd.to_numeric(df[latency_col], errors='coerce').dropna()
                data_ms = raw_data[raw_data >= 0] / 1_000_000.0
                
                # Outlier removal (0.1%)
                if len(data_ms) > 1000:
                    data_ms = data_ms[data_ms <= data_ms.quantile(0.999)]
                
                key = f"{scenario}_{file_info['qos']}_{file_info['frequency_hz']}Hz"
                if key not in scenario_data:
                    scenario_data[key] = []
                scenario_data[key].extend(data_ms.tolist())
                processed_count += 1
                

        except Exception as e:
            print(f"Error in {filename}: {e}")



    # Results calculation
    print(f"\n{'Configuration':<35} | {'Samples':<8} | {'Safety (ms)':<12} | {'Failure (ms)'}")
    print("-" * 85)
    


    threshold_map = {}

    for key, data_list in sorted(scenario_data.items()):
        data = np.array(data_list)
        if len(data) < 100: continue
        
        p95 = np.percentile(data, 95)
        p99 = np.percentile(data, 99)
        max_val = np.max(data)
        

        safety = round(float(p95), 2)
        
        # margin: 0.1 for RTT, 0.2 for others
        margin = 0.1 if key.startswith('rtt_') else 0.2
        failure = round(float(p99 + (max_val - p99) * margin), 2)
        

        # ensure lead-time window exists
        if failure <= safety:
            failure = round(safety * 1.15, 2)
            

        threshold_map[key] = [safety, failure]
        print(f"{key:<35} | {len(data):<8} | {safety:<12.2f} | {failure:<12.2f}")


    print("\n--- UPDATED DICTIONARY ---")
    print(f"THRESHOLD_DEFAULTS = {json.dumps(threshold_map, indent=4)}")

if __name__ == "__main__":
    calibrate_thresholds()