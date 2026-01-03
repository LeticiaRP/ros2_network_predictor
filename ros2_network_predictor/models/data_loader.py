import os
import re
import pandas as pd
from glob import glob

class ROS2DataLoader:
    def __init__(self, raw_data_path):
        self.raw_path = raw_data_path
        # Modified Regex: Still captures everything, but we will filter in the loop
        self.filename_pattern = re.compile(
            r"(?P<metric>latency|host_.*jitter|rtt|mcu_.*jitter)" # Removed clock_sync
            r"(?:_(?P<scenario>h2h|h2m|m2h|m2m))?" 
            r"_(?P<qos>best_effort|reliable)"
            r"_(?P<freq>[\d\.]+Hz)"
            r"_(?P<stressor>True|False)"
            r"_(?P<timestamp>\d+)\.csv"
        )

    def group_files_by_experiment(self):
        """
        Groups files by experiment run (Timestamp). 
        Explicitly IGNORES 'clock_sync' files.
        """
        files = glob(os.path.join(self.raw_path, "*.csv"))
        experiments = {}

        for f_path in files:
            f_name = os.path.basename(f_path)
            
            # 1. EXPLICIT FILTER: Skip clock_sync
            if "clock_sync" in f_name:
                continue

            # 2. Parse Metadata
            match = self.filename_pattern.match(f_name)
            if not match:
                # This catches files that don't match our specific pattern
                # print(f"Skipping unmatched file: {f_name}")
                continue
            
            meta = match.groupdict()
            
            # Key = Unique identifier for this run
            exp_key = (meta['timestamp'], meta['scenario'], meta['qos'], meta['freq'])
            
            if exp_key not in experiments:
                experiments[exp_key] = []
                
            experiments[exp_key].append({
                'path': f_path, 
                'metric': meta['metric']
            })
            
        return experiments

    def process_experiment(self, file_list):
        """
        Merges all CSVs for one experiment into a single DataFrame on 'seq'.
        """
        merged_df = pd.DataFrame() # Start empty

        for item in file_list:
            # Read CSV
            df = pd.read_csv(item['path'])
            
            # Validation: specific files must have 'seq'
            if 'seq' not in df.columns:
                print(f"Warning: File {item['path']} missing 'seq'. Skipping.")
                continue

            # Rename columns to avoid collisions (e.g. both files might have 'jitter')
            # We prefix the value columns with the metric name if needed, 
            # but usually the CSV headers are unique enough. 
            # Let's just merge strictly on 'seq'.
            
            if merged_df.empty:
                merged_df = df
            else:
                # Outer Merge: Keeps rows even if one file dropped a packet that another saw
                merged_df = pd.merge(merged_df, df, on='seq', how='outer')

        if not merged_df.empty:
            # Sort by sequence ID to ensure time-order
            merged_df = merged_df.sort_values('seq').reset_index(drop=True)
            
            # Clean up NaNs (caused by packet loss in one metric but not another)
            # Forward fill is safe for sensor data (assume previous value holds)
            merged_df = merged_df.fillna(method='ffill').fillna(0)
            
        return merged_df

# --- TESTING BLOCK (Run this file directly) ---
if __name__ == "__main__":
    # Update this path to where your CSVs actually are
    loader = ROS2DataLoader("/home/leticia/Desktop/UTFPR/deep-learning/ros2_dl_project/data/raw") 
    
    groups = loader.group_files_by_experiment()
    print(f"Found {len(groups)} unique experiment runs (excluding clock_sync).")
    
    if len(groups) > 0:
        # Test merging the first group found
        first_key = list(groups.keys())[0]
        print(f"\nProcessing Experiment: {first_key}")
        
        df_result = loader.process_experiment(groups[first_key])
        
        print(f"Result Shape: {df_result.shape}")
        print("\nFirst 5 rows:")
        print(df_result.head())
        
        # Check if we successfully merged different metrics
        print("\nColumns found:", df_result.columns.tolist())