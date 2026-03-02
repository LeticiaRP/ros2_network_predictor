import os
import re
import pandas as pd
from glob import glob



class ROS2DataLoader:
    def __init__(self, raw_data_path):
        self.raw_path = raw_data_path
        
        self.filename_pattern = re.compile(
            r"(?P<metric>latency|host_.*jitter|rtt|mcu_.*jitter)" # Removed clock_sync
            r"(?:_(?P<scenario>h2h|h2m|m2h|m2m))?" 
            r"_(?P<qos>best_effort|reliable)"
            r"_(?P<freq>[\d\.]+Hz)"
            r"_(?P<stressor>True|False)"
            r"_(?P<timestamp>\d+)\.csv"
        )



    def group_files_by_experiment(self):

        files = glob(os.path.join(self.raw_path, "*.csv"))
        experiments = {}


        for f_path in files:
            f_name = os.path.basename(f_path)
            
            # Skip clock_sync files 
            if "clock_sync" in f_name:
                continue


            # Parse Metadata
            match = self.filename_pattern.match(f_name)
            if not match:
                continue
            

            meta = match.groupdict()
            

            exp_key = (meta['timestamp'], meta['scenario'], meta['qos'], meta['freq'])
            

            if exp_key not in experiments:
                experiments[exp_key] = []
                

            experiments[exp_key].append({
                'path': f_path, 
                'metric': meta['metric']
            })
            
        return experiments



    def process_experiment(self, file_list):

        merged_df = pd.DataFrame()


        for item in file_list:
            
            df = pd.read_csv(item['path'])
            
            
            if 'seq' not in df.columns:
                print(f"Warning: File {item['path']} missing 'seq'. Skipping.")
                continue

            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='seq', how='outer')


        if not merged_df.empty:
            merged_df = merged_df.sort_values('seq').reset_index(drop=True)

            merged_df = merged_df.fillna(method='ffill').fillna(0)
            
        return merged_df



if __name__ == "__main__":
   
    loader = ROS2DataLoader("../../data/raw") 
    
    groups = loader.group_files_by_experiment()
    print(f"Found {len(groups)} unique experiment runs (excluding clock_sync).")
    
    
    if len(groups) > 0:
        
        first_key = list(groups.keys())[0]
        df_result = loader.process_experiment(groups[first_key])
        
        print("\nColumns found:", df_result.columns.tolist())