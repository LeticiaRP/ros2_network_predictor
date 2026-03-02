import pandas as pd
import numpy as np

class ROS2Preprocessor:
    
    def __init__(self):
        pass




    def derive_features(self, df):
        df = df.copy()
        
        # Ensure sorted by sequence
        df = df.sort_values('seq').reset_index(drop=True)
        
        # Packet loss 
        seq_diff = df['seq'].diff().fillna(1) # Fill first with 1 (no loss)
        
        df['packet_loss_count'] = (seq_diff - 1).clip(lower=0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'jitter' in col:
                df[f'{col}_smooth'] = df[col].rolling(window=5, min_periods=1).mean()

        return df
    




    def split_train_test(self, df, split_ratio=0.8):

        train_size = int(len(df) * split_ratio)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        return train_df, test_df