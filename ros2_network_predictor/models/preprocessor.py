import pandas as pd
import numpy as np

class ROS2Preprocessor:
    def __init__(self):
        pass

    def derive_features(self, df):
        """
        1. Calculates Packet Loss from 'seq' discontinuities.
        2. Handles basic cleanup.
        """
        df = df.copy()
        
        # Ensure sorted by sequence
        df = df.sort_values('seq').reset_index(drop=True)
        
        # --- FEATURE 1: Packet Loss ---
        # Calculate the difference between consecutive sequence numbers
        # If seq is [1, 2, 4], diff is [NaN, 1, 2]. 
        # Loss = diff - 1. (1-1=0 loss, 2-1=1 packet lost)
        seq_diff = df['seq'].diff().fillna(1) # Fill first with 1 (no loss)
        
        # Create the feature: 'packet_loss_count'
        # We clip at 0 because sometimes seq might reset or be weird, preventing negative loss
        df['packet_loss_count'] = (seq_diff - 1).clip(lower=0)
        
        # --- FEATURE 2: Rolling Statistics (Optional but helpful) ---
        # If you have jitter, a rolling mean helps smooth it for the model
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'jitter' in col:
                df[f'{col}_smooth'] = df[col].rolling(window=5, min_periods=1).mean()

        return df

    def split_train_test(self, df, split_ratio=0.8):
        """
        Simple time-based split.
        """
        train_size = int(len(df) * split_ratio)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        return train_df, test_df