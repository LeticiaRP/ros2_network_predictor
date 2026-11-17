import os
import numpy as np
import pandas as pd

BASE_DIR = "../../data"
NUM_SAMPLES = 2000  # Tamanho de cada arquivo CSV
np.random.seed(42)  # Para resultados reproduzíveis

def generate_mock_csv(filepath, freq_hz, qos, topology, num_samples=NUM_SAMPLES):
    
    # 1. Simula o tempo de transmissão (tx)
    tx_period_ns = int(1e9 / freq_hz)
    tx_jitter = np.random.normal(0, tx_period_ns * 0.01, num_samples).astype(int)
    tx = np.cumsum(np.full(num_samples, tx_period_ns) + tx_jitter)
    
    # 2. Simula a latência base
    if topology == "h2h":
        base_latency = 200_000 
    elif topology == "m2m":
        base_latency = 5_000_000
    else: # RTT
        base_latency = 1_000_000
        
    latency = np.random.normal(base_latency, base_latency * 0.1, num_samples).astype(int)
    
    # 3. Insere picos (deadline misses) em 2% dos pacotes - CORRIGIDO AQUI
    spike_indices = np.random.choice(num_samples, size=int(num_samples * 0.02), replace=False)
    spikes = (base_latency * np.random.uniform(2, 5, len(spike_indices))).astype(int)
    latency[spike_indices] += spikes
    
    # 4. O tempo de recepção (rx)
    rx = tx + latency
    
    # 5. Simula a perda de pacotes
    seq = np.arange(num_samples)
    drop_indices = np.random.choice(num_samples, size=int(num_samples * 0.01), replace=False)
    seq = np.delete(seq, drop_indices)
    tx = np.delete(tx, drop_indices)
    rx = np.delete(rx, drop_indices)
    
    # Cria o DataFrame e salva
    df = pd.DataFrame({"seq": seq, "tx": tx, "rx": rx})
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Generated: {filepath} ({len(df)} samples)")

if __name__ == "__main__":
    print("Generating synthetic data...")
    
    # TREINO
    generate_mock_csv(f"{BASE_DIR}/train/latency_h2h_best_effort_5.0Hz.csv", 5, "best_effort", "h2h")
    generate_mock_csv(f"{BASE_DIR}/train/latency_m2m_reliable_20.0hZ.csv", 20, "reliable", "m2m")
    
    # VALIDAÇÃO
    generate_mock_csv(f"{BASE_DIR}/val/latency_h2h_reliable_20.0hZ.csv", 20, "reliable", "h2h")
    
    # TESTE
    generate_mock_csv(f"{BASE_DIR}/test/rtt_h2m_best_effort_100.0Hz.csv", 100, "best_effort", "rtt_h2m")
    
    print("Done!")