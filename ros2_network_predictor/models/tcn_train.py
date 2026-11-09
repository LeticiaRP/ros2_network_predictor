import os

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

def parse_matadata(filename): 
    """
    Extracts QoS, topology, and frequency from the filename

    Argument: 
    - filename: file's path 

    Returns: 
    - topology
    - freq_hz 
    - qos 
    """

    measure_type = None 
    scenario = None
    qos_policy = None
    frequency = None


    basename = os.path.basename(filename)
    parts = basename.split("_")

    parts[0] = measure_type     # latency (one-way-path) / rtt /(round time trip)
    parts[1] = scenario         # scenario (h2h, m2m, h2m, m2h)
    parts[2] = qos_policy       # qos (best effort, reliable)
    parts[3] = frequency        # frequency (5.0Hz, 10.0Hz, 20.0Hz, 50.0Hz, 100.0Hz, 200.0Hz)

    try: 

        if measure_type == "rtt": 
            topology = TOPOLOGY_MAP.get(f"rtt_{scenario}")
        elif measure_type == "latency": 
            topology = TOPOLOGY_MAP.get(f"{scenario}")
        else: 
            return Exception

        qos = TOPOLOGY_MAP.get(f"{qos_policy}")
        freq_hz = TOPOLOGY_MAP.get(f"{frequency}")


        return topology, qos, freq_hz
    
    except Exception as e: 
        print(f"Error {e}")
        pass 

    

