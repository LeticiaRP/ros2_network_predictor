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

        # identify topology 
        if measure_type == "rtt": 
            topology_key = f"rtt_{scenario}"
        elif measure_type == "latency": 
            topology_key = scenario
        else: 
            raise ValueError(f"Unknown measure_type '{measure_type}' in  {basename}")


        # identify qos and frequency 
        if parts[2] == "best" and parts[3] == "effort": 
            qos_policy = "best_effort"
            frequency = parts[4]

        elif parts[2] == "reliable": 
            qos_policy = parts[2]
            frequency = parts[3]
        
        else: 
            raise ValueError(f"Unknown qos policy {parts[2]} in {basename}")


        # get value from the dictionary
        qos = QOS.get(qos_policy)
        freq_hz = FREQUENCY_MAP.get(frequency)
        topology = TOPOLOGY_MAP.get(topology_key)

        return qos, freq_hz, topology

    
    except Exception as e: 
        
        print(f"Error {e}")
        pass



