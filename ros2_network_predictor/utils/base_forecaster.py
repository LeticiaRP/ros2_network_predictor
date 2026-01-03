class BaseForecaster:
    def __init__(self, node):
        self.node = node
    def update_and_predict(self, latency_ms):
        raise NotImplementedError