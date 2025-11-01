class LatencyMonitor:
    def __init__(self):
        self.latencies = []
        self.monitoring = False

    def start_monitoring(self):
        """Start monitoring prediction latency."""
        self.monitoring = True
        self.latencies.clear()

    def stop_monitoring(self):
        """Stop monitoring prediction latency."""
        self.monitoring = False

    def log_latency(self, latency):
        """Log a new latency value."""
        if self.monitoring:
            self.latencies.append(latency)

    def get_average_latency(self):
        """Retrieve the average latency."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)