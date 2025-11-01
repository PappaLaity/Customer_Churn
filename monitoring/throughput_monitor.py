import time

class ThroughputMonitor:
    def __init__(self):
        self.prediction_count = 0
        self.start_time = None
        self.end_time = None

    def start_monitoring(self):
        """Start monitoring throughput by recording the start time."""
        self.start_time = time.time()
        self.prediction_count = 0

    def stop_monitoring(self):
        """Stop monitoring and record the end time."""
        self.end_time = time.time()

    def log_prediction(self):
        """Log a prediction to increment the count."""
        if self.start_time is not None:
            self.prediction_count += 1

    def get_throughput(self):
        """Calculate and return the throughput (predictions per second)."""
        if self.start_time is None or self.end_time is None:
            return 0
        elapsed_time = self.end_time - self.start_time
        return self.prediction_count / elapsed_time if elapsed_time > 0 else 0

    def reset(self):
        """Reset the monitor for a new monitoring session."""
        self.prediction_count = 0
        self.start_time = None
        self.end_time = None