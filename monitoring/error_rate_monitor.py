class ErrorRateMonitor:
    def __init__(self):
        self.total_predictions = 0
        self.error_count = 0

    def log_error(self):
        self.error_count += 1
        self.total_predictions += 1

    def log_prediction(self):
        self.total_predictions += 1

    def calculate_error_rate(self):
        if self.total_predictions == 0:
            return 0.0
        return self.error_count / self.total_predictions

    def get_error_statistics(self):
        return {
            "total_predictions": self.total_predictions,
            "error_count": self.error_count,
            "error_rate": self.calculate_error_rate(),
        }