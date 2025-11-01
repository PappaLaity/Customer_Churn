class AlertSystem:
    def __init__(self):
        self.alerts = []

    def configure_alert(self, metric_name, threshold, alert_type="warning"):
        alert = {
            "metric_name": metric_name,
            "threshold": threshold,
            "alert_type": alert_type,
            "is_triggered": False
        }
        self.alerts.append(alert)

    def check_alerts(self, metrics):
        for alert in self.alerts:
            metric_value = metrics.get(alert["metric_name"])
            if metric_value is not None:
                if alert["alert_type"] == "warning" and metric_value > alert["threshold"]:
                    alert["is_triggered"] = True
                    self.send_notification(alert)

    def send_notification(self, alert):
        print(f"Alert triggered for {alert['metric_name']}: "
              f"Value exceeded threshold of {alert['threshold']}.")

    def reset_alert(self, metric_name):
        for alert in self.alerts:
            if alert["metric_name"] == metric_name:
                alert["is_triggered"] = False
                print(f"Alert for {metric_name} has been reset.")