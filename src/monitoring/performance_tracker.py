import time
import psutil
import torch
from typing import Dict, Any
from datetime import datetime

class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            'start_time': datetime.now(),
            'requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_latency': 0,
            'peak_memory': 0
        }
        
    def track_request(self, func):
        """Decorator to track request performance"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            self.metrics['requests'] += 1
            
            try:
                result = func(*args, **kwargs)
                self.metrics['successful_requests'] += 1
                
                # Track latency
                latency = time.time() - start_time
                self.metrics['total_latency'] += latency
                
                # Track memory
                memory_used = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                self.metrics['peak_memory'] = max(self.metrics['peak_memory'], memory_used)
                
                return result
                
            except Exception as e:
                self.metrics['failed_requests'] += 1
                raise e
                
        return wrapper
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        uptime = (datetime.now() - self.metrics['start_time']).total_seconds()
        avg_latency = self.metrics['total_latency'] / max(1, self.metrics['successful_requests'])
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.metrics['requests'],
            'success_rate': self.metrics['successful_requests'] / max(1, self.metrics['requests']),
            'average_latency': avg_latency,
            'peak_memory_mb': self.metrics['peak_memory'],
            'mps_memory_allocated': torch.mps.current_allocated_memory() if torch.backends.mps.is_available() else 0
        }
    
    def log_metrics(self):
        """Log current metrics"""
        metrics = self.get_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

def test_performance_tracker():
    """Test the performance tracking system"""
    tracker = PerformanceTracker()
    
    @tracker.track_request
    def sample_request():
        time.sleep(0.1)  # Simulate work
        return "Success"
    
    # Run some test requests
    for _ in range(5):
        sample_request()
    
    tracker.log_metrics()

if __name__ == "__main__":
    test_performance_tracker() 