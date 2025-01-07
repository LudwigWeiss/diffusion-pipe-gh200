import torch
from contextlib import contextmanager
from .monitoring import PerformanceMonitor

class GH200PerformanceMonitor(PerformanceMonitor):
    """Performance monitor with GH200-specific metrics."""
    
    def __init__(self, tb_writer=None):
        super().__init__(tb_writer)
        self.nvlink_metrics = {}
        self.unified_memory_metrics = {}
        self.pipeline_metrics = {}
        self._setup_cuda_events()
        
    def _setup_cuda_events(self):
        """Initialize CUDA events for timing."""
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None
            self.end_event = None
            
    def __del__(self):
        """Cleanup CUDA events."""
        if hasattr(self, 'start_event') and self.start_event is not None:
            del self.start_event
        if hasattr(self, 'end_event') and self.end_event is not None:
            del self.end_event
        
    @contextmanager
    def nvlink_measurement(self):
        """Context manager for NVLink measurements."""
        try:
            if torch.cuda.is_available() and self.start_event is not None:
                torch.cuda.synchronize()
                self.start_event.record()
                start_stats = self._get_nvlink_stats()
                yield
                self.end_event.record()
                torch.cuda.synchronize()
                end_stats = self._get_nvlink_stats()
                self._update_nvlink_metrics(start_stats, end_stats)
            else:
                yield
        except Exception as e:
            print(f"Warning: NVLink measurement failed: {str(e)}")
            yield
            
    def _get_nvlink_stats(self):
        """Get NVLink statistics safely."""
        try:
            device = torch.cuda.current_device()
            # Use proper NVIDIA management library API here
            # This is a placeholder for actual NVLink stats
            return {
                'tx_bytes': 0,
                'rx_bytes': 0,
                'bandwidth': 0
            }
        except Exception as e:
            print(f"Warning: Failed to get NVLink stats: {str(e)}")
            return None
            
    def _update_nvlink_metrics(self, start_stats, end_stats):
        """Update NVLink metrics with proper error handling."""
        try:
            if start_stats is not None and end_stats is not None:
                elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000.0  # Convert to seconds
                if elapsed_time > 0:
                    self.nvlink_metrics.update({
                        'tx_bytes': end_stats['tx_bytes'] - start_stats['tx_bytes'],
                        'rx_bytes': end_stats['rx_bytes'] - start_stats['rx_bytes'],
                        'bandwidth': (end_stats['bandwidth'] - start_stats['bandwidth']) / elapsed_time
                    })
        except Exception as e:
            print(f"Warning: Failed to update NVLink metrics: {str(e)}")
        
    def update(self, step, loss, lr, batch_time):
        """Update monitoring metrics including GH200-specific ones."""
        super().update(step, loss, lr, batch_time)
        
        if not torch.cuda.is_available():
            return
            
        try:
            device = torch.cuda.current_device()
            
            # Monitor unified memory with error handling
            try:
                memory_stats = torch.cuda.memory_stats(device)
                self.unified_memory_metrics.update({
                    'unified_used': memory_stats.get('unified_used', 0),
                    'unified_free': memory_stats.get('unified_free', 0),
                    'cpu_page_faults': memory_stats.get('cpu_page_faults', 0)
                })
            except Exception as e:
                print(f"Warning: Failed to collect unified memory stats: {str(e)}")
            
            # Log metrics if available
            if self.tb_writer:
                if self.nvlink_metrics:
                    self.tb_writer.add_scalars('gh200/nvlink', self.nvlink_metrics, step)
                if self.unified_memory_metrics:
                    self.tb_writer.add_scalars('gh200/unified_memory', self.unified_memory_metrics, step)
                    
        except Exception as e:
            print(f"Warning: Failed to update GH200 metrics: {str(e)}")
    
    def log_pipeline_stats(self, model_engine, step):
        """Log GH200 pipeline-specific statistics with validation."""
        if not self.tb_writer:
            return
            
        try:
            # Validate model_engine
            if not hasattr(model_engine, 'pipeline_parallel_size'):
                return
                
            # Collect pipeline metrics safely
            self.pipeline_metrics.update({
                'pipeline_parallel_size': getattr(model_engine, 'pipeline_parallel_size', 1),
                'micro_batch_size': getattr(model_engine, 'micro_batch_size', 1),
                'gradient_accumulation_steps': (
                    model_engine.gradient_accumulation_steps()
                    if hasattr(model_engine, 'gradient_accumulation_steps')
                    else 1
                )
            })
            
            self.tb_writer.add_scalars('gh200/pipeline', self.pipeline_metrics, step)
            
        except Exception as e:
            print(f"Warning: Failed to log pipeline stats: {str(e)}")
    
    def log_memory_stats(self, step):
        """Enhanced memory statistics for GH200 with proper error handling."""
        try:
            super().log_memory_stats(step)
            
            if not self.tb_writer or not torch.cuda.is_available():
                return
                
            device = torch.cuda.current_device()
            memory_stats = torch.cuda.memory_stats(device)
            
            # GH200-specific memory metrics with validation
            gh200_memory = {}
            for key, default in [
                ('unified_capacity', 0),
                ('unified_peak', 0),
                ('device_reserved', 0)
            ]:
                try:
                    gh200_memory[key] = memory_stats.get(key, default)
                except Exception:
                    gh200_memory[key] = default
            
            self.tb_writer.add_scalars('gh200/memory', gh200_memory, step)
            
        except Exception as e:
            print(f"Warning: Failed to log GH200 memory stats: {str(e)}")
    
    def log_communication_stats(self, step):
        """Log GH200 communication statistics with proper error handling."""
        if not self.tb_writer or not torch.cuda.is_available():
            return
            
        try:
            # Calculate communication metrics safely
            batch_time = self.metrics.get('batch_time', 1)
            if batch_time > 0:
                nvlink_stats = {
                    'nvlink_bandwidth': self.nvlink_metrics.get('bandwidth', 0),
                    'cpu_transfer_bandwidth': (
                        self.unified_memory_metrics.get('cpu_page_faults', 0) / batch_time
                    )
                }
                
                self.tb_writer.add_scalars('gh200/communication', nvlink_stats, step)
                
        except Exception as e:
            print(f"Warning: Failed to log communication stats: {str(e)}")
    
    def reset_metrics(self):
        """Reset all metrics including GH200-specific ones."""
        try:
            super().reset_metrics()
            self.nvlink_metrics = {}
            self.unified_memory_metrics = {}
            self.pipeline_metrics = {}
        except Exception as e:
            print(f"Warning: Failed to reset metrics: {str(e)}") 