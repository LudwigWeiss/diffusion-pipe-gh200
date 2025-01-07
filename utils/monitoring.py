import time
from contextlib import contextmanager
import torch
from torch.cuda.amp import GradScaler
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Base class for monitoring training performance metrics."""
    
    def __init__(self, tb_writer=None):
        """Initialize the performance monitor.
        
        Args:
            tb_writer: TensorBoard SummaryWriter instance for logging
        """
        self.tb_writer = tb_writer
        self.step_start_time = None
        self.metrics = {}
        self.grad_scaler = None
        self._initialize_grad_scaler()
        
    def _initialize_grad_scaler(self):
        """Initialize gradient scaler with error handling."""
        try:
            if torch.cuda.is_available():
                self.grad_scaler = GradScaler()
        except Exception as e:
            logger.warning(f"Failed to initialize gradient scaler: {str(e)}")
            self.grad_scaler = None
            
    def __del__(self):
        """Cleanup resources."""
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources and reset state."""
        try:
            self.reset_metrics()
            if self.grad_scaler is not None:
                del self.grad_scaler
                self.grad_scaler = None
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
        
    @contextmanager
    def track_step(self, step):
        """Context manager to track the duration of a training step.
        
        Args:
            step: Current training step number
        """
        error = None
        try:
            self.step_start_time = time.time()
            yield
        except Exception as e:
            error = e
            logger.error(f"Error during step tracking: {str(e)}")
            raise
        finally:
            try:
                if self.step_start_time is not None:
                    duration = time.time() - self.step_start_time
                    if self.tb_writer and not error:
                        self.tb_writer.add_scalar('performance/step_time_sec', duration, step)
                    self.step_start_time = None
            except Exception as e:
                logger.warning(f"Error logging step duration: {str(e)}")
    
    def update(self, step, loss, lr, batch_time):
        """Update monitoring metrics.
        
        Args:
            step: Current training step
            loss: Training loss value
            lr: Current learning rate
            batch_time: Time taken for the batch
        """
        try:
            # Validate inputs
            if not all(isinstance(x, (int, float)) for x in [loss, lr, batch_time]):
                raise ValueError("Invalid metric values")
                
            self.metrics.update({
                'loss': float(loss),
                'learning_rate': float(lr),
                'batch_time': float(batch_time)
            })
            
            if self.tb_writer:
                self.tb_writer.add_scalars('training', self.metrics, step)
                
        except Exception as e:
            logger.warning(f"Failed to update metrics: {str(e)}")
    
    def log_gradient_norm(self, model_engine, step):
        """Log gradient norm statistics.
        
        Args:
            model_engine: DeepSpeed model engine
            step: Current training step
        """
        if not self.tb_writer:
            return
            
        try:
            # Validate model_engine
            if not hasattr(model_engine, 'parameters'):
                logger.warning("Invalid model_engine object")
                return
                
            # Calculate gradient norms safely
            grad_norm = 0.0
            param_count = 0
            
            for param in model_engine.parameters():
                if param.grad is not None:
                    try:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                        param_count += 1
                    except Exception as e:
                        logger.warning(f"Error processing gradient: {str(e)}")
                        continue
            
            if param_count > 0:
                grad_norm = grad_norm ** 0.5
                self.tb_writer.add_scalar('gradients/norm', grad_norm, step)
                self.tb_writer.add_scalar('gradients/param_count', param_count, step)
                
        except Exception as e:
            logger.warning(f"Failed to log gradient norm: {str(e)}")
    
    def log_memory_stats(self, step):
        """Log CUDA memory statistics.
        
        Args:
            step: Current training step
        """
        if not self.tb_writer or not torch.cuda.is_available():
            return
            
        try:
            device = torch.cuda.current_device()
            memory_stats = torch.cuda.memory_stats(device)
            
            # Basic memory metrics
            basic_metrics = {
                'allocated': memory_stats.get('allocated_bytes.all.current', 0),
                'reserved': memory_stats.get('reserved_bytes.all.current', 0),
                'active': memory_stats.get('active_bytes.all.current', 0)
            }
            
            # Peak memory metrics
            peak_metrics = {
                'allocated': memory_stats.get('allocated_bytes.all.peak', 0),
                'reserved': memory_stats.get('reserved_bytes.all.peak', 0)
            }
            
            # Log metrics safely
            if any(basic_metrics.values()):
                self.tb_writer.add_scalars('memory', basic_metrics, step)
            if any(peak_metrics.values()):
                self.tb_writer.add_scalars('memory/peaks', peak_metrics, step)
                
        except Exception as e:
            logger.warning(f"Failed to log memory stats: {str(e)}")
    
    def log_throughput(self, step, samples_per_sec):
        """Log training throughput.
        
        Args:
            step: Current training step
            samples_per_sec: Training samples processed per second
        """
        try:
            if self.tb_writer and isinstance(samples_per_sec, (int, float)) and samples_per_sec > 0:
                self.tb_writer.add_scalar('performance/throughput', float(samples_per_sec), step)
        except Exception as e:
            logger.warning(f"Failed to log throughput: {str(e)}")
    
    def reset_metrics(self):
        """Reset all tracked metrics."""
        try:
            self.metrics = {}
            self.step_start_time = None
        except Exception as e:
            logger.warning(f"Failed to reset metrics: {str(e)}") 