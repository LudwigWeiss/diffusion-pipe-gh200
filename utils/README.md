# Performance Monitoring System

This directory contains the performance monitoring system for training pipelines, with special support for NVIDIA GH200 Grace Hopper architecture.

## Components

### Base Monitor (`monitoring.py`)
The base performance monitor tracks fundamental training metrics with robust error handling:
- Training loss and learning rate
- Batch processing time
- Gradient norms and parameter counts
- Memory utilization with safe fallbacks
- Training throughput
- Automatic resource cleanup

### GH200 Monitor (`gh200_monitoring.py`)
Extended monitoring for GH200 architecture includes:
- NVLink utilization metrics with CUDA event timing
- Unified memory statistics with error recovery
- Pipeline efficiency metrics with validation
- Communication bandwidth monitoring
- Enhanced memory tracking with safe defaults

## Usage

### Basic Usage
```python
from utils.monitoring import PerformanceMonitor

# Initialize with TensorBoard writer
monitor = PerformanceMonitor(tb_writer)

try:
    # Track training steps with automatic cleanup
    with monitor.track_step(step):
        # Your training code here
        monitor.update(step, loss, lr, batch_time)
        monitor.log_memory_stats(step)
finally:
    # Resources are automatically cleaned up
    monitor.cleanup()
```

### GH200-Specific Monitoring
```python
from utils.gh200_monitoring import GH200PerformanceMonitor

# Initialize with TensorBoard writer
monitor = GH200PerformanceMonitor(tb_writer)

try:
    # Track training steps with enhanced metrics
    with monitor.track_step(step):
        # NVLink measurement context
        with monitor.nvlink_measurement():
            # Your training code here
            monitor.update(step, loss, lr, batch_time)
        
        # Log additional metrics with error handling
        monitor.log_pipeline_stats(model_engine, step)
        monitor.log_communication_stats(step)
finally:
    # Automatic cleanup of CUDA events and resources
    monitor.cleanup()
```

## Configuration

Enable GH200 monitoring in your TOML config with safety checks:
```toml
[gh200]
enabled = true

[monitoring]
wall_clock_breakdown = true
memory_breakdown = true
nvlink_monitoring = true
unified_memory_monitoring = true

# Optional: Configure error handling
verbose_logging = true
log_level = "WARNING"  # Options: DEBUG, INFO, WARNING, ERROR
```

## TensorBoard Metrics

### Base Metrics (with error handling)
- `training/loss` (validated float)
- `training/learning_rate` (validated float)
- `training/batch_time` (validated float)
- `performance/step_time_sec` (with error recovery)
- `performance/throughput` (validated positive float)
- `gradients/norm` (with parameter validation)
- `gradients/param_count` (new metric)
- `memory/allocated` (with safe defaults)
- `memory/reserved` (with safe defaults)
- `memory/peaks` (with validation)

### GH200-Specific Metrics (with safety)
- `gh200/nvlink/tx_bytes` (CUDA event timed)
- `gh200/nvlink/rx_bytes` (CUDA event timed)
- `gh200/nvlink/bandwidth` (calculated safely)
- `gh200/unified_memory/used` (with fallback)
- `gh200/unified_memory/free` (with fallback)
- `gh200/unified_memory/cpu_page_faults` (validated)
- `gh200/pipeline/parallel_size` (validated)
- `gh200/pipeline/micro_batch_size` (with defaults)
- `gh200/communication/nvlink_bandwidth` (safely calculated)
- `gh200/communication/cpu_transfer_bandwidth` (with validation)

## Best Practices

1. Error Handling:
   - All metrics collection has proper error recovery
   - Resource cleanup is guaranteed
   - Safe defaults for missing values
   - Proper type validation

2. Memory Management:
   - Automatic CUDA event cleanup
   - Safe memory stat collection
   - Proper resource deallocation
   - Monitored memory peaks

3. Pipeline Optimization:
   - Validated pipeline metrics
   - Safe attribute access
   - Proper error reporting
   - Performance impact monitoring

4. Performance Monitoring:
   - Use TensorBoard visualizations
   - Monitor error logs
   - Track resource usage
   - Watch for performance degradation

5. Resource Management:
   - Proper CUDA synchronization
   - Automatic cleanup in __del__
   - Context manager usage
   - Safe metric collection

## Error Handling

The monitoring system includes comprehensive error handling:
1. All metric collection operations are wrapped in try-except blocks
2. Resource cleanup is guaranteed through context managers
3. Invalid values are caught and reported
4. Safe defaults are provided for missing metrics
5. Proper logging of all errors and warnings 