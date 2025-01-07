# Diffusion-Pipe GH200

A high-performance diffusion model training pipeline optimized for NVIDIA GH200 Grace Hopper architecture, with a focus on video generation models like HunyuanVideo.

## Overview

This project provides a specialized training pipeline that leverages the unique capabilities of the NVIDIA GH200 architecture:
- Unified CPU-GPU memory architecture with optimized memory management
- Enhanced NVLink bandwidth utilization through event-based monitoring
- FP8/BF16 acceleration with automatic mixed precision
- Large-scale pipeline parallelism with dynamic load balancing

Key features:
- DeepSpeed ZeRO-2 integration with GH200-specific optimizations
- Advanced performance monitoring system with real-time metrics
- Efficient video model training support with pipeline parallelism
- Automatic resource management and cleanup
- Comprehensive error handling and recovery

## Technical Details

### GH200 Optimizations

1. Memory Management
   - Unified memory optimization with dynamic page migration
   - Smart CPU offloading with pinned memory
   - Automatic buffer size adjustment
   - Efficient gradient accumulation

2. Communication
   - NVLink-aware data transfer
   - Optimized collective operations
   - Efficient pipeline scheduling
   - Reduced CPU-GPU synchronization

3. Computation
   - FP8 transformer operations
   - BF16 activation checkpointing
   - Efficient pipeline bubbles
   - Dynamic batch sizing

### Performance Monitoring

The monitoring system includes:

1. Memory Tracking
```python
class GH200PerformanceMonitor:
    def log_memory_stats(self):
        # Track unified memory
        memory_stats = {
            'unified_used': stats.get('unified_used', 0),
            'unified_free': stats.get('unified_free', 0),
            'cpu_page_faults': stats.get('cpu_page_faults', 0)
        }
```

2. NVLink Monitoring
```python
    def nvlink_measurement(self):
        # Event-based bandwidth measurement
        with cuda.Event() as start, cuda.Event() as end:
            start.record()
            # Operation
            end.record()
            # Calculate bandwidth
```

3. Pipeline Statistics
```python
    def log_pipeline_stats(self):
        # Track efficiency
        stats = {
            'pipeline_parallel_size': size,
            'micro_batch_size': batch_size,
            'gradient_accumulation_steps': steps
        }
```

## Architecture

### Core Components

1. Training Pipeline (`train.py`)
   - Distributed training coordination
   - Pipeline stage management
   - Memory optimization
   - Gradient synchronization

2. Performance Monitoring (`utils/monitoring.py`, `utils/gh200_monitoring.py`)
   - Base monitoring infrastructure
   - GH200-specific metrics
   - Resource tracking
   - Error handling

3. Model Support (`models/`)
   - HunyuanVideo integration
   - Model wrapper architecture
   - Pipeline adaptation
   - Mixed precision support

## Configuration System

The configuration system uses TOML files with comprehensive settings:

### Base Configuration (`configs/gh200_config.toml`)
```toml
# Training settings
epochs = 1000
micro_batch_size_per_gpu = 1
pipeline_stages = 4
gradient_accumulation_steps = 4

# GH200 optimizations
[ds_config.zero_optimization]
stage = 2
cpu_offload = true
overlap_comm = true
contiguous_gradients = true
stage3_prefetch_bucket_size = 1e9
offload_optimizer = {
    device = "cpu",
    pin_memory = true,
    buffer_count = 4
}
```

### Model-Specific Configuration (`configs/hunyuan_video_gh200.toml`)
```toml
[model]
type = "hunyuan-video"
dtype = "bfloat16"
transformer_dtype = "float8"  # GH200 optimized

[monitoring]
wall_clock_breakdown = true
memory_breakdown = true
nvlink_monitoring = true
unified_memory_monitoring = true
```

## Implementation Details

### Memory Management

1. Unified Memory
```python
# Optimized memory allocation
torch.cuda.set_device(dist.get_rank())
torch.cuda.memory.set_per_process_memory_fraction(0.9)
```

2. Pipeline Buffers
```python
# Efficient pipeline buffer management
pipeline_buffers = {
    'activation': torch.cuda.caching_allocator.allocate(),
    'gradient': torch.cuda.caching_allocator.allocate()
}
```

3. Resource Cleanup
```python
def cleanup(self):
    """Ensure proper resource deallocation"""
    try:
        self.reset_metrics()
        if self.grad_scaler:
            del self.grad_scaler
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")
```

### Error Handling

1. Metric Collection
```python
try:
    memory_stats = torch.cuda.memory_stats()
    metrics.update({
        'allocated': memory_stats.get('allocated_bytes.all.current', 0),
        'reserved': memory_stats.get('reserved_bytes.all.current', 0)
    })
except Exception as e:
    logger.warning(f"Failed to collect metrics: {e}")
```

2. Resource Management
```python
@contextmanager
def track_step(self, step):
    """Safe step tracking with cleanup"""
    try:
        yield
    finally:
        self.cleanup_step_resources()
```

## Setup and Installation

1. System Requirements
   - NVIDIA GH200 GPU(s)
   - CUDA 12.0+
   - Python 3.8+
   - DeepSpeed 0.10.0+

2. Installation
```bash
# Clone the repository
git clone https://github.com/your-org/diffusion-pipe-gh200
cd diffusion-pipe-gh200

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

3. Configuration
```bash
# Copy and modify example configs
cp examples/hunyuan_video.toml configs/my_training.toml
# Edit paths and parameters as needed
```

## Usage

### Basic Training
```bash
# Single GPU training
python train.py --config configs/hunyuan_video_gh200.toml

# Multi-GPU training
deepspeed train.py --config configs/hunyuan_video_gh200.toml
```

### Advanced Options
```bash
# Resume from checkpoint with memory optimization
python train.py --config configs/hunyuan_video_gh200.toml \
    --resume_from_checkpoint \
    --memory_efficient_training

# Cache only mode for dataset preparation
python train.py --config configs/hunyuan_video_gh200.toml --cache_only
```

## Performance Monitoring

Access monitoring data through TensorBoard:
```bash
tensorboard --logdir outputs/hunyuan_video_gh200
```

Available metrics:
1. Training Metrics
   - Loss curves
   - Learning rates
   - Gradient norms
   - Parameter statistics

2. Memory Metrics
   - Unified memory utilization
   - GPU memory allocation
   - CPU memory usage
   - Page fault statistics

3. Performance Metrics
   - NVLink bandwidth
   - Pipeline efficiency
   - Throughput
   - Step timing

4. Resource Metrics
   - GPU utilization
   - CPU utilization
   - Memory transfer rates
   - Pipeline balance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with:
   - Clear description
   - Test coverage
   - Documentation updates

## License

This project is licensed under [LICENSE]. See the LICENSE file for details.

## Citation

If you use this project in your research, please cite:
```bibtex
@software{diffusion_pipe_gh200,
  title = {Diffusion-Pipe GH200},
  author = {Your Organization},
  year = {2024},
  url = {https://github.com/your-org/diffusion-pipe-gh200},
  description = {High-performance diffusion model training pipeline optimized for NVIDIA GH200}
}
```
