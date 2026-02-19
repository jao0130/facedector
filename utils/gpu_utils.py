"""GPU setup utilities for resource management."""

import torch


def setup_gpu(device_str: str = "cuda:0", memory_limit_mb: int = 13000):
    """
    Setup GPU with memory limit.

    Args:
        device_str: CUDA device string.
        memory_limit_mb: Maximum GPU memory in MB (default 13GB = 80% of 16GB).
    """
    if not torch.cuda.is_available():
        print("[GPU] CUDA not available, using CPU")
        return torch.device("cpu")

    device = torch.device(device_str)
    gpu_idx = device.index if device.index is not None else 0

    gpu_name = torch.cuda.get_device_name(gpu_idx)
    total_mem = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 2)
    print(f"[GPU] {gpu_name} | Total: {total_mem:.0f}MB | Limit: {memory_limit_mb}MB")

    # Set memory fraction
    if memory_limit_mb > 0 and total_mem > 0:
        fraction = min(memory_limit_mb / total_mem, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, gpu_idx)

    # Enable cuDNN benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True

    return device


def get_device(cfg) -> torch.device:
    """Get device from config."""
    return setup_gpu(cfg.DEVICE, cfg.GPU_MEMORY_LIMIT_MB)


def print_gpu_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[GPU] Allocated: {allocated:.0f}MB | Reserved: {reserved:.0f}MB")
