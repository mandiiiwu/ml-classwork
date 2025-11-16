"""
GPU Utility Functions for ML Class
Helps students easily select and use available NVIDIA GPUs
"""

import torch
import subprocess
from typing import Optional


def get_available_gpu() -> Optional[int]:
    """
    Finds the GPU with the most available memory using PyTorch.

    Returns:
        int: GPU index of the most available GPU, or None if no GPUs available
    """
    if not torch.cuda.is_available():
        print("No CUDA GPUs available on this system")
        return None

    num_gpus = torch.cuda.device_count()
    gpu_info = []

    for i in range(num_gpus):
        # Get memory stats for each GPU
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**2)  # Convert to MB

        # Get currently allocated memory
        allocated = torch.cuda.memory_allocated(i) / (1024**2)  # Convert to MB
        reserved = torch.cuda.memory_reserved(i) / (1024**2)  # Convert to MB

        # Free memory is total minus reserved (reserved includes allocated)
        free_memory = total_memory - reserved

        gpu_info.append((i, free_memory, total_memory))

    if not gpu_info:
        return None

    # Sort by free memory (descending) and return the GPU with most free memory
    gpu_info.sort(key=lambda x: x[1], reverse=True)
    best_gpu = gpu_info[0]

    print(
        f"Selected GPU {best_gpu[0]}: {best_gpu[1]:.0f}MB free out of {best_gpu[2]:.0f}MB"
    )
    return best_gpu[0]


def setup_device(gpu_id: Optional[int] = None, verbose: bool = True) -> torch.device:
    """
    Sets up and returns a PyTorch device for model training.

    This function automatically selects the best available GPU or falls back to CPU.
    Students can simply call this function and move their model to the returned device.

    Args:
        gpu_id: Specific GPU ID to use. If None, automatically selects the GPU with most free memory.
        verbose: If True, prints information about the selected device.

    Returns:
        torch.device: PyTorch device object ready to use

    Example:
        >>> device = setup_device()
        >>> model = MyModel()
        >>> model = model.to(device)
        >>>
        >>> # Or in one line:
        >>> model = MyModel().to(setup_device())
    """
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available. Using CPU.")
        return torch.device("cpu")

    # If specific GPU requested, use it
    if gpu_id is not None:
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus or gpu_id < 0:
            print(f"Warning: GPU {gpu_id} not available. {num_gpus} GPUs detected.")
            gpu_id = get_available_gpu()
        elif verbose:
            print(f"Using requested GPU {gpu_id}")
    else:
        # Auto-select best GPU
        gpu_id = get_available_gpu()

    if gpu_id is None:
        if verbose:
            print("No GPU available. Using CPU.")
        return torch.device("cpu")

    device = torch.device(f"cuda:{gpu_id}")

    if verbose:
        print(f"Device set to: {device}")
        print(f"GPU Name: {torch.cuda.get_device_name(gpu_id)}")

    return device


def print_gpu_status():
    """
    Prints the status of all available GPUs on the system.
    Useful for students to see all GPU availability before training.
    """
    if not torch.cuda.is_available():
        print("No CUDA GPUs available on this system")
        return

    print(f"\n{'='*70}")
    print("GPU Status Report")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print(
            f"{'GPU':<5} {'Name':<20} {'Memory Used':<15} {'Memory Total':<15} {'GPU Util':<10}"
        )
        print(f"{'-'*70}")

        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 5:
                idx = parts[0].strip()
                name = parts[1].strip()
                used = parts[2].strip()
                total = parts[3].strip()
                util = parts[4].strip()
                print(
                    f"{idx:<5} {name:<20} {used:>6} MB{' '*6} {total:>6} MB{' '*6} {util:>3}%"
                )

        print(f"{'='*70}\n")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Could not query GPU status: {e}")
        print(f"Total GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


if __name__ == "__main__":
    # Demo usage
    print("GPU Utility Demo\n")
    print_gpu_status()

    print("\nSetting up device:")
    device = setup_device()
    print(f"\nYou can now use: model.to({device})")
    
    # Example Usage:
    """
        from gpu_utils import setup_device, print_gpu_status
        # Check what GPUs are available
        print_gpu_status()
        # Get the best available device
        device = setup_device()
        # Use it with your model
        model = MyModel().to(device)
        data = data.to(device)
    """
