#!/usr/bin/env python3
"""
Safe execution script for PoDD with CUDA debugging enabled
This script sets appropriate environment variables and provides better error handling
"""

import os
import sys
import subprocess
import signal
import torch


def set_cuda_debug_env():
    """Set CUDA debugging environment variables"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Additional debugging options
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only first GPU
    os.environ['CUDNN_DETERMINISTIC'] = '1'
    os.environ['CUDNN_BENCHMARK'] = '0'
    
    print("CUDA debugging environment variables set:")
    print(f"  CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    print(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")


def check_gpu_status():
    """Check GPU status and memory"""
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return False
    
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / 1024**3  # GB
        memory_free = torch.cuda.memory_reserved(i) / 1024**3  # GB
        print(f"GPU {i}: {props.name}, Total Memory: {memory_total:.2f}GB")
        
        # Check if we have enough memory (recommend at least 8GB free)
        if memory_total < 8:
            print(f"WARNING: GPU {i} has only {memory_total:.2f}GB memory, may not be sufficient")
    
    return True


def run_main_with_error_handling():
    """Run main.py with proper error handling"""
    try:
        # Import and run main
        import main
        
        # Parse command line arguments
        if len(sys.argv) > 1:
            # Pass through command line arguments
            sys.argv = sys.argv[1:]
        
        # Run main
        main.main()
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"\n{'='*60}")
            print("CUDA ERROR DETECTED!")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print("\nSuggested solutions:")
            print("1. Reduce batch size (--batch-size)")
            print("2. Reduce gradient accumulation steps (--grad-accumulation-steps)")
            print("3. Enable mixed precision training (--use-mixed-precision)")
            print("4. Reduce image resolution")
            print("5. Restart with fresh GPU state")
            print("6. Check GPU temperature and hardware health")
            print(f"{'='*60}")
        else:
            print(f"Non-CUDA error: {e}")
        
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Starting PoDD with CUDA debugging enabled...")
    
    # Set debugging environment
    set_cuda_debug_env()
    
    # Check GPU status
    if not check_gpu_status():
        print("GPU check failed, exiting...")
        sys.exit(1)
    
    # Run with error handling
    run_main_with_error_handling() 