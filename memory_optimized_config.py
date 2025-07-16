"""
Memory-optimized configuration for PoDD training
Use these settings to reduce memory usage and prevent CUDA errors
"""

# Memory-optimized training parameters
MEMORY_OPTIMIZED_ARGS = {
    # Reduce batch size to prevent memory overflow
    'batch_size': 32,  # Reduced from default (usually 64 or 128)
    
    # Enable gradient accumulation to maintain effective batch size
    'grad_accumulation_steps': 4,  # Effective batch size = batch_size * grad_accumulation_steps
    
    # Enable mixed precision training for memory efficiency
    'use_mixed_precision': True,
    
    # Reduce distillation parameters
    'distill_batch_size': 16,  # Reduced from default
    'window': 10,  # Reduced from default (usually 20)
    'totwindow': 40,  # Reduced from default
    
    # Reduce image resolution if using ImageNet
    'imagenet_subset_res': 224,  # Use 224 instead of 256 or higher
    
    # Reduce number of workers to save memory
    'workers': 2,  # Reduced from default (usually 4-8)
    
    # Reduce print frequency to avoid memory accumulation
    'print_freq': 100,  # Increased from default
    
    # Memory management settings
    'eps': 1e-6,  # Smaller epsilon to prevent numerical issues
    'wd': 1e-4,   # Weight decay
}

# GPU-specific memory limits (in GB)
GPU_MEMORY_LIMITS = {
    'RTX_3060': 12,
    'RTX_3070': 8,
    'RTX_3080': 10,
    'RTX_3090': 24,
    'RTX_4060': 8,
    'RTX_4070': 12,
    'RTX_4080': 16,
    'RTX_4090': 24,
    'V100': 16,
    'A100': 40,
    'A6000': 48,  # NVIDIA A6000 - 48GB VRAM
    'T4': 15,
}

def get_memory_optimized_config(gpu_name=None, available_memory_gb=None):
    """
    Get memory-optimized configuration based on GPU
    
    Args:
        gpu_name: Name of GPU (e.g., 'RTX_3060')
        available_memory_gb: Available GPU memory in GB
    
    Returns:
        Dictionary with optimized parameters
    """
    config = MEMORY_OPTIMIZED_ARGS.copy()
    
    # Determine available memory
    if available_memory_gb is None:
        if gpu_name and gpu_name in GPU_MEMORY_LIMITS:
            available_memory_gb = GPU_MEMORY_LIMITS[gpu_name]
        else:
            available_memory_gb = 8  # Conservative default
    
    # Adjust parameters based on available memory
    if available_memory_gb < 8:
        # Very limited memory (< 8GB)
        config.update({
            'batch_size': 16,
            'grad_accumulation_steps': 8,
            'distill_batch_size': 8,
            'window': 5,
            'totwindow': 20,
            'workers': 1,
            'imagenet_subset_res': 192,
        })
        print(f"[CONFIG] Using ultra-low memory config for {available_memory_gb}GB GPU")
        
    elif available_memory_gb < 12:
        # Medium memory (8-12GB)
        config.update({
            'batch_size': 24,
            'grad_accumulation_steps': 6,
            'distill_batch_size': 12,
            'window': 8,
            'totwindow': 30,
            'workers': 2,
            'imagenet_subset_res': 224,
        })
        print(f"[CONFIG] Using low memory config for {available_memory_gb}GB GPU")
        
    elif available_memory_gb < 16:
        # Good memory (12-16GB)
        config.update({
            'batch_size': 32,
            'grad_accumulation_steps': 4,
            'distill_batch_size': 16,
            'window': 10,
            'totwindow': 40,
            'workers': 3,
            'imagenet_subset_res': 256,
        })
        print(f"[CONFIG] Using medium memory config for {available_memory_gb}GB GPU")
        
    elif available_memory_gb < 32:
        # High memory (16-32GB)
        config.update({
            'batch_size': 48,
            'grad_accumulation_steps': 2,
            'distill_batch_size': 24,
            'window': 15,
            'totwindow': 60,
            'workers': 4,
            'imagenet_subset_res': 256,
        })
        print(f"[CONFIG] Using high memory config for {available_memory_gb}GB GPU")
        
    else:
        # Ultra high memory (32GB+) - A6000, A100 등
        config.update({
            'batch_size': 80,
            'grad_accumulation_steps': 1,
            'distill_batch_size': 40,
            'window': 20,
            'totwindow': 80,
            'workers': 6,
            'imagenet_subset_res': 320,
        })
        print(f"[CONFIG] Using ultra-high memory config for {available_memory_gb}GB GPU")
    
    return config


def apply_memory_optimized_config(args, gpu_name=None, available_memory_gb=None):
    """
    Apply memory-optimized configuration to argument namespace
    
    Args:
        args: Argument namespace from argparse
        gpu_name: Name of GPU
        available_memory_gb: Available GPU memory in GB
    """
    config = get_memory_optimized_config(gpu_name, available_memory_gb)
    
    for key, value in config.items():
        if hasattr(args, key):
            old_value = getattr(args, key)
            setattr(args, key, value)
            print(f"[CONFIG] {key}: {old_value} -> {value}")
        else:
            print(f"[CONFIG] Adding new parameter: {key} = {value}")
            setattr(args, key, value)
    
    return args


# Emergency memory reduction settings (use when all else fails)
EMERGENCY_CONFIG = {
    'batch_size': 8,
    'grad_accumulation_steps': 16,
    'distill_batch_size': 4,
    'window': 3,
    'totwindow': 15,
    'workers': 1,
    'imagenet_subset_res': 128,
    'use_mixed_precision': True,
    'print_freq': 200,
}

# A6000 전용 최적화 설정
A6000_OPTIMIZED_CONFIG = {
    'batch_size': 96,
    'grad_accumulation_steps': 1,
    'distill_batch_size': 48,
    'window': 25,
    'totwindow': 100,
    'workers': 8,
    'imagenet_subset_res': 384,
    'use_mixed_precision': True,
    'print_freq': 50,
    'eps': 1e-8,
    'wd': 1e-4,
    'lr': 0.001,  # A6000은 높은 배치 사이즈로 인해 학습률 조정 필요
}

def apply_a6000_optimized_config(args):
    """Apply A6000-specific optimized settings"""
    print("[CONFIG] Applying A6000-specific optimized settings!")
    print("[CONFIG] A6000 48GB VRAM을 최대한 활용하는 설정을 적용합니다.")
    
    for key, value in A6000_OPTIMIZED_CONFIG.items():
        if hasattr(args, key):
            old_value = getattr(args, key)
            setattr(args, key, value)
            print(f"[A6000] {key}: {old_value} -> {value}")
        else:
            setattr(args, key, value)
            print(f"[A6000] Adding: {key} = {value}")
    
    return args


def apply_emergency_config(args):
    """Apply emergency memory reduction settings"""
    print("[CONFIG] Applying EMERGENCY memory reduction settings!")
    
    for key, value in EMERGENCY_CONFIG.items():
        if hasattr(args, key):
            old_value = getattr(args, key)
            setattr(args, key, value)
            print(f"[EMERGENCY] {key}: {old_value} -> {value}")
        else:
            setattr(args, key, value)
            print(f"[EMERGENCY] Adding: {key} = {value}")
    
    return args 