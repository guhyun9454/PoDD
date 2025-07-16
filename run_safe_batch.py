#!/usr/bin/env python3
"""
안전한 배치 사이즈로 PoDD 실행하는 스크립트
CuDNN 오류를 방지하기 위해 배치 사이즈를 자동으로 조정합니다.
"""

import os
import sys
import re
import subprocess
import torch


def parse_batch_size_from_args(args):
    """명령줄 인수에서 배치 사이즈 파싱"""
    batch_size = None
    distill_batch_size = None
    
    for i, arg in enumerate(args):
        if arg.startswith('--batch_size=') or arg.startswith('--batch-size='):
            batch_size = int(arg.split('=')[1])
        elif arg.startswith('--distill_batch_size=') or arg.startswith('--distill-batch-size='):
            distill_batch_size = int(arg.split('=')[1])
        elif arg == '--batch_size' or arg == '--batch-size':
            if i + 1 < len(args):
                batch_size = int(args[i + 1])
        elif arg == '--distill_batch_size' or arg == '--distill-batch-size':
            if i + 1 < len(args):
                distill_batch_size = int(args[i + 1])
    
    return batch_size, distill_batch_size


def adjust_batch_size_for_cudnn(batch_size, distill_batch_size):
    """CuDNN 호환성을 위한 배치 사이즈 조정"""
    # CuDNN이 안전하게 처리할 수 있는 최대 배치 사이즈 
    max_safe_batch_size = 256
    max_safe_distill_batch_size = 64
    
    adjusted_batch_size = min(batch_size, max_safe_batch_size) if batch_size else 128
    adjusted_distill_batch_size = min(distill_batch_size, max_safe_distill_batch_size) if distill_batch_size else 32
    
    # 8의 배수로 조정 (GPU 효율성을 위해)
    adjusted_batch_size = (adjusted_batch_size // 8) * 8
    adjusted_distill_batch_size = (adjusted_distill_batch_size // 8) * 8
    
    return adjusted_batch_size, adjusted_distill_batch_size


def modify_args_with_safe_batch_size(args):
    """안전한 배치 사이즈로 인수 수정"""
    batch_size, distill_batch_size = parse_batch_size_from_args(args)
    
    print(f"📊 원본 배치 사이즈: batch_size={batch_size}, distill_batch_size={distill_batch_size}")
    
    # 안전한 배치 사이즈로 조정
    safe_batch_size, safe_distill_batch_size = adjust_batch_size_for_cudnn(batch_size, distill_batch_size)
    
    print(f"✅ 안전한 배치 사이즈: batch_size={safe_batch_size}, distill_batch_size={safe_distill_batch_size}")
    
    # 인수 리스트에서 배치 사이즈 교체
    modified_args = []
    skip_next = False
    
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
            
        if arg.startswith('--batch_size=') or arg.startswith('--batch-size='):
            modified_args.append(f"--batch_size={safe_batch_size}")
        elif arg.startswith('--distill_batch_size=') or arg.startswith('--distill-batch-size='):
            modified_args.append(f"--distill_batch_size={safe_distill_batch_size}")
        elif arg == '--batch_size' or arg == '--batch-size':
            modified_args.append(arg)
            modified_args.append(str(safe_batch_size))
            skip_next = True
        elif arg == '--distill_batch_size' or arg == '--distill-batch-size':
            modified_args.append(arg)
            modified_args.append(str(safe_distill_batch_size))
            skip_next = True
        else:
            modified_args.append(arg)
    
    return modified_args


def setup_cudnn_safe_environment():
    """CuDNN 안전 환경 설정"""
    env_vars = {
        'CUDA_LAUNCH_BLOCKING': '1',
        'CUDNN_DETERMINISTIC': '0',
        'CUDNN_BENCHMARK': '1',
        'CUDNN_ALLOW_TF32': '1',
        'CUDNN_CONV_ALGO_SEARCH': 'HEURISTIC',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,roundup_power2_divisions:16',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("🔧 CuDNN 안전 환경 설정 완료:")
    for key, value in env_vars.items():
        print(f"   {key}={value}")


def run_with_safe_batch_size():
    """안전한 배치 사이즈로 실행"""
    print("🛡️  안전한 배치 사이즈로 PoDD 실행")
    print("=" * 50)
    
    # 환경 설정
    setup_cudnn_safe_environment()
    
    # 명령줄 인수 수정
    original_args = sys.argv[1:]
    safe_args = modify_args_with_safe_batch_size(original_args)
    
    print(f"\n📝 수정된 인수:")
    for arg in safe_args:
        print(f"   {arg}")
    
    # main.py 실행
    cmd = ['python', 'main.py'] + safe_args
    
    print(f"\n🚀 실행 명령어:")
    print(f"   {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # subprocess로 실행
        result = subprocess.run(cmd, check=True)
        print("\n✅ 훈련이 성공적으로 완료되었습니다!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 훈련 중 오류 발생:")
        print(f"   종료 코드: {e.returncode}")
        
        # 더 작은 배치 사이즈로 재시도 제안
        print("\n🔄 더 작은 배치 사이즈로 재시도해보세요:")
        batch_size, distill_batch_size = parse_batch_size_from_args(safe_args)
        smaller_batch = max(16, batch_size // 2) if batch_size else 64
        smaller_distill = max(4, distill_batch_size // 2) if distill_batch_size else 16
        
        print(f"   --batch_size {smaller_batch} --distill_batch_size {smaller_distill}")
        
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자가 훈련을 중단했습니다.")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python run_safe_batch.py [main.py 인수들...]")
        print("예시: python run_safe_batch.py --batch_size=600 --distill_batch_size=32 [기타 인수들...]")
        sys.exit(1)
    
    run_with_safe_batch_size() 