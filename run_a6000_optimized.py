#!/usr/bin/env python3
"""
A6000 최적화 실행 스크립트
NVIDIA A6000 GPU의 48GB VRAM을 최대한 활용하는 최적화된 설정으로 PoDD를 실행합니다.
"""

import os
import sys
import subprocess
import torch
import time
from memory_optimized_config import apply_a6000_optimized_config


def setup_a6000_environment():
    """A6000에 최적화된 환경 변수 설정"""
    env_vars = {
        'CUDA_LAUNCH_BLOCKING': '1',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'CUDA_VISIBLE_DEVICES': '0',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:1024',  # A6000은 더 큰 분할 사이즈 사용
        'CUDNN_DETERMINISTIC': '1',
        'CUDNN_BENCHMARK': '1',  # A6000은 benchmark 모드 활성화
        'NCCL_DEBUG': 'WARN',
        'TORCH_CUDNN_V8_API_ENABLED': '1',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("A6000 최적화 환경 변수 설정 완료:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")


def check_a6000_status():
    """A6000 GPU 상태 확인"""
    if not torch.cuda.is_available():
        print("❌ CUDA가 사용 불가능합니다!")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"🔍 감지된 GPU 수: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / 1024**3  # GB
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        
        print(f"🖥️  GPU {i}: {props.name}")
        print(f"    전체 메모리: {memory_total:.2f}GB")
        print(f"    할당된 메모리: {memory_allocated:.2f}GB")
        print(f"    예약된 메모리: {memory_reserved:.2f}GB")
        print(f"    사용 가능한 메모리: {memory_total - memory_reserved:.2f}GB")
        
        # A6000 여부 확인
        if 'A6000' in props.name:
            print(f"✅ A6000 GPU 감지됨! 최적화 설정을 적용합니다.")
            return True
        elif memory_total >= 40:  # 40GB 이상이면 A6000 급으로 간주
            print(f"🔥 대용량 GPU 감지됨! A6000 수준의 최적화 설정을 적용합니다.")
            return True
    
    print("⚠️  A6000이 감지되지 않았습니다. 일반 최적화 설정을 사용합니다.")
    return False


def monitor_a6000_performance():
    """A6000 성능 모니터링"""
    print("\n📊 A6000 성능 모니터링 시작...")
    
    try:
        # GPU 온도 확인
        temp_cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        temp_result = subprocess.run(temp_cmd, shell=True, capture_output=True, text=True)
        if temp_result.returncode == 0:
            temp = int(temp_result.stdout.strip())
            print(f"🌡️  GPU 온도: {temp}°C")
            if temp > 80:
                print("🔥 경고: GPU 온도가 높습니다! 쿨링을 확인하세요.")
        
        # 전력 소비 확인
        power_cmd = "nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits"
        power_result = subprocess.run(power_cmd, shell=True, capture_output=True, text=True)
        if power_result.returncode == 0:
            power_info = power_result.stdout.strip().split(', ')
            power_draw = float(power_info[0])
            power_limit = float(power_info[1])
            print(f"⚡ 전력 소비: {power_draw:.1f}W / {power_limit:.1f}W ({power_draw/power_limit*100:.1f}%)")
        
        # 메모리 사용량 확인
        mem_cmd = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
        mem_result = subprocess.run(mem_cmd, shell=True, capture_output=True, text=True)
        if mem_result.returncode == 0:
            mem_info = mem_result.stdout.strip().split(', ')
            mem_used = int(mem_info[0])
            mem_total = int(mem_info[1])
            print(f"💾 메모리 사용량: {mem_used}MB / {mem_total}MB ({mem_used/mem_total*100:.1f}%)")
        
    except Exception as e:
        print(f"⚠️  모니터링 중 오류 발생: {e}")


def run_with_a6000_optimization():
    """A6000 최적화 설정으로 PoDD 실행"""
    try:
        # 메인 모듈 임포트
        import main
        
        # A6000 최적화 설정 적용
        print("\n🚀 A6000 최적화 설정 적용 중...")
        
        # 기본 args 파싱
        args = main.parse_args() if hasattr(main, 'parse_args') else None
        
        if args is None:
            print("❌ 인수 파싱 실패. 기본 설정 사용.")
            # 기본 설정 생성
            import argparse
            parser = argparse.ArgumentParser()
            args = parser.parse_args()
        
        # A6000 최적화 설정 적용
        args = apply_a6000_optimized_config(args)
        
        # 성능 모니터링
        monitor_a6000_performance()
        
        print("\n🎯 A6000 최적화 설정으로 훈련 시작...")
        print("="*60)
        
        # 훈련 시작
        main.main()
        
    except RuntimeError as e:
        if "CUDA" in str(e) or "out of memory" in str(e):
            print(f"\n{'='*60}")
            print("🚨 A6000 CUDA 오류 발생!")
            print(f"{'='*60}")
            print(f"오류: {e}")
            print("\nA6000 전용 해결책:")
            print("1. 다른 프로세스가 GPU를 사용하고 있는지 확인:")
            print("   nvidia-smi")
            print("2. GPU 메모리 완전 정리:")
            print("   sudo nvidia-smi --gpu-reset")
            print("3. 배치 사이즈 조정 (현재 96에서 64로):")
            print("   --batch-size 64 --distill-batch-size 32")
            print("4. 해상도 낮추기:")
            print("   --imagenet-subset-res 256")
            print("5. 워커 수 줄이기:")
            print("   --workers 4")
            print(f"{'='*60}")
        else:
            print(f"일반 오류: {e}")
        
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자가 훈련을 중단했습니다.")
        sys.exit(0)
        
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("🔥 A6000 최적화 PoDD 실행 스크립트")
    print("=" * 50)
    
    # A6000 환경 설정
    setup_a6000_environment()
    
    # A6000 상태 확인
    if not check_a6000_status():
        response = input("\nA6000이 감지되지 않았습니다. 계속하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("실행을 중단합니다.")
            sys.exit(0)
    
    # A6000 최적화 실행
    run_with_a6000_optimization()
    
    print("\n🎉 A6000 최적화 훈련이 완료되었습니다!") 