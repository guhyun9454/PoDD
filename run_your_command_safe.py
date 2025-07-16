#!/usr/bin/env python3
"""
사용자의 정확한 명령어를 안전한 배치 사이즈로 실행하는 스크립트
"""

import os
import subprocess
import sys

def setup_environment():
    """환경 변수 설정"""
    env_vars = {
        'CUDA_LAUNCH_BLOCKING': '1',
        'CUDA_VISIBLE_DEVICES': '0',
        'CUDNN_DETERMINISTIC': '0',
        'CUDNN_BENCHMARK': '1',
        'CUDNN_ALLOW_TF32': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,roundup_power2_divisions:16',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("🔧 환경 변수 설정 완료")

def run_safe_command():
    """안전한 배치 사이즈로 명령어 실행"""
    setup_environment()
    
    # 원본 명령어 (배치 사이즈 600 -> 256으로 조정)
    cmd = [
        'python', 'main.py',
        '--name=PoDD-ImageNet-nette-fast1.5-convnet5-IPC1',
        '--imagenet_subset',
        '--imagenet_subset_res=128',
        '--imagenet_subset_key=nette',
        '--root=/home/tgwithiu/datasets/ImageNet_nette',
        '--arch=convnet5',
        '--batch_size=256',  # 600 -> 256으로 조정
        '--distill_batch_size=32',  # 그대로 유지
        '--patch_num_x=5',
        '--patch_num_y=4',
        '--poster_width=640',
        '--poster_height=256',
        '--poster_class_num_x=5',
        '--poster_class_num_y=2',
        '--class_area_width=128',
        '--class_area_height=128',
        '--workers=4',
        '--ddtype=curriculum',
        '--cctype=2',
        '--inner_optim=Adam',
        '--outer_optim=Adam',
        '--inner_lr=0.001',
        '--lr=0.001',
        '--window=35',
        '--minwindow=0',
        '--totwindow=100',
        '--num_train_eval=3',
        '--epochs=8000',
        '--test_freq=30',
        '--print_freq=15',
        '--comp_ipc=1',
        '--train_y',
        '--syn_strategy=flip_rotate',
        '--real_strategy=flip_rotate',
        '--update_steps=1',
        '--seed=0'
    ]
    
    print("🚀 안전한 배치 사이즈로 실행:")
    print("   배치 사이즈: 600 -> 256으로 조정")
    print("   증류 배치 사이즈: 32 (그대로)")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ 훈련이 성공적으로 완료되었습니다!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 훈련 중 오류 발생 (종료 코드: {e.returncode})")
        
        # 더 작은 배치 사이즈로 재시도 제안
        print("\n🔄 더 작은 배치 사이즈로 재시도해보세요:")
        print("   python run_your_command_safe.py --batch_size=128")
        print("   python run_your_command_safe.py --batch_size=64")
        
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자가 훈련을 중단했습니다.")
        sys.exit(0)

if __name__ == "__main__":
    run_safe_command()
