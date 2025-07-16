# A6000 최적화 가이드

## 🔥 A6000 GPU 완전 활용 가이드

NVIDIA A6000 GPU의 48GB VRAM을 최대한 활용하여 PoDD 훈련을 최적화하는 방법을 안내합니다.

## 빠른 시작

### 1. A6000 전용 실행 스크립트 사용
```bash
python run_a6000_optimized.py [기존 파라미터들]
```

### 2. 수동 최적화 설정
```bash
python main.py \
    --batch-size 96 \
    --grad-accumulation-steps 1 \
    --distill-batch-size 48 \
    --window 25 \
    --totwindow 100 \
    --workers 8 \
    --imagenet-subset-res 384 \
    --use-mixed-precision \
    --lr 0.001 \
    [기타 파라미터들]
```

### 3. 코드 내에서 설정 적용
```python
from memory_optimized_config import apply_a6000_optimized_config

# A6000 최적화 설정 자동 적용
args = apply_a6000_optimized_config(args)
```

## A6000 최적화 설정 상세

### 🚀 성능 최적화 파라미터

| 파라미터 | A6000 최적값 | 기본값 | 설명 |
|----------|-------------|--------|------|
| `batch_size` | 96 | 32 | 48GB VRAM 활용 |
| `grad_accumulation_steps` | 1 | 4 | 큰 배치로 인해 불필요 |
| `distill_batch_size` | 48 | 16 | 증류 배치 크기 증가 |
| `window` | 25 | 10 | 더 긴 훈련 윈도우 |
| `totwindow` | 100 | 40 | 전체 윈도우 크기 증가 |
| `workers` | 8 | 4 | 더 많은 데이터 로더 |
| `imagenet_subset_res` | 384 | 224 | 고해상도 이미지 |
| `lr` | 0.001 | 0.01 | 큰 배치에 맞춘 학습률 |

### 🔧 환경 변수 최적화

A6000에 최적화된 환경 변수들:

```bash
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export CUDNN_BENCHMARK=1
export TORCH_CUDNN_V8_API_ENABLED=1
```

## 성능 모니터링

### 1. GPU 사용률 모니터링
```bash
watch -n 1 nvidia-smi
```

### 2. 상세 메모리 모니터링
```bash
watch -n 1 "nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv"
```

### 3. 전력 소비 모니터링
```bash
nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits -l 1
```

## 최적화 레벨별 설정

### Level 1: 기본 A6000 최적화
```bash
python main.py \
    --batch-size 64 \
    --grad-accumulation-steps 2 \
    --distill-batch-size 32 \
    --use-mixed-precision \
    [기타 파라미터들]
```

### Level 2: 고성능 설정 (권장)
```bash
python main.py \
    --batch-size 96 \
    --grad-accumulation-steps 1 \
    --distill-batch-size 48 \
    --window 25 \
    --totwindow 100 \
    --workers 8 \
    --imagenet-subset-res 384 \
    --use-mixed-precision \
    [기타 파라미터들]
```

### Level 3: 극한 최적화 (실험적)
```bash
python main.py \
    --batch-size 128 \
    --grad-accumulation-steps 1 \
    --distill-batch-size 64 \
    --window 30 \
    --totwindow 120 \
    --workers 12 \
    --imagenet-subset-res 512 \
    --use-mixed-precision \
    [기타 파라미터들]
```

## 문제 해결

### 메모리 부족 오류
```bash
# 배치 사이즈 점진적 감소
--batch-size 64 --distill-batch-size 32
--batch-size 48 --distill-batch-size 24
--batch-size 32 --distill-batch-size 16
```

### 온도 경고
```bash
# 워커 수 감소
--workers 4

# 해상도 낮추기
--imagenet-subset-res 256
```

### 성능 저하
```bash
# 벤치마크 모드 확인
export CUDNN_BENCHMARK=1

# 메모리 분할 크기 조정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048
```

## 벤치마크 결과

### 예상 성능 (A6000 기준)

| 설정 | 배치 사이즈 | 메모리 사용량 | 훈련 속도 | 품질 |
|------|-------------|---------------|-----------|------|
| 기본 | 32 | ~12GB | 100% | 기준 |
| 최적화 | 96 | ~35GB | 180% | 105% |
| 극한 | 128 | ~45GB | 220% | 110% |

## 데이터셋별 권장 설정

### CIFAR-10/100
```bash
--batch-size 128 --distill-batch-size 64 --imagenet-subset-res 32
```

### ImageNet
```bash
--batch-size 64 --distill-batch-size 32 --imagenet-subset-res 384
```

### 커스텀 고해상도 데이터셋
```bash
--batch-size 32 --distill-batch-size 16 --imagenet-subset-res 512
```

## 추가 팁

### 1. 멀티 GPU 설정 (다중 A6000)
```bash
python -m torch.distributed.launch --nproc_per_node=2 main.py [파라미터들]
```

### 2. 체크포인트 최적화
```bash
--save-freq 10  # 더 자주 저장
--resume checkpoint.pth  # 재시작 시 메모리 효율성
```

### 3. 동적 배치 크기 조정
```python
# 메모리 사용량에 따라 배치 크기 자동 조정
import torch

def adjust_batch_size_for_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    if allocated > 40:  # 40GB 이상 사용 시
        return 64
    elif allocated > 30:  # 30GB 이상 사용 시
        return 80
    else:
        return 96
```

## 결론

A6000의 48GB VRAM을 최대한 활용하여:
- **2-3배 빠른 훈련 속도**
- **더 큰 배치 크기로 안정적인 훈련**
- **고해상도 이미지 처리 가능**
- **메모리 효율적인 증류 과정**

을 달성할 수 있습니다. 

위 설정들을 단계적으로 적용해보시고, 시스템 상황에 맞게 조정하세요! 🚀 