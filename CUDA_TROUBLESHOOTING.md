# CUDA 오류 문제 해결 가이드

## 발생한 오류
```
RuntimeError: CUDA error: unspecified launch failure
```

이 오류는 CUDA 커널 실행에 실패했을 때 발생하며, 주로 다음과 같은 원인들이 있습니다:

## 해결 방법들

### 1. 안전한 실행 스크립트 사용
```bash
python run_with_cuda_debug.py [기존 파라미터들]
```

이 스크립트는 다음을 자동으로 설정합니다:
- `CUDA_LAUNCH_BLOCKING=1`: 정확한 오류 위치 파악
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`: 메모리 분할 최적화
- GPU 상태 확인 및 메모리 모니터링

### 2. 메모리 최적화 설정 적용

#### 자동 메모리 최적화
```python
from memory_optimized_config import apply_memory_optimized_config

# 기존 args에 메모리 최적화 설정 적용
args = apply_memory_optimized_config(args, gpu_name='RTX_3060', available_memory_gb=12)
```

#### 수동 메모리 최적화
다음 파라미터들을 줄여보세요:

```bash
python main.py \
    --batch-size 16 \
    --grad-accumulation-steps 8 \
    --distill-batch-size 8 \
    --use-mixed-precision \
    --window 10 \
    --totwindow 40 \
    --workers 2 \
    [기타 파라미터들]
```

### 3. 단계별 디버깅

#### 1단계: CUDA 상태 확인
```bash
nvidia-smi
```

#### 2단계: 메모리 사용량 확인
```bash
watch -n 1 nvidia-smi
```

#### 3단계: 정확한 오류 위치 찾기
```bash
CUDA_LAUNCH_BLOCKING=1 python main.py [파라미터들]
```

#### 4단계: 응급 설정 적용
```python
from memory_optimized_config import apply_emergency_config
args = apply_emergency_config(args)
```

### 4. 코드 수정 사항

다음과 같은 안전한 메모리 관리 함수들이 추가되었습니다:

- `safe_cuda_empty_cache()`: 안전한 CUDA 캐시 정리
- `check_cuda_health()`: CUDA 상태 확인
- `get_gpu_memory_info()`: GPU 메모리 정보 출력

### 5. 추가 해결 방법들

#### GPU 재시작
```bash
# GPU 드라이버 재시작 (관리자 권한 필요)
sudo systemctl restart nvidia-persistenced
```

#### 온도 확인
```bash
nvidia-smi -q -d TEMPERATURE
```

#### 다른 CUDA 애플리케이션 종료
```bash
# 다른 CUDA 프로세스 확인
nvidia-smi pmon
```

### 6. 환경 변수 설정

다음 환경 변수들을 설정하면 도움이 됩니다:

```bash
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
export CUDNN_DETERMINISTIC=1
```

### 7. 하드웨어 관련 점검

#### GPU 메모리 부족
- 더 작은 batch size 사용
- Gradient accumulation 사용
- Mixed precision training 활성화

#### 하드웨어 문제
- GPU 온도 확인 (80°C 이상이면 위험)
- 전원 공급 확인
- VRAM 오류 테스트

### 8. 권장 설정 (GPU별)

#### RTX 3060/3070 (8GB)
```bash
--batch-size 16 --grad-accumulation-steps 8 --distill-batch-size 8
```

#### RTX 3080/3090 (10GB+)
```bash
--batch-size 24 --grad-accumulation-steps 6 --distill-batch-size 12
```

#### RTX 4070/4080 (12GB+)
```bash
--batch-size 32 --grad-accumulation-steps 4 --distill-batch-size 16
```

#### RTX 4090/A100 (24GB+)
```bash
--batch-size 48 --grad-accumulation-steps 2 --distill-batch-size 24
```

#### A6000 (48GB) - 최적화 설정
```bash
--batch-size 96 --grad-accumulation-steps 1 --distill-batch-size 48 --window 25 --totwindow 100 --workers 8 --imagenet-subset-res 384 --use-mixed-precision
```

### 9. 마지막 해결책

모든 방법이 실패한 경우:

1. 시스템 재부팅
2. 다른 GPU에서 테스트
3. PyTorch 버전 다운그레이드
4. CUDA 드라이버 재설치

### 10. 예방 방법

- 정기적인 `safe_cuda_empty_cache()` 호출
- 메모리 사용량 모니터링
- 적절한 batch size 사용
- Mixed precision training 활용

## A6000 전용 최적화 설정

### A6000 자동 설정 적용
```python
from memory_optimized_config import apply_a6000_optimized_config

# A6000 전용 최적화 설정 적용
args = apply_a6000_optimized_config(args)
```

### A6000 특별 고려사항

#### 1. 메모리 활용 최적화
A6000의 48GB VRAM을 최대한 활용하기 위해:
- 큰 배치 사이즈 (96) 사용
- 높은 해상도 (384) 지원
- 많은 워커 프로세스 (8) 활용

#### 2. 성능 최적화
```bash
# A6000 최적화 명령어 예시
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

#### 3. 메모리 모니터링
A6000은 메모리가 많아도 모니터링이 중요합니다:
```bash
watch -n 1 "nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv"
```

#### 4. 온도 관리
고성능 작업 시 온도 모니터링:
```bash
nvidia-smi -q -d TEMPERATURE
```

## 문제가 계속 발생하면

1. GPU 하드웨어 문제일 수 있습니다
2. 메모리 최적화 설정을 더 공격적으로 적용해보세요
3. 다른 데이터셋으로 테스트해보세요

이 가이드를 따라하면 대부분의 CUDA 오류를 해결할 수 있습니다! 