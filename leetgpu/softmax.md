# Softmax CUDA 커널 (Softmax)

> 테마: GPU 프로그래밍 (CUDA/LeetGPU) · 개념 정리

## 한 줄 요약
입력 벡터를 확률 분포로 변환하는 softmax를, 수치 안정성을 위해 max를 빼고 병렬 reduction으로 max·sum을 구해 계산하는 CUDA 커널.

## 핵심 개념
- **softmax**: `softmax(x_i) = exp(x_i) / Σ_j exp(x_j)`. 출력은 모두 양수이고 합이 1인 확률 분포.
- **수치 안정성(max 빼기)**: `exp`는 큰 입력에서 쉽게 overflow한다. 모든 원소에서 최댓값 `m`을 빼도 결과가 동일하므로 `exp(x_i - m)`으로 계산한다.
- **병렬 reduction**: max와 sum처럼 벡터 전체를 하나의 값으로 줄이는 연산. 트리 형태로 반씩 접어(stride를 절반씩 줄여) `O(log n)` 단계에 수행.
- **두 단계 구조**: ① reduce로 max·sum을 구하고 ② 각 원소를 `exp(x_i - m)/sum`으로 normalize.

## 원리 / 수식
- 안정화 항등식: `exp(x_i) / Σ exp(x_j) = exp(x_i - m) / Σ exp(x_j - m)` (`m = max_j x_j`).
- reduction 단계 수는 `log2(n)`. 각 단계에서 스레드 절반이 비활성화되며 `__syncthreads()`로 단계 간 동기화.
- sum reduction도 max와 동일 패턴이되, 사전에 `exp(x_i - m)`을 계산해 둔 값들을 더한다.

## CUDA 구현 포인트
```cuda
__global__ void softmax(const float* x, float* y, int n) {
    extern __shared__ float s[];          // 블록 크기만큼 동적 shared memory
    int tid = threadIdx.x;
    // 1단계: max reduction
    s[tid] = (tid < n) ? x[tid] : -INFINITY;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
        __syncthreads();
    }
    float m = s[0];
    __syncthreads();
    // 2단계: exp(x-m) 후 sum reduction
    float e = (tid < n) ? expf(x[tid] - m) : 0.0f;
    s[tid] = e;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    // 3단계: normalize
    if (tid < n) y[tid] = e / s[0];
}
```
- 벡터가 블록 하나에 안 들어가면 grid 단위 reduction(블록별 부분 max/sum → 2차 reduce)으로 확장한다.

## 자주 하는 실수 / 팁
- max 빼기를 생략하면 큰 입력에서 `expf`가 `inf`가 되어 결과가 NaN이 된다(수치 안정성 필수).
- reduction 루프 안의 `__syncthreads()`를 빠뜨리면 아직 갱신 안 된 값을 읽어 잘못된 합이 나온다.
- `if (tid < stride)` 분기 안에 `__syncthreads()`를 넣으면 deadlock — 배리어는 분기 밖에 둔다.
- 블록 크기는 2의 거듭제곱으로 두면 stride 반감 reduction이 깔끔하다. n이 그보다 작으면 패딩(max는 `-INFINITY`, sum은 `0`).
- max와 e를 쓰는 두 reduction 사이에 `s`를 재사용하므로 `m`을 레지스터에 보관한 뒤 `__syncthreads()`로 덮어쓰기 안전성을 확보한다.

## 더 보기
- LeetGPU 문제 (Softmax): https://leetgpu.com/challenges/softmax
- CUDA Parallel Reduction (Mark Harris): https://developer.nvidia.com/blog/
- 비교: [`tiled_matrix_multiplication.md`](./tiled_matrix_multiplication.md) — shared memory 타일링
