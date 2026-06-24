# 타일 행렬 곱 (Tiled Matrix Multiplication)

> 테마: GPU 프로그래밍 (CUDA/LeetGPU) · 개념 정리

## 한 줄 요약
공유 메모리(shared memory)에 행렬 일부(타일)를 올려 재사용함으로써, 느린 global memory 접근 횟수를 줄이는 행렬 곱 최적화 기법.

## 핵심 개념
- **메모리 대역폭 병목**: 기본 matmul은 결과 한 원소를 위해 A의 한 행과 B의 한 열을 매번 global memory에서 읽는다. 같은 데이터를 여러 스레드가 중복 로드해 대역폭이 병목이 된다.
- **shared memory**: 블록 내 스레드가 공유하는 온칩 메모리. global memory보다 훨씬 빠르고 latency가 낮다.
- **타일링(tiling)**: A·B를 `TILE x TILE` 크기 블록으로 나눠, 한 타일을 shared memory에 한 번 적재하고 블록 내 스레드들이 반복 재사용한다.
- **`__syncthreads()`**: 블록 내 모든 스레드가 타일 적재를 끝낸 뒤 계산을 시작하도록(그리고 계산 후 다음 타일 적재 전에) 동기화하는 배리어.

## 원리 / 수식
- `C[i][j] = Σ_k A[i][k] · B[k][j]`. 합을 `TILE` 단위로 쪼개 부분합을 누적한다.
- 한 원소당 global 읽기 횟수가 기본 방식의 `2N`에서 타일링 시 약 `2N/TILE`로 줄어, **계산-메모리 비율(arithmetic intensity)**이 TILE 배만큼 향상된다.
- 각 스레드는 자기 타일 부분합을 레지스터(`acc`)에 누적하고, 모든 타일 처리 후 한 번만 `C`에 기록한다.

## CUDA 구현 포인트
```cuda
#define TILE 16
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // 경계 검사 후 타일을 shared memory로 적재 (범위 밖은 0)
        As[threadIdx.y][threadIdx.x] = (row < M && t*TILE+threadIdx.x < K)
            ? A[row*K + t*TILE + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && t*TILE+threadIdx.y < K)
            ? B[(t*TILE+threadIdx.y)*N + col] : 0.0f;
        __syncthreads();                       // 적재 완료 대기
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                       // 다음 타일 적재 전 계산 완료 대기
    }
    if (row < M && col < N) C[row*N + col] = acc;
}
```

## 자주 하는 실수 / 팁
- `__syncthreads()`를 빼면 적재 전 데이터로 계산하거나, 한 스레드가 다음 타일을 덮어써 race condition이 생긴다. 두 위치 모두 필요.
- `__syncthreads()`를 `if (row < M ...)` 같은 발산 분기 안에 넣으면 일부 스레드가 배리어에 도달하지 못해 deadlock. 항상 모든 스레드가 동일하게 실행하도록 둔다.
- K가 TILE의 배수가 아니거나 행렬이 정사각이 아니면 경계 검사로 범위 밖을 0 패딩해야 한다.
- TILE 크기는 shared memory 용량·점유율(occupancy)과 트레이드오프(16 또는 32가 일반적).

## 더 보기
- LeetGPU 문제 (Matrix Multiplication을 타일링으로 최적화): https://leetgpu.com/challenges/matrix-multiplication
- CUDA C++ Programming Guide — Shared Memory: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 비교: [`softmax.md`](./softmax.md) — 병렬 reduction이 쓰이는 커널
