---
title: GPGPU
date: 2024-02-21 13:42:27
tags: gpu
category: GPU
---

# 名詞解釋
- CUDA (Compute Unified Device Architecture)
  - 強大的平行計算平台，讓開發者能夠充分利用 NVIDIA GPU 的計算能力(NVIDIA 專用)，進行高效的計算任務處理
  
- OpenCL (Open Computing Language)
  - 用於編寫在多種處理器上運行的程序，包括 CPU、GPU、DSP（數字訊號處理器）和其他類型的處理器，主要用於通用計算，特別是那些可以利用平行計算的任務
  
# 從 GPU 到 GPGPU
CPU 單核心性能的提高受到**功耗**、**存取記憶體速度**、**設計複雜度**等多重瓶頸的限制，而 GPU 僅侷限於處理圖形繪製的計算任務，是極大的資源浪費。

2006 年，NVIDIA 公布了統一著色器架構(unified shader architecture)，從此 GPU 進入了通用計算時代。 傳統的 GPU 通常採用固定比例的頂點著色器和像素著色器單元，但這種作法會導致單元使用率低下的問題。為解決這一問題，統一著色器架構整合了頂點著色器和像素著色器，這種無差別的著色器設計，使 GPU 成為一個多核心的通用處理器。

# 計算模型
### SIMT (Single Instruction Multiple Threads)，單指令多執行緒
  - 一行指令被多個執行緒同時執行，與 SIMD 平行類似，在 GPGPU 中被稱為 SIMT 計算模型
  - ex: 矩陣乘法
```cpp
// 從輸入矩陣 A 和 B 中讀取一部份向量 a, b
for (i = 0; i < N; i++){
  c += a[i] + b[i];
}
// 將 c 寫回結果矩陣 C 的對應位置中
```
  - CUDA 為 SIMT 計算模型引入 thread grid、thread block、thread，對等地，OpenCL 為 SIMT 計算模型引入 NDRange、work-group、work-item

### 裝置端和核心函數
在 CUDA 和 OpenCL 模型中，會把程式劃分成**主機端 (host)** 和**裝置端 (device)** ，分別在 CPU 和 GPGPU 上執行。 CPU 硬體執行主機端程式，GPGPU 硬體將根據程式設計人員給定的執行緒網格 (上面提到的 thread grid) 組織方式等參數，將裝置端程式進一步分發到執行緒中。每個執行緒執行相同的程式，但是是不同的資料。

以上面的矩陣乘法為例，主機端程式分成三個步驟：

#### 資料複製
- CPU 將主記憶體資料複製到 GPGPU。主機端程式會先完成 GPGPU 的待處理資料宣告和前置處理，然後 CPU 呼叫 API 對 GPGPU 進行初始化和控制。
```c
// 主記憶體的資料
float A[M * N], B[N * K], C[M * K];
// GPGPU 裝置端全域記憶體
float* d_A, * d_B, * d_C;

int size = M * N * sizeof(float);
// CPU 呼叫 API 分配裝置端空間 
cudaMalloc((void**)& d_A, size);
// CPU 呼叫 API 控制 CPU 和 GPGPU 之間的通訊
// 將資料從主機端記憶體複製到 GPGPU 全域記憶體裡面
cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

size = N * K * sizeof(float);
cudaMalloc((void**)& d_B, size);
cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

size = M * K * sizeof(float);
cudaMalloc((void**)& d_C, size);
```

#### GPGPU 啟動
- CPU 喚醒 GPGPU 執行緒進行運算，並將執行緒的組織方式和參數傳入 GPGPU 中。
``` c
unsigned T_size = 16;
dim3 gridDim(M / T_size, K / T_size, 1);
dim3 blockDim(T_size, T_size, 1);

/// 喚醒對應的裝置端程式
/// 啟動名為 basic_mul 的裝置端函數
basic_mul <<< gridDim, blockDim >>> (d_A, d_B, d_C);

// 因為 CPU 和 GPGPU 是非同步執行，要使用此函數讓他們同步
// 不然可能 CPU 還沒等到 GPGPU 算完就繼續跑
cudaDeviceSynchronize();
```

#### 資料寫回
- GPGPU 運算完畢，並將結果寫回主機端記憶體中。
```c
size = M * K * sizeof(float);
// 將裝置端記憶體 d_C 傳回 主機端記憶體 C
cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

// GPGPU 裝置端空間釋放
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
return 0;
```

裝置端程式常常由多個函數組成，這些函數被稱為**核心函數 (kernel)**，這些核心函數會被分配到每個 GPGPU 的執行緒中執行。

```c
// __global__ 關鍵字定義了這個函數會作為核心函數在 GPGPU 上跑
__global__ void basic_mul(float* d_A, float* d_B, float* d_C){
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int col = threadIdx.y + blockIdx.y * blockDim.y;
  for (int i = 0; i < N; i++){
    d_C[row * K + col] += d_A[row * N + i] * d_B[col + i * k];
  }
}
```

# 執行緒模型

### 執行緒組織結構
上面提到，主機端在啟動核心函數時，利用 <<<>>> 向 GPGPU 傳送兩個參數 gridDim 和 blockDim，這兩個參數構造了 GPGPU 計算所採用的執行緒結構。

CUDA 和 OpenCL 都採用了層次化的執行緒結構，就是前面說的 thread grid、thread block、thread 和 NDRange、work-group、work-item，一一對應。同一個 Block 內的 Thread 可以互相通訊。

![CUDA 的層次化執行緒結構](https://www.researchgate.net/publication/328752788/figure/fig3/AS:689781692432384@1541468179263/CUDA-programming-grid-of-thread-blocks-Source-NVIDIA.png)


### 資料索引
基於上面的執行緒層次，我們需要知道 Thread 在 Grid 中的具體位置，才能讀取合適的資料執行對應的計算。上面例子的 blockIdx、threadIdx 就是用來決定 Thread 的位置。
