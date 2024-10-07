---
title: Parallel Programming
date: 2024-09-13 15:03:34
tags: 
---

# 平行計算 Parallel Computing 
過往我們寫的程式都是 **Sequential Code**，都是線性的邏輯，同個問題只能用一個 Processor 處理。**平行計算** 是讓 **多個Processors** 一起解決 **單一問題**，把 Instructions 餵到不同的 Processor，讓他們同時去處理。
- 這些 Processor 每次執行的先後順序可能不固定，所以像是用 print 來 debug 可能結果每次都不一樣，處理起來非常複雜
- 有時候會有 Dependency 的問題要自己特別處理

平行計算可以應用在 Bottleneck 的地方，像是以往高速公路的 Bottleneck 在收費站，現在改成電子化大幅提升處理速度

## 平行計算 VS 分散式計算
兩者都需要多台電腦，但是背景有很大的差異性

**平行計算** 主要是用來 **高效能計算**，通常需要大量計算，因此 **效能** 是最重要的，在有限時間內解決某問題
**分散式計算** 主要是 **商用的**，通常用來提供 **計算服務的共享**，讓使用者共享資源，效能並非最重要的，重視 **分享與管理計算的資源**

## 為甚麼需要平行計算
- 節省時間
使用 **更多資源** 縮短 **執行時間**
- 解決更大的問題
以往某些問題就算給予大量時間，依然無法解決，一定需要大量的計算能力才可以，像是大部分的科學計算 (物理模擬、台積電製程等等)
- 最大化榨乾硬體效能

## 平行計算的演化
1. Single-Core Era
2. Multi-Core Era
3. Distributed System Era
4. Heterogeneous Systems Era
   - 發現 CPU 不再夠用，開始使用 GPU 來解決問題
![Trend of Parallel Computong](./images/parallel-programming-1/TrendofParallelComputong.png)


# 平行電腦的分類 Parallel Computer Classification
## Flynn's classic taxonomy
把電腦架構依據兩個獨立的象限: **Instruction** & **Data**
- 又可以分成 SISD、SIMD、MISD、MIMD

### SISD
- 一次讀一個 Instruction，一種 Data
- Serail Computer，最後到硬體處理都是 Sequential Code

### SIMD
- 一次讀一個 Instruction，可以 Apply 這個 Instruction 到不同的 Data
- ex: vector processor、**GPU**

### MISD
- 多種 Instruction，但是吃的 Data 一樣
- ex: 幾乎沒有應用

### MIMD
- 多種 Instruction，可以吃不同的 Data
- 現在電腦中的 Multi-core CPU 都是這種

## Memory Architecture Classfication
- Shared Memory
   - Uniform Memory Access (UMA)
   - Non-Uniform Memory Access (NUMA)
  
- Distributed Memory 
  - Cluster、Supercomputer、Datacenter

![Memory Architecture Classfication](./images/parallel-programming-1/MemoryArchitectureClassfication.png)


# Parallel Programming Model
不管硬體長怎樣，都把它們做 abstraction，製造假象讓上層使用
   - Message Passing Model 可以支援 Shared Memory Machine
   - Shared Memory Model 可以支援 Distributed Memory Machine

## Shared Memory Programming Model
Thread 之間藉由 Global Memory 做溝通
Ex: POSIX Thread (Pthread)、OpenMP
比較簡單，很多 Sync. 的問題，可能會更慢

## Message Passing Programming Model
不同 Machine 需要 Send、Receive 來溝通，因此需要 Memory Copy
Ex: MPI API
Scability 比較好

# Supercomputer
專門為了高效能計算而設計的

效能通常用 **FLOPS** 來定義，也就是每秒鐘能做多少 **浮點數運算**

執行的 Benchmark 是 HPL benchmark，來做超級電腦排名

## HPL Benchmark
能夠計算 **浮點數執行的速率**

實作方式沒有限制，甚至連記憶體大小之類的也會影響結果。

如果這個 Benchmark 的執行效果好，其他計算基本上也會不錯

# Interconnect
平行計算的時候，真的瓶頸在 **網路** 、 **IO** 這邊，遠慢於 CPU 計算，因此會有其他優化

## Interconnection Networks
四大考量因素: Scalability、Performance、Resilience、Cost

### Network Topology
- Diameter(lantency)
  - 最長的點對點有多少 Link
- Bisection(resilience)
  - 斷掉幾個 Link 會壞掉
- Links(cost)
  - 要多少 Edge
- Degree(scalability)
  - 每個 Node 需要多少 Link 的 Port 的數量


---|---|---|---|---|---
| Linear Array | P-1   | 1   | P-1    | 2   |
| ------------ | ----- | --- | ------ | --- | --- |
| Ring         | P/2   | 2   | P      | 2   |
| ---          | ---   | --- | ---    | --- | --- |
| Tree         | 2logp | 1   | 2(p-1) | 3   |
| ---          | ---   | --- | ---    | --- | --- |
2-D Mesh
2-D Torus
Hypercube

#### 4-D Hypercube
scability 不理想
每個 Node 有 ID，是由 4 bits 組成，鄰居之間都把某個 bit flip

#### 6-Dimensional Mesh/ Torus

#### Dragonfly topology
在比較接近的 Node 或 Rack 之間，會很有密集的 Link，越遠的 Link 越少

#### InfiniBand
效能很高，成本也很高

在超級電腦中，幾乎都是使用 InfiniBand

Ethernet 是用 IP 傳輸，每個網路層彼此都是獨立的，因此 Layer 之間傳輸都要做 Memory Copy，一堆 Context Switch，不利於大量傳輸
InfiniBand 拋棄了 Layer IP 的方式，是藉由 DMA 的控制器去做 IO，不經過 OS、CPU。且 Remote DMA 甚至可以讀到遠端電腦的 Local Memory

![InfiniBand vs. Gigabit Ethernet]


## IO & Storange
Parallel file and IO 系統
![Lustre file system]

### Burst Buffering
用 Non-Volatile RAM
用 Write Back 而不是 Write Thorugh
IO 的 Wrokflow 常常忽高忽低，所以你的網路頻寬要夠，要擴充到 Peak load 才能，但是會變成成本很高，使用率很低。因此 Burst Buffer 會 Smooth IO 傳輸速率。


# Parallel Program Analysis
### Speedup Factor


