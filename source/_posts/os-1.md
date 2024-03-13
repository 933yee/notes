---
title: os-1
date: 2024-03-13 10:43:41
tags: OS
---

> 周志遠教授作業系統開放式課程

## Natchos Machine Problem
- C++ Language
- Linux coding environment
- Code tracing

## Course Contents
- Overview
- Process Management
- Process Coordination
- Memory Management
- Storage Management

## Historical Prospective
- Mainframe Systems
  - Batch
  - Multi-programming
  - Time-sharing
  
- Computer-system Architecture
  - Desktop Systems: single processor
  - Parellel Systems: tightly coupled
  - Distributed Systems: loosely coupled

- Special-purpose Systems


### Mainframe Systems
- 最早的電腦，體積很大，用於科學計算
- IO device 很慢
  - card reader/printer, tape drivers
- Evolution:
  - Batch -> Multi-programming -> Time-sharing
- 現在說的 Mainframe 是指專門做某工作的機器
  - For critical application with better **reliability & security**
  - Bulk data processing
  - 廣泛使用於醫院、銀行

#### Batch Systems
![Batch System](./images/BatchSystem.png)
- Process steps:
  - Users submit jobs (program, data, control card)
  - Operator(人) sort jobs with similar requirements
  - **OS simply transfer control from one job to the next**
- 缺點
  - 一次只能做一個工作
  - user 和 jobs 之間無法互動，執行過程中不能做其他事情
  - **CPU 常常耍廢**
    - 資源使用率很低
    - IO 速度 << CPU 速度


#### Multi-programming Systems
- Overlaps the I/O and computation of jobs
  - 某 program 在做 IO 時，CPU 不用等他做完，先做其他的 program
- Spooling (Simultaneous Peripheral Operation On-Line)
  - 做 IO 的過程不需要 CPU 參與
  - 只需要 IO 做完再 notify CPU
  - 靠 interrupt 達成
- Load 很多 jobs 到 Main memory，讓 CPU 決定現在要執行哪個
- 以上的工作包含
  - Memory Management
  - CPU Scheduling
  - I/O System

#### Multi-tasking Systems
- Multi-programming 雖然 CPU 的使用率提高了，但不是 **interactive**，最大問題就是依然只能一次執行一個
- Time-sharing
  - CPU 執行時，可以很頻繁的偵測 IO device 有沒有 input，就可以偵測使用者的互動
  - 每個 job 只執行幾個 millisecond，就切換其他 job，讓使用者覺得每個程式好像同時在執行
  - 螢幕、鍵盤都是這種概念
- 可以讓多個使用者分享同個電腦
- Switch jobs when
  - finish
  - waiting IO
  - a short time peroid
- 以上的工作包含
  - Virtual Memory
    - 讓 memory 越大越好，可以 Load 更多 programs
  - File System and Disk Management
  - Process Synchronization and Deadlock

![Mainframe System Summary](./images/MainframeSystemSummary.png)


### Computer-system Architecture
#### Desktop Systems: PC
- Single user
- GUI
- lack of file and OS protection
  - 那時候沒有考慮到後來出現的網路

#### Parallel Systems
- **multiprocessor** or **tightly coupled system**
  - More than one CPU/core in close communication
  - Usually communicate through **shared memory**
- Purpose
  - Throughput,
    - 計算量增加
  - Economical
    - 很多東西可以共用，memory、CPU、IO device、主機板...
  - Reliability
    - 一個 CPU 掛了，其他還能繼續工作

![Parallel Systems](/images/ParallelSystems.png)

##### Symmetric multiprocessor system (SMP)
- 每個 processor 角色都相同，都由 OS 控制
- 現在幾乎都是 SMP
- Require **extensive synchronization**
  - overhead 會比較大

##### Asymmetric multiprocessor system
- 有一個 Master CPU 和很多 multiple slave CPUs
- 每個 processor 會被 assign 特定工作
- Master CPU 無法用來計算，只能用來管理
- 通常用在比較大的 system

##### Multi-Core Processor
- A CPU with **multiple cores on the same die (chip)**
- On-chip communication 會比 between-chip communication 還要快
- One chip with multiple cores 會比 multiple single-core chips 還要省電
![Multi-Core Processor](./images/MultiCoreProcessor.png)

##### Many-Core Processor
- GPGPU
  - Single Instruction Multiple Data (SIMD) 的操作
  - 上千個 core
- Intel Xeon Phi
- TILE64

##### Memory Access Architecture
- Uniform Memory Access (UMA)
  - 每個 CPU access memory 的速度相同，使用者不用在乎現在是在哪個 CPU 上執行
  - Identical processors
  - most commodity computers
![UMA](./images/UMA.png)


- Non-Uniform Memory Access (NUMA)
  - often made by physically linking two or more SMPs
  - One SMP can directly access memory or another SMP
  - Memory access across link 會比較慢
  - hierarchy 的架構，可以建構更大的電腦
  - 高效能計算系統都是 NUMA
![NUMA](./images/NUMA.png)


#### Distributed Systems
- loosely coupled system
- 每個 system 有自己的 local memory
- Easy to scale
- Purposes
  - Resource sharing
  - Load sharing
    - 某電腦工作量太大可以分給別台做
  - Reliability
    - 一台電腦壞掉不會影響其他台


##### Client-Server
- Eaiser to manage and control resources
- Server 可能會變成 bottleneck 和 single failure point
- FTP

##### Peer-to-Peer
- Decentralized
- ppStream, bitTorrent, Internet

##### Clustered Systems
- Cluster computers share storage
- Local Area Network (LAN)，更快
- Asymmetric clustering, Symmetric clustering

![System Architecture](./images/SystemArchitecture.png)