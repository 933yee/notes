---
title: 作業系統筆記 Historical Prospective
date: 2024-07-02 14:48:31
tags: OS
---

> 周志遠教授作業系統開放式課程

作業系統介於硬體和軟體之間，是不可或缺的

# Mainframe Systems

- 最早的電腦，體積很大
- IO 很慢，做的事情非常單一
- 演化順序：Batch -> Multi-programming -> Time-shared
- 現在仍然存在，意旨專門處理一件事情的機器，很多伺服器在用，負責大量資料的處理

## Batch

- 電腦管理者要親自決定執行順序，決定哪個程式要先執行
- 工作很單純，所以 OS 不需要做甚麼特別的事情，除了基本工作以外，只需要把一項工作切換到下一個
- 缺點
  - 一次只能處理一個程式
  - 沒有 **interaction**，在執行一個工作的過程中，使用者完全沒辦法做任何改變
  - IO 非常慢，導致 CPU 常常掛機，浪費資源
    - IO 速度遠比 CPU 速度慢

![Batch System](./images/os-chapter0/BatchSystem.png)

## Multi-programming

- 讓系統裡面放多個程式
  - 當一個程式在做 IO 時，就可以切換到別的程式，overlap CPU 和 IO 的計算，保證 CPU 持續工作
- Spooling (Simutaneous Peripheral Operation On-Line)
  - 不用 CPU 干預就可以完成 I/O 的工作
  - CPU 只需要在 IO 做完的時候被通知就好
- CPU 在執行的時候只能讀取 memory，所以我們把多個程式同時 load 到 main memory
  - 要 load disk 的哪些程式到 memory？-> **Job Scheduling**
  - 要 load memory 的哪些程式到 CPU？-> **CPU Scheduling**
- 在其中，OS 的工作包含：
  - Memory management，要分配多少記憶體給這些程式
  - CPU Scheduling，CPU 要執行 memory 的哪些程式
  - IO system，如何透過 interrupt，實現 Spooling

## Time-Sharing System (Multi-tasking)

- Multi-programming 讓 CPU 的使用率變高，但是對於單一程式而言，可能會執行更慢
- 使用者想要隨時與程式做互動，也希望可以很多使用者同時使用電腦
- 用時間的觀念去切割資源，讓所有程式一起執行
- 很頻繁切換 CPU 和 IO，就可以偵測使用者的 input，就有互動的感覺
- 不是一次做完一個程式的計算，而是每個程式只計算幾毫秒，就切換到下一個程式，讓使用者以為所有程式同時在執行
- 在其中，OS 的工作包含
  - Virtual memory，把 disk 當作 memory，因為我想要塞更多程式到 memory，但是 memory 不夠大
  - File system 和 disk management，有了 interactive，使用者就可以直接管理檔案
  - Process synchronization 和 deadlock，程式和程式之間可以溝通，當他們同時修改 memory 的內容時就會有問題

![Mainframe System Summary](./images/os-chapter0/MainframeSystemSummary.png)

# Computer-system architecture

## Desktop Systems: single processor

- PC (Personal Computers)，只給一位 user 使用
- GUI (User **convenience** and **responsiveness**)
- I/O devices
- Windows, MacOS, Unix, Linux

## Parallel Systems: tightly coupled

- 又稱為 **Multiprocessor** or **tightly coupled system**
- 很多 CPU 緊密在一起
- 這些 CPU 用同個 **shared memory**
- Purposes
  - Throughput
    - 計算量增加
  - Economical
    - 很多東西可以共用，memory、CPU、IO device、主機板...
  - Reliability
    - 一個 CPU 掛了，其他還能繼續工作
- 現今所有系統都是這種系統

![Parallel Systems](./images/os-chapter0/ParallelSystems.png)

### SMP (Symmetric multiprocessor system)

- 每個 CPU 的角色相同，都由 OS 控制，沒有 Master
- 比較簡單，所以比較普及，我們手上有的系統幾乎都是這類
- 需要 **synchronization**，增加 overhead
- Master CPU 無法用來計算，只能用來管理

### AMP (Asymmetric multiprocessor system)

- 超級電腦等需要大量計算的會用這類
- 有一個 Master CPU，只用來管理其他 CPU ，所以 core 可以比較多

### Multi-Core Processor

- 一個 CPU 裡面有很多 Core

![Multi-Core Processor](./images/os-chapter0/MultiCoreProcessor.png)

### Many-Core Processor

- GPGPU
  - Single Instruction Multiple Data (SIMD) 的操作
  - 上千個 core
- Intel Xeon Phi
- TILE64

### Memory Access Architecture

#### Uniform Memory Access (UMA)

- 每個 CPU access memory 的速度相同，使用者不用在乎現在是在哪個 CPU 上執行
- CPU 增加時，memory 可能開始有 bottleneck

![UMA](./images/os-chapter0/UMA.png)

#### Non-Uniform Memory Access (NUMA)

- 每個 CPU 的 access time 變得不同
- hierarchy 的架構，可以建構更大的電腦
- 高效能計算系統都是 NUMA

![NUMA](./images/os-chapter0/NUMA.png)

## Distributed Systems: loosely coupled

- 每個 processor 有 local memory，不會 share
- 很好擴展更多裝置
- Purposes
  - Resource sharing
  - Load sharing
    - 某電腦工作量太大可以分給別台做
  - Reliability
    - 一台電腦壞掉不會影響其他台

### Client-Server

- 很好管理與控制資源
- Server 可能會變成 bottleneck 或是 single failure point
- FTP

### Peer-to-Peer

- Decentralized，每個系統的角色是一樣的
- ppStream, bitTorrent, Internet

### Clustered Systems

- Share Storage
- 通常用 Local Area Network (LAN)，在同個區域，所以會更快
- Asymmetric clustering, Symmetric clustering

![System Architecture](./images/os-chapter0/SystemArchitecture.png)

# Special-purpose Systems

## Real-Time Operating Systems

- Well-defined **fixed-time contraints**
  - Real-time 代表會在 deadlines 之前做完，跟做得速度沒關係

### Soft real-time

- 盡量完成，沒完成也不會有什麼損失
- 會有 priority，比較重要的會先做
- Ex: multimedia streaming
  - 畫面不是馬上畫出來，可能會先出現線條，然後顏色，最後高解析度
-

### Hard real-time

- 沒有在 deadline 完成，會造成嚴重的後果
- Secondary storage limited or absent
  - 沒有 disk，因為讀取太慢，而且讀取時間不好掌握
- Ex: nuclear power plant controller

## Multimedia Systems

- A wide range of applications including audio and video files
- Issues
  - Timing contraints，屬於 soft real-time
  - On-demand/live streaming
  - Compression，有各種壓縮技術

## Handheld/Embedded Systems

- Hardware specialized OS
- Issues
  - Limited memory
  - Slow processors
  - Battery consumption
  - Samll display screens
