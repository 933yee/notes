---
title: 作業系統筆記
date: 2024-03-13 10:43:41
tags: OS
---

> 周志遠教授作業系統開放式課程


# Course Contents
- Overview
- Process Management
- Process Coordination
- Memory Management
- Storage Management

# Historical Prospective
- Mainframe Systems
  - Batch
  - Multi-programming
  - Time-sharing
  
- Computer-system Architecture
  - Desktop Systems: single processor
  - Parellel Systems: tightly coupled
  - Distributed Systems: loosely coupled

- Special-purpose Systems
  - Real-Time Systems
  - Multimedia Systems
  - Handheld Systems

## Mainframe Systems
- 最早的電腦，體積很大，用於科學計算
- IO device 很慢
  - card reader/printer, tape drivers
- Evolution:
  - Batch -> Multi-programming -> Time-sharing
- 現在說的 Mainframe 是指專門做某工作的機器
  - For critical application with better **reliability & security**
  - Bulk data processing
  - 廣泛使用於醫院、銀行

### Batch Systems
![Batch System](./images/os-1/BatchSystem.png)
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


### Multi-programming Systems
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

### Multi-tasking Systems
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

![Mainframe System Summary](./images/os-1/MainframeSystemSummary.png)


## Computer-system Architecture
### Desktop Systems: PC
- Single user
- GUI
- lack of file and OS protection
  - 那時候沒有考慮到後來出現的網路

### Parallel Systems
- **multiprocessor** or **tightly coupled system**
  - More than one CPU/core in close communication
  - Usually communicate through **shared memory**
- Purposes
  - Throughput
    - 計算量增加
  - Economical
    - 很多東西可以共用，memory、CPU、IO device、主機板...
  - Reliability
    - 一個 CPU 掛了，其他還能繼續工作

![Parallel Systems](./images/os-1/ParallelSystems.png)

#### Symmetric multiprocessor system (SMP)
- 每個 processor 角色都相同，都由 OS 控制
- 現在幾乎都是 SMP
- Require **extensive synchronization**
  - overhead 會比較大

#### Asymmetric multiprocessor system
- 有一個 Master CPU 和很多 multiple slave CPUs
- 每個 processor 會被 assign 特定工作
- Master CPU 無法用來計算，只能用來管理
- 通常用在比較大的 system

#### Multi-Core Processor
- A CPU with **multiple cores on the same die (chip)**
- On-chip communication 會比 between-chip communication 還要快
- One chip with multiple cores 會比 multiple single-core chips 還要省電
  
![Multi-Core Processor](./images/os-1/MultiCoreProcessor.png)

#### Many-Core Processor
- GPGPU
  - Single Instruction Multiple Data (SIMD) 的操作
  - 上千個 core
- Intel Xeon Phi
- TILE64

#### Memory Access Architecture
- Uniform Memory Access (UMA)
  - 每個 CPU access memory 的速度相同，使用者不用在乎現在是在哪個 CPU 上執行
  - Identical processors
  - most commodity computers
  
![UMA](./images/os-1/UMA.png)


- Non-Uniform Memory Access (NUMA)
  - often made by physically linking two or more SMPs
  - One SMP can directly access memory or another SMP
  - Memory access across link 會比較慢
  - hierarchy 的架構，可以建構更大的電腦
  - 高效能計算系統都是 NUMA
  
![NUMA](./images/os-1/NUMA.png)


### Distributed Systems
- loosely coupled system
- 每個 system 有自己的 local memory
- Easy to scale
- Purposes
  - Resource sharing
  - Load sharing
    - 某電腦工作量太大可以分給別台做
  - Reliability
    - 一台電腦壞掉不會影響其他台


#### Client-Server
- Eaiser to manage and control resources
- Server 可能會變成 bottleneck 和 single failure point
- FTP

#### Peer-to-Peer
- Decentralized
- ppStream, bitTorrent, Internet

#### Clustered Systems
- Cluster computers share storage
- Local Area Network (LAN)，更快
- Asymmetric clustering, Symmetric clustering

![System Architecture](./images/os-1/SystemArchitecture.png)


## Special-purpose Systems
### Real-Time Operating Systems
- Well-defined **fixed-time contraints**
  - Real-time 代表會在 deadlines 之前做完，跟速度沒關係
- Soft real-time
  - Missing the deadline is unwanted, but is not immediately critical
  - Critical real-time task gets **priority** over others
  - Ex: multimedia streaming
    - 畫面不是馬上畫出來，可能會先出現線條，然後顏色，最後高解析度
- Hard real-time
  - Fundamental failure
  - Secondary storage limited or absent
    - 沒有 harddrive，因為讀取太慢，而且讀取時間不好掌握
  - Ex: nuclear power plant controller

### Multimedia Systems
- A wide range of applications including audio and video files
- Issues
  - Timing contraints
  - On-demand/live streaming
  - Compression

### Handheld/Embedded Systems
- Hardware specialized OS
- Issues
  - Limited memory
  - Slow processors
  - Battery consumption
  - Samll display screens


# Introduction
- What is an Operating System?
- Computer-System Organization
- Hardware Protection

## What is an Operating System
- Computer System
  - Hardware
    - provides basic **computing resources**
    - CPU, memory, I/O devices...
  - OS
    - **controls** and **coordinates** the use of the **hardware/resources**
  - Application
    - define the ways in which the system resources are used to solve computer problems
  - User
    - people, machines, other computers
  
- An OS is the **permanent** software that **controls/abstract** hardware resources for user applications
  - 把下層的 hardware 變成一堆 API 讓使用者用，virtual 的概念
  
- Multi-tasking Operating Systems
  - Manages resources and processes to support different user applications
  - Provides API for user applications
  
- General-Purpose Operating Systems
  - 使用 printf 的時候要印到螢幕，是 OS 負責的，所以要 system call
  - user call printf -> printf call system call -> system call call driver...
  - Device drivers 是 OS 的一部分
![General-Purpose Operating Systems](./images/os-1/GeneralPurposeOperatingSystems.png)
  
  
- Definition of an Operating System
  - Resource allocator
    - **manages** and **allocates resources** to insure efficiency and fairness
  - Control program
    - **controls** the execution of user **programs** and operations of **I/O devices** to prevent errors and improper use of computer
  - Kernel
    - the one program running at all times

- Goals of an Operating Systems
  - Convenience
    - make computer system easy to use and compute
    - in particular for small PC
  - Efficiency
    - use computer hardware in an efficient manner
    - especially for large, shared, multiuser systems
  - Two goals are sometimes contradictory

- Importance of an Operating Systems
  - System API are the **only** interface between user applications and hardware
    - API are designed for general-purpose, not performance driven
  - OS code cannot allow any bug
    - Any break causes reboot
    - 有 bug 代表整台電腦都毀了
  - The owner of OS technology controls the software & hardware industry
    - ex: hardware 和 software 都要 fllow microsoft 的 API、Mac hardware 都自己做 


## Computer-System Organization
- One or more CPUs, device controllers connect through **common bus** providing access to **shared memory**
- Goal: **Concurrent** execution of CPUs and devices competing for memory cycles
  - OS 要負責不讓 access memory 出問題，不讓它發生衝突
  
![Computer-System Organization](./images/os-1/ComputerSystemOrganization.png)

### Device Controller
- Each device controller is in charge of a particular device type
- Status reg 用來記錄現在 device controller 是 busy 還是 idle
- Data reg 和 buffer 都是用來存資料，會先寫到 reg 再寫到 buffer
- Device controller 有自己的 CPU 去 access disk 資料到自己的 buffer
- Memory 是 CPU 在用的，所以 CPU 負責 moves data from/to memory to/from local buffers in device controllers
  
![Device Controller](./images/os-1/DeviceController.png)


### Busy/wait output
- Simplest way to program device
  - Use instructions to test when device is ready
- 浪費 CPU，常常 IDLE
```cpp
#define OUT_CHAR 0x1000 // device data register
#define OUT_STATUS 0x1001 // device status register

current_char = mystring;
while (*current_char != '\0') {
  poke(OUT_CHAR,*current_char); // 寫到 device controller 的 buffer
  while (peek(OUT_STATUS) != 0); // busy waiting
  current_char++;
}
```

### Interrupt I/O
- Busy/wait 很沒效率
  - CPU can't do other work while testing device
  - Hard to do simultaneous I/O
- Interrupts allow a device to **change the flow of control in the CPU**
  - 讓 CPU 可以做其他程式的事情，而不是等 IO 做完
  - Causes subroutine call to handle device

#### Interrupt I/O Timeline
- Interrupt time line for I/O on a single process
![Interrupt I/O Timeline](./images/os-1/InterruptTimeline.png)
可以看到在做 IO 時 CPU 在做其他事情，等 IO 做完後 call 一個 interrupt 打斷 CPU 原本在做的事，讓 CPU 先過來搬資料，interrupt 結束後 CPU 又回去做它原本的事情

#### Interrupt-Driven I/O
![Interrupt-Driven I/O](./images/os-1/InterruptDrivenIO.png)

#### Interrupt
- 現在每個 OS 都是 interrupt driven
- 可能是 Hardware interrupt 或 Software interrupt
  - **Hardware** may trigger an interrupt at any time by sending a **signal** to CPU
    - 比較被動
  - **Software** may trigger an interrupt either by an **error** or by a user request for an operating system serivce (system call)
    - error 像是 division by zero 或 invalid memory access，會讓 program counter reset 到可以印出錯誤訊息的地方，而不是 crash，比較被動
    - system call 就是要叫 OS 做事，呼叫 OS 的 API，比較主動
      - 間接處理，可以區分使用者的 function call 和 OS 的 function call
    - Software interrupt 叫作 **trap**

#### Hardware interrupt
![Hardware interrupt](./images/os-1/Hardwareinterrupt.png)
- interrupt vector 是 array of function pointers，array 大小是固定的
- signal 都會有一個 singal number，根據這個 number 去找 vector 上的欄位
- 每個 port 的 hardware 有燒死的 singal number， 裝 driver 的時候會 overwrite 那個欄位的 pointer 的位置，去執行你要處理的程式碼

#### Software interrupt
- 是用 switch case 而不是 array，因為軟體有無限的可能性，跟硬體無關，可以任意增加不同的 system call
- 流程跟 hardware interrupt 差不多

![Software interrupt](./images/os-1/Softwareinterrupt.png)

#### Common Functions of Interrupts
- Interrupt transfers control to the interrupt service routine generally, through the **interrupt vector**, which contains the **addresses** (function pointer) of all the **service (i.e. interrupt handler) routines**
- Interrupt architecture must save the address of the interrupted instruction
  - 才能在 interrupt 結束後執行原本的程式
- Incoming interrupts are **disabled** while another interrupt is being processed to prevent a lost interrupt
  - 避免發生很多 synchronize 的問題，這些問題需要大量的 overhead 去處理


### Storage-Device Hierarchy
![Storage-Device Hierarchy](./image/os-1/StorageDeviceHierarchy.png)
- 真正的大型的系統最後還是用 tapes，因為非常 reliable
- 這是最傳統的架構，現在有很多其他的 storage device 會插在中間

- Storage systems organized in hierarchy
  - speed, cost, volatility
  - volatile 關掉會遺失
- **Main memory** only large storage media that the **CPU can access directly**
- Secondary storage
  - memory 以下都叫做 secondary storage
  - **large nonvolatile storage**

#### Random-Access Memory
- DRAM (Dynamic RAM)
  - one transistor
  - less power
  - must be periodically refreshed 
  - 體積小，速度比較慢
  - 因為 CPU 有很多 core， RAM 的速度其實就那樣，channel 的 bus 其實才是真正的 bottleneck

- SRAM (Static Ram)
  - six transistors
  - more power
  - cache memory




