---
title: 作業系統筆記 Processes Conecpt
date: 2024-07-26 15:15:10
tags: OS
---

> 周志遠教授作業系統開放式課程

# Process Concept

### Program
- 被動、Binary 的 File 存在硬碟裡面

### Process
- 主動、正在記憶體裡面執行的程式

- 一個 Process 裡面包含：
  - Code Segment (text section)
  - Data Section (global variables)
  - Stack (暫時的 local variables 和 functions)
  - Heap (動態分配的 variables 或 classes)
  - 記錄現在的資料 (**program counter**、register contents)
  - 其他相關的資源 (OS resources, e.g. open file handlers)

![Process In Memory]()

### Threads
- A.K.A **lightweight process**
  - 跟 Process 長的一樣，不過有些 Threads 可以共用 Memory 空間
  - 是 Basic unit of CPU utilization
  
- 在同一個 Process 底下的 Threads 會共用
  - Code Section
  - Data Section
  - OS resources

- 每個 Thread 有自己的
  - Thread ID
  - Program Counter
  - Register Set
  - Stack

因此，在寫 Multi-Thread 的時候可以用 Global Variables (Data Section) 來做溝通

![Threads]()

## Process States
- States
  - New: 這個 Process 剛被創造出來
    - Program Load 到 Memory，並初始化前面提到的那些 (Code Section、Data Section...)
    - 分配要分配多少給 Process

  - Ready: Process 要競爭的資源是 CPU，會有一個 Queue 存放這些 Process，等待 CPU 排程
  
  - Running: 在 Ready 中被選到了，可以開始執行程式
    - 有時候 Running State 會直接回到 Ready State，通常是因為 Timer 到了，送出 Interrupt，而不是因為 IO
  
  - Waiting: 在做 IO 的時候不需要 CPU 參與，等到完成後會回到 Ready
    - 也有一個 Queue 來儲存
  
  - Terminated: 釋放所有分配給這 Process 的資源

![Diagram of Process State]()

## Process Control Block (PCB)
- OS 要能掌握每個 Process 的邏輯來管理，所以每一個 Process 會有一個 Process Control Block
- 像是前面說的把 Process 放進 Queue 其實是一個抽象的概念，實際上是放進 PCB，然後裡面的
- PCB 裡面包含
  - Process State (Ready、Waiting...)
  - Program Counter
  - CPU Registers
  - CPU Scheduling Information (這個 Process 的 Priority)
  - Memory-Management Information (Base/ Limit Register)
  - I/O Status Information (正在做哪個 IO Device 的 IO)
  - Accounting Information (你開了幾個檔案)

![Process Control Block]()

## Context Switch
- 藉由 Interrupt，把原來的 Process 替換成另一個 Process
  - 會把舊 Process 的資料存到 PCB 裡，把新 Process 的資料 Load 到 PCB 裡
- Context Switch 所花的時間就是 Overhead，在這期間兩個 Process 都在 Idle，為了 Time-Sharing 這是無法避免的

- Context Switch Time 基於
  - Memory Speed
  - Register 數量
  - 用特殊的 Instruction，像是某個 Instruction 可以一次 Load 所有 Register
  - Hardware Support: CPU 包含很多 Sets of Registers，一次去記很多程式的狀態，在 Context Switch 得時候就不用寫到 Memory 

![Context Switch]()

# Process Scheduling
- 為了實現 Multiprogramming 和 Time Sharing

### Queues
- Job Queue (New State): 哪些 Process 可以 Load 到 Memory
- Ready Queue (Ready State)
- Device Queue (Wait State)

![Process Scheduling Queues]()

![Process Scheduling Diagram]()

## Scheculers

# Operations on Processes

# Interprocess Communication