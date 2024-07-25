---
title: 作業系統筆記 OS Structure
date: 2024-07-18 17:33:12
tags: OS
---

> 周志遠教授作業系統開放式課程

# OS Services
## User Interface
### CLI (Command Line Interface)
- GUI 是 based on CLI，所以 GUI 能做到的是 CLI 一定可以
  
- **Shell**: Command-line Interpreter (CSHELL, BASH)
  - 不屬於 OS，你打的指令不是直接交給 OS，是交給 Shell，方便使用者使用指令
  - 一台電腦可能有很多使用者，每個使用者的喜好不同，介面、顏色、指令等等，可以做一些客製化的調整

### GUI (Graphic User Interface)
- Microsoft 崛起的原因

## Communication
- 不是單指跨電腦的網路，也可以是 Multi-thread、Multi-processor 等同一台電腦內部的溝通

### Message Passing
- 為了先前提到的 Protection (Base register、Limit register)，程式之間不能直接互相影響，會先把資料複製到 OS，再從 OS 複製到另一支程式
- 會比較慢

### Shared Memory
- 也需要透過 System Call 去建立這塊 Shared Memory，不過像是 Multi-thread 預設就有
- 會有 Synchronization 的問題

![Communication](./images/os-chapter2/Communication.png)

OS Service 除了 User interface 和 Communication 以外，還有
- Program Execution 
- I/O operations 
- File-system manipulation
- Error detection 
- Resource allocation
- Accounting
- Protection and security 

# OS-Application Interface

## System Calls
- OS 提供很多 Services，要使用這些 Service 都會需要透過 System Call，所以 System Call 就是 OS 的 Interface
- 也是一種 Software Interrupt，這樣才能去改變 Mode
- 為了效能，是使用 Assembly Language 撰寫


## API (Application Program Interface)
- 直接使用 System Call 是一件很麻煩的事情，所以 User 的程式通常是使用 API 來做到這件事，而不是直接 Call System Call。大部分的 API 都是使用 C 語言做成的 Library
  
- API 有可能包含很多 System Call，也有可能完全沒有 System Call
  - 沒有 System Call: 方便使用者使用，像是一些數學的計算
  
- 一些常見的 API
  - **Win32** API for **Windows**
  - **POSIX** API for **POSIX-based Systems** (UNIX、 Linux、Mac OS)
  
    - POSIX: (Portable Operating System Interface for Unix)
    - 我在 Linux 上寫一個程式可以執行，直接拿到 Mac 上一定也可以跑，因為 Interface 的定義完全一樣 (Library 可能不一樣)
  - **Java** API for **Java Virtual Machine** (JVM)
  
![OS Interface](./images/os-chapter2/OSInterface.png)

# OS Structure

## Simple OS Architecture
- 開發很快，但是系統裡面的架構全部混在一起
- 定義不清楚，非常不安全，也不好維護

![Simple OS Architecture](./images/os-chapter2/SimpleOSArchitecture.png)

## Layered OS Architecture
- 功能分割得很清楚，上層可以 Call 下層，下層無法 Call 上層
- 很好 Debug、維護
- 因為是 Layerd，可能涉及到許多 Memory Copy，效能不好

![Layered OS Architecture](./images/os-chapter2/LayeredOSArchitecture.png)

## Microkernel OS
- Kernel 的程式碼越少越好，比較 Reliable，不要有 bug 就好

- Modularize 的概念，Kernel 只負責溝通不同 Module，Kernel 以外的全部在 User Space
  
- 效能比 Layered 還要更糟糕
  - User Space 的東西之間要溝通，都需要 **System Call**
  - 為了避免 Synchronization 的問題，都是透過 **Message Passing**

![Microkernel OS](./images/os-chapter2/MicrokernelOS.png)

## Modular OS Architecture
- 很常見，現在大多是使用這種架構
- 跟 Microkernel OS 的差別在，都是在 Kernel Space，方便 Module 之間溝通，跑起來更有效率

![Modular OS Architecture](./images/os-chapter2/ModularOSArchitecture.png)

## Virtual Machine
- 一台電腦有很多使用者，每個人可能會需要自己的 OS
- VM 能夠做一個硬體抽象層，映射到原本電腦的硬體，讓 VM 使用

![Virtual Machine](./images/os-chapter2/VirtualMachine.png)

### 問題
- VM 全部都是跑在 User Space，無法直接執行 Privileged Instruction
  - 需要送出一個 Intrucupt 到原本的 OS (Kernel Space)，然後原本的 OS 再幫它重複執行一次，才在 User Space 做 Kernel Space 的事情
  - 有些 CPU 會特別支援 Hardware Support，也就是多一個 bit 去記錄 User Mode、Kernel Mode 以及 Virtual Machine Mode，就可以直接執行 Privileged Instruction

- Critical Instruction
  - User Space 可以執行，但是執行結果和在 Kernel Space 的執行結果不一樣

### Usage
- 提供完全的 Protection，使用者不會互相影響，一個 OS 被 Hack 其它 OS 也沒事
- 提供特定的執行環境
- 測試開發 OS，避免整台電腦 Crash
- 實現資源管理，像是有些雲端計算會用到 VM


### Full Virtualization
- Guest OS 的程式碼完全不用動，可以直接裝在原本的 OS 上
- Vmware

![Full Virtualization](./images/os-chapter2/FullVirtualization.png)

### Para-virtualization
- Guest OS 會需要修改
- 有一個 Manager 去管理所有 Guest OS
- Xen

![Para-virtualization](./images/os-chapter2/Para-virtualization.png)

### Java Virtual Machine
- Java 執行的方式就像跑在一個 Virual Machine 上
- 跟 Nachos 很像，只做 Instruction 的轉換，把 Java Machine 上 Compile 出的 Bytecodes 轉換成其它的
- 有一些 Translation 上的優化，像是 **Just-In-Time (JIT)**，記錄 Translation 過的 Instruction

![Java Virtual Machine](./images/os-chapter2/JavaVirtualMachine.png)