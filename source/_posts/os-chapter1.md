---
title: 作業系統筆記 Introduction
date: 2024-07-09 17:50:22
tags: OS
---

> 周志遠教授作業系統開放式課程

# Introduction

## Computer System
- 由 Hardware、OS、Application、User 組成
  - User：people、machines
  - Application：使用 system resources 來解決問題的方式，像是各種軟體
  - Operation System：**controls** 與 **coordinates** 資源的使用
  - Hardware：提供基本的計算資源（CPU、memory、IO devices）

## What is an Operation System
- **Permanent** Software
  - 電腦啟動後永遠存在，沒有它電腦就不會運作
- **abstracts** 硬體資源給使用者使用
  - 把底層的各種資源變成一堆 API，給使用者方便使用，virtual 的概念

### General-Purpose Operating Systems
- Linker 會 link System Library，裡面有一堆 System call 的 API
- 使用 printf 的時候要印到螢幕，是 OS 負責的，所以要 system call
  - user 呼叫 printf -> printf 呼叫 system call -> system call 呼叫 driver...
- Device drivers 是 OS 的一部分
  
![General-Purpose Operating Systems](./images/os-1/GeneralPurposeOperatingSystems.png)

### OS 的定義
- Resource allocator
  - **manages** 和 **allocates resources** 來確保公平性和效率
- Control program
  - **controls** 程式的執行 (driver) 和 IO devices 的操作
- Kernel
  - OS 又稱為 Kernel

### OS 的目的
- 方便性 (convenience)
  - 像是 windows 的崛起，也就是發展出圖形介面，讓使用者更方便操作
- 效率 (Efficiency)
  - 更有效率的使用資源，當問題很複雜時，Efficiency 是最後追求的東西
  - 像是很多人用 Linux，是因為圖形介面也會吃資源，但那不是必要的
- 兩者衝突，追求方便性效率就下降，追求效率方面性就下降

### OS 的重要性
- OS 是 user program 和 hardware 之間 **唯一** 的 interface，一定需要經過它
- OS 絕對不能有 bug，一旦 crash 掉整台電腦就掛了
- OS 和電腦的架構息息相關，因此隨著需求不同，也發展出形形色色的 OS

## Computer-System Organization