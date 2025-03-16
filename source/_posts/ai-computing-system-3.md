---
title: AIAS - Digital Design
date: 2025-03-17 05:29:23
tags: [AI]
category: AI
math: true
---

## Lecture

- Moore's Law is slowing down  
  製成技術的瓶頸
- Dennard Scaling is dead two decades ago
  雖然 Transistor 數量增加，但是 Power Consumption 也增加，晶片的運作沒辦法滿載，因此 Power 是現今重要的議題。
- Specialized & Customization
  - 現今逐漸趨向 Specialized Hardware，像是專門 for AI training、inference 的晶片，比較少用 General Purpose 的晶片來做，這樣 Power Consumption 和 Performance 會比較好
- 設計成本高

### Design Flow and Abstraction

以 RISC-V Design Process 為例

- Specification (in plain text)
  如果要設計 RISC-V 相容的 CPU，要去官方網站下載 Spec。RISC-V 是一個 Modulized 的 Spec，除了 Basic Instruction Set 之外，還有很多 Extension，像是矩陣乘法、浮點數運算等等。不同公司會為了不同的需求，選擇不同的 Extension。

  Spec 是軟體與硬體共同 Follow 的標準，只要你硬體是符合 Spec 的，其他跟這個 Spec 相容的軟體都可以跑在這個硬體上。

- Model (in C/C++/SystemVerilog)
  用 High Level Language 把重要的 Parameter、Feature Model 出來，去衡量做出的晶片能否符合需求。不過 ASIC 不太會做 Model，因為它行為固定，很容易預測。但像是 CPU、GPU 這種複雜的晶片，就會做 Model。像是 Performance Modeling，幫助去了解要設計的硬體長什麼樣子，衡量 Design Trade-off。

  另一個原因是軟體和硬體是同時開發的，去 Model 執行硬體的 Behavior，可以幫助軟體早期開發。

- Architecture (in-order/out-of-order)
  這個階段會決定 CPU 的架構，像是 In-Order、Out-of-Order，這個架構會影響到 CPU 的 PPA。Modeling 是幫助 Architecture 去選擇最適合的架構。

- RTL Logic Design (in Verilog/SystemVerilog)
  在 RISC-V 的 CPU 世界，Chisel 比較常見

- Physical Design
- Manufacture part
