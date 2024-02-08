---
title: Unreal Engine 5 - Render Dependency Graph (RDG)
date: 2024-01-30 20:26:44
tags: UE5
categories: ['UE5', 'Render']
---

### RDG 是什麼
[官方文件](https://docs.unrealengine.com/4.26/en-US/ProgrammingAndScripting/Rendering/RenderDependencyGraph/)提到
: 渲染依賴圖（Rendering Dependency Graph，RDG）是一種基於圖形的排程系統，旨在對渲染管線進行整幀優化。

#### 基本介紹
- RDG 於 Unreal 4.22 引入
  
- 前身是 Frame Graph ，在 2017 年的 GDC 中由 [Frostbite Engine](https://zh.wikipedia.org/zh-tw/%E5%AF%92%E9%9C%9C%E5%BC%95%E6%93%8E) 提出
  
- RDG 的概念是不在 GPU 上立即執行通道 (Pass)，而是先收集所有需要渲染的通道，然後按照它們之間的依賴關係順序對圖表進行編譯和執行。過程中，系統會執行各種裁剪和優化操作。

#### 為什麼要用 RDG？
- Render pipeline 越來越複雜，導致難以管理且性能不好
  
  - 隨著硬體性能日漸提升，各大廠商為了渲染出更出色的畫面效果，render pipeline 也日趨複雜。
  
![rendering systems overview](\images\ue5-rdg\rendering-systems-overview.png "Rendering systems overview")


- 硬體的體系結構以及圖形 API 的優化擴展無法得到充分的利用
  
  - 現代圖形API (如DirectX 12、Vulkan 和 Metal 2) 與傳統圖形API (如DirectX 11、OpenGL) 的主要區別在於現代圖形 API 將更多的 GPU 管理的責任轉移到應用程式的開發者身上，能夠更有效的利用有限的 GPU 資源，進而提升效能。
  
    - [傳統 API 和現代 API 的介紹](https://zhuanlan.zhihu.com/p/73016473)
    
  - RDG 與現代圖形 API 的能力相結合，使 RDG 能夠在幕後執行複雜的排程任務：
      1. 執行異步計算通道的自動排程和隔離。
      2. 在幀的不相交間隔期間，使資源之間的別名 (Aliasing) 記憶體保持活躍狀態。
      3. 盡早啟動屏障和佈局轉換，避免管線延遲。

### RDG 的原理
![rdg in the game engine](\images\ue5-rdg\rdg-in-engine.png "RDG in the game engine")

位於 RHI 和 Render Pass 的中間，RDG 作為 Pass 管理器，在搜集資源描述訊息後，對 Pass 和資源進行分析，並結合硬體特性，以最優的方式執行 Pass，主要有三個階段：

1. Setup
   - 蒐集 pass 的訊息(主要是該 Pass 使用到的資源)
  
2. Compile
   - Render Graph 的生成以及分析 (包含 Pass culling、Resources state 的最終生成、Async Compute 優化等等)
  
3. Execute
   - 將 Command 提交到 CommandList (包含設置 Barrier、平行優化等)

![Three stages of RDG](\images\ue5-rdg\rdg-stages.png "Three stages of RDG")

### FRDGBuilder
- RDG 系统的心臟和驅動器，同時也是管家，負責儲存數據、處理狀態轉換、自動管理資源生命週期和屏障 (barrier)、裁剪無效資源，和收集、編譯、執行Pass，提取紋理或緩衝等等功能。


### 參考資料
- [剖析虚幻渲染体系（11）- RDG](https://zhuanlan.zhihu.com/p/554758862)
- [Rendering Dependency Graph](https://docs.unrealengine.com/4.26/en-US/ProgrammingAndScripting/Rendering/RenderDependencyGraph/)
- [UE5 Render Dependency Graph-实用指南](https://zhuanlan.zhihu.com/p/637889120)
  