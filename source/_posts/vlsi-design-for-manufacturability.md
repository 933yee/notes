---
title: VLSI Design for Manufacturability 筆記
date: 2024-02-25 10:48:15
tags:
math: true
---

> 參考清大麥偉基老師課程講義

## Lithography

- Lithography: 光刻
  框出電路圖案，然後進行蝕刻
- Diffraction: 繞射
  會造成 Lithography 不準確，可以從目標圖案反推出 Mask 的設計，稱為 Computational Lithography

![Lithography Development](./images/vlsi-design-for-manufacturability/LithographyDevelopment.png)

- 193i 的 `i` 代表 Immersion Lithography
  用液體取代空氣，縮短光的波長，就好像換了一個 Light Source，減少繞射的影響，而且波長 193nm 的技術已經很成熟
- EUV 代表 Extreme Ultraviolet Lithography
  波長更短，繞射更少，但是製程更難，因為要用真空，光源也很難做
- Multi-Patterning
  把一個圖案分成多個小圖案，然後分別曝光，最後合併成一個圖案。在 EUV Lithography 之前很常用，現在因為 EUV 很貴，有些先進製程還是用 193i + Multi-Patterning
- 隨著技術進步，單位面積上的 Transistor 數量越來越多，但是製程成本也越來越高，所以不是每個裝置都用最先進的製程

## Routing in Advanced Nodes

受限於 `Lithography printability`、`Process Variation` 等，即使你用一樣的方法，做出來的也不一定是你要的。所以要考慮 `Design for Manufacturability`。

### Design Rule Handling

**Optical Proximity Correction (OPC)** 是一種 Computational Lithography 的技術，修改 mask 來補償繞射和光刻製程中的非理想效應，讓實際製造出的 Pattern 更接近設計的目標。

- OPC 會造成線寬變寬，所以要考慮 `Design Rule`，ex：兩條線之間要留多少空間

![OPC](./images/vlsi-design-for-manufacturability/OPC.png)

#### End-to-End Seperation Rule

![End-to-End Seperation Rule](./images/vlsi-design-for-manufacturability/EndToEndSeperationRule.png)

- 如果一條 Wire 的端點的相鄰軌道上有另一條 Wire，那這條 Wire 的末端至少要留 **S2** 的空間
- 其他情況只需要預留 **S1** 的空間

#### Minimum Length Rule

每條線至少要有多長

![Minimum Length Rule](./images/vlsi-design-for-manufacturability/MinimumLengthRule.png)

### MANA (A Shortest Path MAze Algorithm under Separation and Minimum Length NAnometer Rules)

傳統的 Maze Routing Algorithm 是在 Grid 上找一條最短的路徑，可能用 BFS 或 Dijkstra 等等，但是在 VLSI 設計中，還要考慮一些 `Design Rule`

#### 考慮到 Design Rule 的 Maze Routing Algorithm

- Post-Processing
  把找到的路徑再做一些調整，像是延伸 Wire，讓他們符合先前提到的 `Design Rule`，不過這方法沒那麼好，會花很多資源

  ![Post-Processing](./images/vlsi-design-for-manufacturability/PostProcessing.png)

- 增強型 Maze Routing Algorithm
  在找路徑的時候就考慮 `Design Rule`，像是「每條 Wire 至少要有多長」
  ![Enhanced Maze Routing Algorithm](./images/vlsi-design-for-manufacturability/EnhancedMazeRoutingAlgorithm.png)

## Redundant Via Insertion

一個 IC 有幾十億個 `via`，任何一個掛掉，整個 IC 也會掛掉

### Double Via Insertion

可以讓上下兩層 `metal` 凸出來一個，多塞一個 `via` 進去，這樣就算其中一個 `via` 掛掉，還有另一個 `via` 可以通電，增加 `Reliabllity`。多出來的 `metal` 可以不用符合 `Design Rule` (L 型 metal)

相鄰的 `via` 不能在同個位置，稱之為 `via conflict`，還要考慮 `via enclosure`，每個 `via` 要有一個最小的空間

<!-- 這可以看成是 `Maximum Independent Set` 的問題，找到最大的 `Independent Set`，然後把其他的 `via` 都拔掉，是 `NP-Hard` 問題 (Polynomial Time 完成不了)

因此我們著重於 `Maximal Independent Set` (Polynomial Time)，從最小 `degree` 的 `vertex` (via) 開始，因為它所連接的 `vertex` 最少，最不會影響別人。 -->
