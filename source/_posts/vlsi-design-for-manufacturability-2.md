---
title: VLSI DFM - Redundant Via Insertion
date: 2025-03-18 22:48:15
tags:
math: true
---

> 參考清大麥偉基老師課程講義

## Redundant Via Insertion

一個 IC 有幾十億個 `via`，任何一個掛掉，整個 IC 也會掛掉

## Redundant Via

可以讓上下兩層 `metal` 凸出來一個，多塞一個 `via` 進去，這樣就算其中一個 `via` 掛掉，還有另一個 `via` 可以通電，增加 `Reliabllity`。多出來的 `metal` 也要考慮 `Design Rule`。

![Redundant Via](./images/vlsi-design-for-manufacturability/RedundantVia.png)

## Post-Routing Double Via Insertion (DVI)

輸入一個已經繞線好、 `via` 也打好的設計，目標是取代原本單一的 `via`，插入兩個 `via`，越多越好

### Maximum Independent Set (MIS)-based approach to DVI

把 DVI 問題轉換成 MIS 問題，找出最大的獨立集合，然後把這些 `via` 插入進去

![Post-Routing Double Via Insertion](./images/vlsi-design-for-manufacturability/PostRoutingDoubleViaInsertion.png)

#### Conflict Graph Construction

![Conflict Graph Construction](./images/vlsi-design-for-manufacturability/ConflictGraphConstruction.png)

Feasible Candidate Via 會變成 Graph 裡面的 Node，如果兩個 `via` 有 `via conflict`，就會有一條 Edge

- 如果原本有的兩個 Single Via，他們長出的 Double Via 有衝突，就會有一條 **External Edge** (上圖綠線)
- 源自同個 Single Via 的 Double Via 之間，會有一條 **Internal Edge** (上圖黑線)

#### Heuristic for solving the MIS Problem – H2K

H2K 會迭代很多次，每次都從 `Priority Queue` 裡面選出前 `k` 個 `via` 組成的 subgraph，然後這個算出 subgraph 的 Maximal Independent Set。選好之後更新 `Conflict Graph`，把 **這些 via 和他們的鄰居 via 通通刪掉**，這樣就完成一次 iteration。

其中，`Priority Queue` 的 `Priority` 是由兩個數值決定的

- **Feasible Number**: 這個 `Double Via` 源自的 `Single Via` 有多少個 `Feasible Candidate Via` - 1 (去掉自己)
- **Degree**: 這個 `Double Via` 有多少條 Edge

**Feasible Number** 和 **Degree** 越小，`Priority` 越高，因為去掉它比較不會影響其他 `via` 的選擇，有機會選到更多的 `Double Via`。

##### Example

我有一個這樣的設計

![Example Design](./images/vlsi-design-for-manufacturability/ExampleDesign.png)

可以建構出這樣的 `Conflict Graph`

![Example Conflict Graph](./images/vlsi-design-for-manufacturability/ExampleConflictGraph.png)

其中 Vertex 上的數字代表 (Degree, Feasible Number)，藉由這組數字來決定 `Priority`

這邊假設 `k` = 4，取出前 4 個 `via` (f, g, l, n) 組成 subgraph，然後算出 MIS。

![Example Subgraph](./images/vlsi-design-for-manufacturability/ExampleSubgraph.png)

算出 MIS (f, l) 後，更新 `Conflict Graph`

![Example Updated Conflict Graph](./images/vlsi-design-for-manufacturability/ExampleUpdatedConflictGraph.png)

這樣就完成一次 iteration。持續迭代直到 `Conflict Graph` 為空。

- Reference: [Post-routing redundant via insertion for yield/reliability improvement](https://ieeexplore.ieee.org/document/1594699)

### 0-1 ILP approach to DVI (Integer Linear Programming)

把 DVI 問題轉換成受很多限制條件的 ILP 問題

![0-1 ILP approach to DVI](./images/vlsi-design-for-manufacturability/01ILPApproachToDVI.png)

直接硬解問題太難 (ILP 是 NP-Hard)，所以要先用其他方式簡化問題

#### Pre-selection

盡量不要選碰到 `External Edge` 的 `via`，因為這樣會影響其他 `via` 的選擇

![Pre-selection](./images/vlsi-design-for-manufacturability/PreSelection.png)

#### Connected Components

做完 Pre-selection 之後，可以想像會有很多獨立的 Subgraph，使用 `DFS` 把這些 `Connected Components` 找出來，每個 `Connected Component` 都是一個更小的 ILP 問題

#### Reduction in Constraints

合併一些限制條件，讓 ILP 問題變得更簡單

![Reduction in Constraints](./images/vlsi-design-for-manufacturability/ReductionInConstraints.png)

#### Via Density Constraint

在現實層面，因為 CMP (Chemical Mechanical Planarization) 的關係，每個區域內的 `via` 數量是有限制的，所以要多考慮 Via Density。

以剛剛的 0-1 ILP 問題為例，加上 Via Density Constraint 之後：

多了一個限制條件，讓每個區域內的 `via` 數量不能超過一定的數量

![Via Density Constraint](./images/vlsi-design-for-manufacturability/ViaDensityConstraint.png)

Pre-selection 時除了避免選到 `External Edge` 的 `via`，還要避免選到 `Potential violating region` 的 `via`

![Pre-selection with Via Density Constraint](./images/vlsi-design-for-manufacturability/PreSelectionWithViaDensityConstraint.png)

分割 `Connected Components` 時，不能把 `Potential violating region` 分開

![Connected Components with Via Density Constraint](./images/vlsi-design-for-manufacturability/ConnectedComponentsWithViaDensityConstraint.png)
