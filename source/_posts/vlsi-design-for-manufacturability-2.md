---
title: VLSI DFM - Redundant Via Insertion
date: 2025-03-10 10:48:15
tags:
math: true
---

> 參考清大麥偉基老師課程講義

## Redundant Via Insertion

一個 IC 有幾十億個 `via`，任何一個掛掉，整個 IC 也會掛掉

### Redundant Via

可以讓上下兩層 `metal` 凸出來一個，多塞一個 `via` 進去，這樣就算其中一個 `via` 掛掉，還有另一個 `via` 可以通電，增加 `Reliabllity`。多出來的 `metal` 也要考慮 `Design Rule`。

![Redundant Via](./images/vlsi-design-for-manufacturability/RedundantVia.png)

### Post-Routing Double Via Insertion (DVI)

輸入一個已經繞線好、 `via` 也打好的設計，目標是取代原本單一的 `via`，插入兩個 `via`，取代越多越好

![Post-Routing Double Via Insertion](./images/vlsi-design-for-manufacturability/PostRoutingDoubleViaInsertion.png)

#### Conflict Graph Construction

![Conflict Graph Construction](./images/vlsi-design-for-manufacturability/ConflictGraphConstruction.png)

Feasible Candidate Via 會變成 Graph 裡面的 Node，如果兩個 `via` 有 `via conflict`，就會有一條 Edge

#### Heuristic for solving the MIS Problem – H2K

H2K 會迭代很多次，每次都找 size 為 k 的 subgraph，找出 Maximal Independent Set，並且更新 Conflict Graph。

某個 Vertex 擁有越小的 feasible number and degree，他的 priority 越高

- Feasible Number of a vertex: 源自於同個 single via 的 feasible double vias 的數量 - 1

via density 會影響 CMP 的平坦程度
