---
title: VLSI
date: 2024-02-25 10:48:15
tags:
math: true
---

## Redundant Via Insertion

一個 IC 有幾十億個 `via`，任何一個掛掉，整個 IC 也會掛掉

### Double Via Insertion

可以讓上下兩層 `metal` 凸出來一個，多塞一個 `via` 進去，這樣就算其中一個 `via` 掛掉，還有另一個 `via` 可以通電，增加 `Reliabllity`。多出來的 `metal` 可以不用符合 `Design Rule` (L 型 metal)

相鄰的 `via` 不能在同個位置，稱之為 `via conflict`，還要考慮 `via enclosure`，每個 `via` 要有一個最小的空間

<!-- 這可以看成是 `Maximum Independent Set` 的問題，找到最大的 `Independent Set`，然後把其他的 `via` 都拔掉，是 `NP-Hard` 問題 (Polynomial Time 完成不了)

因此我們著重於 `Maximal Independent Set` (Polynomial Time)，從最小 `degree` 的 `vertex` (via) 開始，因為它所連接的 `vertex` 最少，最不會影響別人。 -->
