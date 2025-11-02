---
title: Tech Reflections
date: 2025-11-01 20:14:32
tags: [tech]
math: true
---

## 軟體開發的第一原則

1. Make it work
2. Make it right
3. Make it fast

### Make it work

- 如果總是糾結於 refactor、優化，沒有完整能運作的系統，一切都是空談。
- 軟體的生命始於上線，一個沒有上線、沒有被實際使用過的軟體，無論架構多好、程式碼多漂亮，都是毫無意義的。

### Make it right

- 隨著對問題的理解，會接觸到更多的 edge case 和新技術，會逐漸改變對「正確」的認知，不太可能一開始就做出正確的東西。
- 市面上所謂的「state-of-the-art」也是漸進演變的，不是一夕之間決定的，不會是某個人突然發明了一個完美無缺的解法。
- 軟體開發不存在絕對的 right 或 wrong，只有 better 跟 worse ，應該把目標放在 good enough。

### Make it fast

大多數的軟體根本不會走到這一步。

- 一般使用者感覺不出來。
- 外部優化 >> 程式碼優化： 用好的 CDN 服務商、設計合理的部署策略，對使用者體驗的提升遠大於在程式碼層面的優化。
- 優先級很低，等到 work 和 right 兩大根基打穩了，追求極致的效能優化才有意義。
  `
