---
title: Hardware Security
date: 2024-03-07 11:25:11
tags: [hardware, security]
category: hardware
---

## Reliable IC Design and Fabrication withGlobal Electronics Supply Chain

- **Hardware security issues** arise from

  - 硬體內部的漏洞
  - 缺乏內建的安全機制來保護軟體與系統

  用旁路攻擊或軟體利用硬體漏洞進行攻擊，導致密碼學被破解、記憶體可以任意 Access、竊取物理資訊等

- **Hardware Trust Issues** arise from

  - 硬體開發過程涉及不可信任的第三方（IP 供應商、製造商、測試或銷售商）
  -

  在設計、製造、測試、銷售過程中，惡意植入後門或木馬電路，導致被 DoS 之類的問題，還可能減少 Performance、增加功耗等

兩個差別在：前者是 **硬體本身的漏洞導致被攻擊**，後者是 **硬體開發與生產過程中不受信任的實體**
