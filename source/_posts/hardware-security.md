---
title: Hardware Security
date: 2024-03-15 11:25:11
tags: [hardware, security]
category: hardware
---

## Reliable IC Design and Fabrication withGlobal Electronics Supply Chain

- **Hardware security issues** arise from

  - 硬體內部的漏洞，不管是 Gate Layer、Transistor Layer 或是 Voltage 和 Current
  - 缺乏內建的安全機制來保護軟體與系統

  用旁路攻擊或軟體利用硬體漏洞進行攻擊，導致密碼學被破解、記憶體可以任意 Access、竊取物理資訊等

- **Hardware Trust Issues** arise from

  - 硬體開發過程涉及不可信任的第三方（IP 供應商、EDA 工具、製造商、測試或銷售商）

  在 IC Design Flow 中，IC Design、Fab、Test、Assembly、Package & Testing、PCB & Synthesis 都有風險，導致被 DoS 之類的問題，還可能減少 Performance、增加功耗

兩個差別在：前者是 **硬體本身的漏洞導致被攻擊**，後者是 **硬體開發與生產過程中不受信任的實體**

#### IP

**IP** 是一個 Predeifined、Designed、Verified、Resuable 的 Building Block，可以直接放在 SoC 裡面
可能是一段 Verilog Code、做好的 Gate Level Netlist 或是做好的 Layout

- **Soft IP** 後續可以做調整
- **Hard IP** 一個黑盒子，不可以做調整

根據層級小到大可以分成

- Foundation IP：基本的元件，像是 Cell Library、Gate Array
- Standard IP：有特定的功能，像是 JPEG、USB、PCI
- Star IP：複雜的元件，可以做很多功能，像是 ARM、MIPS

但是因為 IP 來自各種不同的地方，要考慮到 IP 的可信度

### Hardware Threats on IC Supply Chain

- Piracy: 剽竊裡面的設計
- IP Overuse: 使用超過授權的次數
- Reverse Engineering: 逆向工程，找出裡面的設計
- Malicious Modification: Trojan

- Trojan
  - 不能太常發生，不然 verification 會發現 (functional test)
  - payload 不能太大，不然會被發現
  - payload 也有分 combinational 和 sequential，sequential 可以延遲幾個 cycle 再發動攻擊

![Untrusted Entities](./images/hardware-security/untrusted-entities.png)

- Side Channel Attack: 透過量測 Voltage、Thermal 來猜測裡面的資訊
- Scan Based Attack: 透過 Scan Chain 來改變裡面的資訊
  - Scan Chain: 在 Design for Testability 中，每個 FF 後面接一個 MUX，可以藉由控制 MUX 來監測 FF 的值。像如果某個 Combinational Circuit 是加密的計算，可以透過 Scan Chain 來看裡面的值
- IC Counterfeit: 我知道這個 IC 的功能在幹嘛，仿造一個一模一樣的 IC，但裡面參雜一些惡意的電路 (案例比較少
- Watermarking: 在 IC 裡面放一些特定的電路 (個人訊息)，可以用來證明這個 IC 是我的設計
- Hardware Obfuscation: 多花一些額外的 Cost，把電路設計變得很複雜，讓別人不容易看懂
- Side Channel Resistant Design: 加入 Noise、Random Delay、Randomize Power Consumption，讓別人不容易透過 Side Channel Attack 來猜測裡面的資訊
- Secure Scan Chain: 避免在 Operation Mode 的時候被使用 Scan Chain，只有在 Test Mode 的時候才能被使用
