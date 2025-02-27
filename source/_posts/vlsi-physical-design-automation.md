---
title: VLSI Physical Design Automation
date: 2025-02-25 13:17:03
tags: [vlsi, physical design, automation]
category: VLSI
---

## IC Design Flow

- System Specification

  定義系統的需求，例如：功耗、面積、效能、功能等等。

- Functional Design

  定義系統的功能，例如：模擬、驗證、合成等等。

- Logic Synthesis

  把功能描述轉換成電路描述，並做邏輯上的優化，例如：RTL、Netlist。

- Circuit Design

  早期才有，這些 Logic Gate 要用那些 Transistor 來做。現在都用 `Cell Based Design`，從 `Cell Library` 拿標準元件來做，這些元件的 Layout 都已經設計好了。

Gate Array

- 裡面的 transistor 都做好了，但還沒有做 intra-cell routing 和 inter-cell routing
- 可以自己決定要把哪些 transistor 連接起來

因為現在 Metal 可以很多層，所以 Row 和 Row 之間不用再留 Routing Channel，都直接在 Cell 上面連線

FPGA

- 第 31 頁中間是 switch bus
- 主要是 Program 圖中的一堆 switch，設定成 0 或 1

- Physical Design
- Fabrication
- Packaging & Testing

## Physical Desgin

把 `Circuit Netlist` 轉換成 `Layout` 的過程，每個元件要擺哪、要怎麼連接、怎麼樣才能達到最佳的 **Power**, **Performance**, **Area** (PPA)，甚至於 **Security**。

![Circuit Netlist](./images/vlsi-physical-design-automation/CircuitNetlist.png)

#### Computer-Aided Design (CAD)

- **CAD** 是一個廣泛的領域，包含了各種不同的應用，例如：電路設計、機械設計、建築設計、電子設計等等。
- **EDA** 是 CAD 的一個子集，專門用來設計電子電路。

### Physical Design Flow

- Partitioning
  將整個設計拆分成較小的模組或區塊，以便於後續的設計。
- Floorplanning
  確定各個功能模組 (Functional Unit Block) 的位置
- Placement
  將標準單元（Standard Cells）、IPs 放到前面的 Functional Unit Block 裡面，常常跟 Floorplanning 一起做。
- Clock Tree Synthesis
  讓所有 Clock 訊號能夠同步傳遞到各個元件
- Routing
  根據 Netlist 和 Placement 的資訊，把元件之間的連線接起來
- Post-routing Optimization
- Compaction
  早期才有，把 Placement 的結果做最佳化，現在都直接在 Placement 決定面積要多大
- Extraction & Verification

不同步驟之間常常會有 feedback loop

#### IP (Intellectual Property)

- Hard IP
  通常是一個完整的功能模組，例如：CPU、GPU、DDR Controller
- Soft IP
  通常是一個功能模組的 RTL Code，Layout 還沒決定，可以根據不同的製程和需求做修改

#### Moore's Law

每隔 18-24 個月，晶片上的元件數量會增加一倍

##### More Moore

依賴於先進製程技術的推進（7nm → 5nm → 3nm → 2nm）。

- FinFET → GAAFET（環繞閘極電晶體）→ CFET (互補場效應電晶體)
- EUV (Extreme Ultraviolet Lithography)
- 先進封裝技術，Chiplet、3D IC

##### More than Moore

專注於縮小電晶體尺寸

- Compute-in-Memory

### VLSI Design Considerations

- Design Complexity
- Performance
- Time-to-Market
- Cost: Die Area, Packaging, Testing
- Power Consumption、Noise、Reliability

考慮到不同的目標，會有不同的設計方法，像是：`Full Custom Design`, `Standard Cell Design`, `Gate Array Design`, `FPGA`, `CPLD`, `SPLD`, `SSI`

#### Full Custom Design

完全自訂，可以達到最佳的 PPA，但是花費時間和金錢最多

#### Standard Cell Design

![Standard Cell Design](./images/vlsi-physical-design-automation/StandardCellDesign.png)
有一個 `Cell Library`，裡面有很多標準元件，每個都有固定的高度

早期 Metal 層數不多，可以留 Routing Channel、Feedthrough Cell 來連接不同的 Cell。現在層數比較多，連線都在上空，所以可以把整 Row 的 Cell 翻轉，讓 GND 在一邊、VDD 在另一邊，減少 Routing 的複雜度

#### Gate Array Design

![Gate Array Design](./images/vlsi-physical-design-automation/GateArrayDesign.png)

Cell 裡面、Cell 之間的連線都沒有決定，可以根據需求來做 (沒什麼人在用?

#### FPGA (Field Programmable Gate Array)

![FPGA](./images/vlsi-physical-design-automation/FPGA.png)

可以決定每個 Cell 的功能，線也連好了，線可以用 Switch、Switch Box 控制

##### LUT (Look-Up Table)

把某個計算過程的所有 Input 組合對應的 Output 存起來，這樣就不用每次都重新計算

|                  | Full Custom | Standard Cell | Gate Array |     FPGA     |     SPLD     |
| :--------------: | :---------: | :-----------: | :--------: | :----------: | :----------: |
|    Cell Size     |  variable   | fixed height  |   fixed    |    fixed     |    fixed     |
|    Cell Type     |  variable   |   variable    |   fixed    | programmable | programmable |
|  Cell Placement  |  variable   |    in row     |   fixed    |    fiexed    |    fixed     |
| Interconnections |  variable   |   variable    |  variable  | programmable | programmable |

高度的單位通常用 `Track` (**T**) 表示

##### Macro Cells

包含數個 `Standard Cells`，感覺有點像 `IP`

##### Structured ASIC

> 越低層的 Metal，RC 的表現越差，Timing 會比較長，上層表現比較好

介於 `Gate Array` 和 `FPGA` 之間，有幾層 Layer 已經是 Pre-Defined，還有一些 Layer 可以自己設計。

流行於 `Engineering Change Order (ECO)`，只需要改某些 Layer，不用全部重新做

##### Design Rule Checking (DRC)

- Size Rules
- Spacing (Separation) Rules
  Minimum spaceing 不是一個常數，要考慮到兩條相鄰的線重疊的長度來決定
- OVerlap Rules
  每一層 Metal 都會分別做光照，上下兩層的 Overlapping 面積不能太小，光照的時候沒辦法 Align

## Partitioning

比較重要的 edge 或是 input 有數個的 output，讓他們的 weight 大一點。
