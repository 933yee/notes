### IC Design Flow

前端設計 (Front-end Design) → 後端設計 (Back-end / Physical Design) → Tape-out → 製造 → 測試 → 產品

Gate Array

- 裡面的 transistor 都做好了，但還沒有做 intra-cell routing 和 inter-cell routing
- 可以自己決定要把哪些 transistor 連接起來

因為現在 Metal 可以很多層，所以 Row 和 Row 之間不用再留 Routing Channel，都直接在 Cell 上面連線

FPGA

- 第 31 頁中間是 switch bus
- 主要是 Program 圖中的一堆 switch，設定成 0 或 1

- Physical Design
  把 Circuit Netlist 轉換成幾何圖形，並且將其放置在晶片上，並且將其連接起來。
  1. Floorplanning
  2. Placement
     現在 Floorplanning 和 Placement 通常是一起做的，所以有時候會被合併成一個步驟。差別在於 Floorplanning 是在整個晶片上規劃，Placement 是在晶片上的每個 Cell 規劃。
  3. Clock Tree Synthesis (CTS)
  4. Routing
  5. Layout
  6. Design Rule Check (DRC)
  7. Layout vs. Schematic (LVS)
- PPA
  Power, Performance, Area
