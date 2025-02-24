---
title: 人工智慧運算架構與系統 (一) 筆記
date: 2025-02-20 01:11:23
tags: [AI, Computing]
category: AI
math: true
---

> 單純記錄課堂筆記，不保證正確性。

# 第一週 Everything is a number

##　 Lecture

- Base Conversion
  可以直接用 gdb 來看數字的表示方式，例如

  ```shell
    (gdb) p/t 43275
    $1 = 1010100100001011
    (gdb) p 0b1010100100001011
    $2 = 43275
    (gdb) p/x 43275
    $3 = 0xa90b
    (gdb) p 0xa90b
    $4 = 43275
  ```

  這樣就可以看到 43275 的二進位、十進位、十六進位表示方式。

- Digital & Analog
  - Quantization: 把類比轉成數位訊號。
    會先在時間軸上做 Sampling，再對 Sample 出的訊號強度做數值 Mapping。

## Lab1

### Python

### C/C++

#### Bitwise Operation

多用 `bitwise` 增加 `readability` 和 `performance`。

```cpp
#define BIT_MASK(start, end) (((~0U) >> (31 - (start))) & ~((1U << (end)) - 1))
#define GET_ADDR_BITS(addr, start, end)  (((addr) & BIT_MASK(start, end)) >> (end))

int decode( uint32_t addr){

  if (GET_ADDR_BITS(addr, 31, 8) == 0){
      // 0x00000000 - 0x000000FF
      return VIRT_DEBUG;
  } else if (GET_ADDR_BITS(addr, 31, 16) == 0 &&
             GET_ADDR_BITS(addr, 16, 12) !=0){
      // 0x00001000 - 0x0000FFFF
  }
  ....
}
```

- `BIT_MASK(start, end)` 會產生從 start 到 end 都是 1 的 mask
  - `(~0U) >> (31 - (start))` 會產生從 start 和 start 之後的 bit 都是 1 的 mask
  - `~((1U << (end)) - 1)` 會產生 end 之後的 bit 都是 0 的 mask

方便對以下做 decode

![Example](https://course.playlab.tw/md/uploads/5d04d2c9-3cfb-4222-bddb-bdfddc6483cf.png)

#### Valgrind

C 語言沒有 garbage collection，記憶體要自己控制，可以用 `valgrind` 來檢查記憶體是否有釋放。

> 編譯時要加上 `-g` 參數，才能看到程式碼的行數。

```shell
valgrind {-valgrind parameter} ./my_program {-program parameter}
```

Valgrind 訊息可以分成幾種：

- `definitely lost`: 直接遺失，程式完全無法訪問這塊記憶體

  ```c
  #include <stdlib.h>
  int main() {
    int* ptr = malloc(10 * sizeof(int));
    return 0;
  }
  ```

  ```shell
  ==346==    definitely lost: 40 bytes in 1 blocks
  ==346==    indirectly lost: 0 bytes in 0 blocks
  ==346==      possibly lost: 0 bytes in 0 blocks
  ==346==    still reachable: 0 bytes in 0 blocks
  ==346==         suppressed: 0 bytes in 0 blocks
  ```

- `indirectly lost`: 因為其他記憶體被洩漏而無法釋放

  ```c
  #include <stdlib.h>

  typedef struct Node {
      int value;
      struct Node *next;
  } Node;

  void leak_linked_list() {
      Node *head = (Node *)malloc(sizeof(Node));
      head->next = (Node *)malloc(sizeof(Node));
      head->next->next = (Node *)malloc(sizeof(Node));
      head->next->next->next = NULL;

      free(head);
  }

  int main() {
      leak_linked_list();
      return 0;
  }
  ```

  ```shell
  ==334== LEAK SUMMARY:
  ==334==    definitely lost: 16 bytes in 1 blocks
  ==334==    indirectly lost: 16 bytes in 1 blocks
  ==334==      possibly lost: 0 bytes in 0 blocks
  ==334==    still reachable: 0 bytes in 0 blocks
  ==334==         suppressed: 0 bytes in 0 blocks
  ```

- `possibly lost`: 有奇怪的操作，valgrind 也不確定這塊記憶體是不是真的 `lost`
- `still reachable`: 記憶體在程式結束時仍然可訪問，但沒有 free 掉

  ```c
  #include <stdlib.h>

  int *global_ptr;

  void still_reachable_memory() {
      global_ptr = (int *)malloc(10 * sizeof(int));
  }

  int main() {
      still_reachable_memory();
      return 0;
  }
  ```

  ```shell
  ==406== LEAK SUMMARY:
  ==406==    definitely lost: 0 bytes in 0 blocks
  ==406==    indirectly lost: 0 bytes in 0 blocks
  ==406==      possibly lost: 0 bytes in 0 blocks
  ==406==    still reachable: 40 bytes in 1 blocks
  ==406==         suppressed: 0 bytes in 0 blocks
  ```

### Verilog

- Combinational Circuit

輸出只取決於當前的輸入，與電路中的狀態無關。根據 `input signal` 的組合，直接計算出 `output signal` 的邏輯函數，過程中不會有任何的 `memory`。

- Sequential Circuit

輸出不僅取決於當前的輸入，還取決於電路中留存的狀態。在 `combinational circuit` 的基礎上，加入了 `memory` 元件，例如 `flip-flop`。

#### Verilog Syntax

##### Data Type

- `wire`: 用來連接不同的元件，只能用來連接 `output` 和 `input`，沒有特別註明的話，預設就是 `wire`。
- `reg`: 用來儲存 `state`，可以用來儲存 `output`。
  - 對於 `module` 內部而言，input 只能用 `wire` 連接，output 可以用 `wire` 或 `reg` 連接。
  - 對於 `module` 外部而言，input 可以用 `wire` 或 `reg` 連接，output 只能用 `wire` 連接。

當 `wire` 沒有被 `assign` 時，預設值為 **X(Unknown)**，當 `reg` 沒有被 `assign` 時，預設值為 **Z(High Impedance/ Floating)**。

##### Vector

`[MSB:LSB]` 或 `[LSB:MSB]` 都可以

```verilog
input  [7:0]   in;         // Declare 8-bit input wire `in`
output [127:0] out;        // Declare 128-bit output wire `out`
reg    [3:0]   reg0, reg1; // Declare 4-bit register `reg0` and `reg1`
wire   [63:0]  wire0;      // Declare 64-bit wire `wire0`
```

##### Array

要把宣告的 `Dimension` 寫在變數名稱的後面

```verilog
wire wire0 [7:0];               // Declare 8 1-bit wire `wire0`
reg [127:0] reg0 [3:0] [3:0];   // Declare 4*4(double arrays) 128-bit register `reg0`
```

##### Data Assignment

在 HDL 如 Verilog 中最常使用到的表示方法通常是 2 進制以及 16 進制，不過為了方便也可以用 10 進制表示。另外可以加上 `_` 來增加可讀性。

```verilog
wire [4:0]  wire0 = 5'b01101;             // Binary
wire [7:0]  wire1 = 22;                   // Decimal(default), equals to 8'd22
reg  [11:0] reg0  = 12'b0000_1111_0000;   // Divide with bottom line per 4-bit for readability
reg  [3:0]  reg1  = 4'hf;                 // Hexadecimal, equals to 4'b1111
```

##### Dataflow Level

- Data assignment

  - `assign LValue = RValue`
    LValue 只可以是 `wire`，但是 RValue 可以是 `wire` 或是 `reg`
    - 通常用在 assign module output port

- Concatenation operator

  - `LValue = {Concat0, Concat1, ...}`
    - `Concat0`, `Concat1` 可以是 `wire`、`reg` 或直接寫數字，像是 `8'b1010_1010`

- Replication operator
  - `LValue = {N{Pattern}}`
    ```verilog
    wire [31:0] combine;
    assign combine = {3{in0}, 2{in1}, 12'hfaa};
    // combine = 00_in0_in0_in0_in1_in1_1111_1010_1010
    ```

##### Behavior Level

主要就是用 `always block`，但裡面的 Data Assignment **只能對 `reg` 進行操作**，Combinational Circuit 或 Sequential Circuit 都可以用。

- Combinational Circuit

  ```verilog
  module combinational_circuit (
      input A,
      input B,
      output reg C
  );
      always @(*) begin
          // Combinational logic: output Y is the result of ANDing A and B
          C = A & B;
      end
  endmodule
  ```

  - `always @(*)`: 當所有的 `input` 有變動時，就會執行 `always block` 裡面的內容。
  - 可以看到 `C` 要用 `reg` 宣告，因為 `always block` 裡面只能對 `reg` 進行操作。

- Sequential Circuit

  ```verilog
    module sequential_circuit (
      input clk,
      input reset,
      input data,
      output reg out
  );
      always @(posedge clk or posedge reset) begin
          if (reset) begin
              // Asynchronous reset: reset out to 0 when reset is asserted
              out <= 1'b0;
          end else begin
              // Sequential logic: out is updated on the rising edge of clk
              out <= data;
          end
      end
  endmodule
  ```

  - `always @(posedge clk or posedge reset)`: 當 `clk` 或 `reset` 有上升沿時，就會執行 `always block` 裡面的內容。
  - `<=`: Non-blocking assignment，用在 `Sequential Circuit` 中

- Blocking
  描述的順序會影響電路執行結果，執行 assignment 時，會按照描述順序逐一執行，下一個 assignment 會等待前一個 assignment 操作完成。

  ```verilog
  always @(*) begin
    A = 2;      // A = 2
    B = C;      // B = C
    C = A;      // C = 2
    // These assignments will be executed in order
  end
  ```

- Non-blocking
  Parallel 的進行資料傳遞，描述的順序不會影響電路執行結果。
  ```verilog
  always @(posedge clk) begin
    A <= 2;     // A = 2
    B <= C;     // B = C
    C <= A;     // C = A
    // These assignments will be executed at the same time
  end
  ```

##### Module Connection

```verilog
  module top_module (
      input  in0,
      input  in1,
      output out0
  );
      sub_module sub_module0 (
          .input0(in0),   // Connect to input0
          .input1(in1),   // Connect to input1
          .output0(out0)  // Connect to output0
      );
  endmodule

  module sub_module (
      input  input0,
      input  input1,
      output output0
  );
      // Skip...
  endmodule
```

盡量用 `named connection`，增加程式碼的可讀性。開心的話也可以用 `ordered connection`。

其他像是 `If Statement`、`Case Statement` 或 `? :` 本質上都是 `MUX`，跟軟體有差

# 第二週 AI Models

生成式 AI 模型訓練成本居高不下，很多學者會針對模型的演算法去優化，降低運算量

`Zero-Shot` 或 `Few-Shot` 可以讓模型在沒有看過任何樣本的情況下做出預測，是 `General Purpose` 的模型，可以應用在各種不同的任務上。在 Training 的時候都是用 `Unsupervised Learning`，不需要標註的資料。

- Grouped Convolution Layer
- Depthwise Convolution Layer
- Normalization Layer
  - Batch Normalization
  - Layer Normalization
  - Instance Normalization
  - Group Normalization

## Lecture

### AlexNet

![AlexNet](./images/machine-learning/AlexNet.png)

### VGG-16

![VGG-16](./images/machine-learning/VGG-16.png)

### ResNet-50

![ResNet-50](./images/machine-learning/ResNet-50.png)

### MobileNetV2

![MobileNetV2](./images/machine-learning/MobileNetV2.png)

[Source](https://www.dropbox.com/scl/fi/2qx0cfz7vim0fdhrmy986/lec02.pdf?rlkey=wdjw92hwohp4bhyos8wf5iinb&e=2&dl=0)

#### Model Analysis

- Latency: 一個任務完成所需的時間
- Throughput: 一個時間內完成的任務數量
  `Latency` 跟 `Throughput` 沒有絕對的關聯，優化 `Latency` 會更難一些
- Power Consumption
  不同 Building Block 的 Power Consumption 也不同，像是 floating point operation 會比 integer operation 耗電量高、DRAM (off-chip) 耗電量也比 SRAM 高
- Number of Parameters
- Model Size
  `Model Size` = `Number of Parameters` \* `Bit Width`
- Total/Peak Number of Activations
  - `Peak Number of Activations` 成為一個系統能不能跑起來的關鍵 (Inference)
  - Early Layer 的 `Activations` 會比較多，後面的 `Activations` 會比較少，`Weight` 會比較大
- MACs (Multiply-Accumulate Operations)
- FLOPs (Floating Point Operations)
  - `FLOPs` = `MACs` \* `2` (一個 `MAC` 會有兩次 `FLOPs`)
  - `FLOPS` = `FLOPs` / `second`
- Roofline Model

## Lab2

`ONNX`(Open Neural Network Exchange)，可以讓不同的深度學習框架之間進行模型的轉換，例如 `PyTorch`、`TensorFlow`、`Caffe2`、`MXNet` 等。
