---
title: AIAS 筆記
date: 2025-03-03 17:56:23
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

## Lab

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

[EfficientML.ai Lecture 02: Basics of Neural Networks](https://www.dropbox.com/scl/fi/2qx0cfz7vim0fdhrmy986/lec02.pdf?rlkey=wdjw92hwohp4bhyos8wf5iinb&e=2&dl=0)

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
- MACs (Multiply-Accumulate Operations)，一次乘法和一次加法
- FLOPs (Floating Point Operations)
  - `FLOPs` = `MACs` \* `2` (一個 `MAC` 會有兩次 `FLOPs`)
  - `FLOPS` = `FLOPs` / `second`
- Roofline Model

## Lab

`ONNX`(Open Neural Network Exchange)，可以讓不同的深度學習框架之間進行模型的轉換，例如 `PyTorch`、`TensorFlow`、`Caffe2`、`MXNet` 等。

### Pytorch

#### PyTorch Model 轉換成 ONNX

```python
import torch
import torchvision.models as models

# 下載並載入預訓練的 AlexNet 模型
model = models.alexnet(pretrained=True)

# 創建一個隨機的輸入張量（Dummy Input），形狀為 (10, 3, 224, 224)
# 代表 10 張 RGB 影像，每張大小為 224x224
dummy_input = torch.randn(10, 3, 224, 224)

# 定義 ONNX 模型的輸入名稱
# 第一個輸入名稱為 "actual_input_1"（實際的輸入）
# 後面 16 個 "learned_X" 其實是多餘的，通常用於標記權重或內部變數（但這裡不需要）
input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]

# 定義 ONNX 模型的輸出名稱
# 這裡只設定一個輸出名稱 "output1"，代表分類結果（1000 個類別的機率分佈）
output_names = ["output1"]

# 將 PyTorch 的 AlexNet 模型轉換為 ONNX 格式，並儲存為 "models/alexnet.onnx"
torch.onnx.export(
    model,            # PyTorch 模型
    dummy_input,      # 範例輸入，用來確保模型的輸入形狀正確
    "models/alexnet.onnx",  # 轉換後的 ONNX 模型儲存路徑
    verbose=True,     # 顯示轉換過程的詳細資訊
    input_names=input_names,  # 設定 ONNX 模型的輸入名稱
    output_names=output_names  # 設定 ONNX 模型的輸出名稱
)
```

#### Model Analysis

- 取得 Parameter Size

  ```python
  total_params = sum(p.numel() for p in model.parameters())
  print("Total number of parameters: ", total_params)
  ```

  印出這個 Model 總共的參數數量，其中 `p.numel()` 會回傳某個 Tensor 裡面總共有多少個元素

- 算出總共需要多少 Memory

  ```python
  param_size = sum(p.numel() * p.element_size() for p in model.parameters())
  print("Total memory for parameters: ", param_size)
  ```

  `p.element_size()` 會回傳某個 Tensor 裡面的 Data Type 佔用多少 Bytes

- Summary

  ```python
  from torchvision import models
  from torchsummary import summary

  model = models.alexnet(pretrained=True)
  summary(model, (3, 224, 224))
  ```

  這樣可以印出這個 Model 的 Summary，包含模型的結構、每層輸出的大小、參數數量、總參數數量等等

  ```
  ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 64, 55, 55]          23,296
                  ReLU-2           [-1, 64, 55, 55]               0
            MaxPool2d-3           [-1, 64, 27, 27]               0
                Conv2d-4          [-1, 192, 27, 27]         307,392
                  ReLU-5          [-1, 192, 27, 27]               0
            MaxPool2d-6          [-1, 192, 13, 13]               0
                Conv2d-7          [-1, 384, 13, 13]         663,936
                  ReLU-8          [-1, 384, 13, 13]               0
                Conv2d-9          [-1, 256, 13, 13]         884,992
                ReLU-10          [-1, 256, 13, 13]               0
              Conv2d-11          [-1, 256, 13, 13]         590,080
                ReLU-12          [-1, 256, 13, 13]               0
            MaxPool2d-13            [-1, 256, 6, 6]               0
    AdaptiveAvgPool2d-14            [-1, 256, 6, 6]               0
              Dropout-15                 [-1, 9216]               0
              Linear-16                 [-1, 4096]      37,752,832
                ReLU-17                 [-1, 4096]               0
              Dropout-18                 [-1, 4096]               0
              Linear-19                 [-1, 4096]      16,781,312
                ReLU-20                 [-1, 4096]               0
              Linear-21                 [-1, 1000]       4,097,000
    ================================================================
    Total params: 61,100,840
    Trainable params: 61,100,840
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 8.38
    Params size (MB): 233.08
    Estimated Total Size (MB): 242.03
    ----------------------------------------------------------------
  ```

  如果單純用 `print(model)` 只會印出這個 Model 的結構，但是不會有其他資訊

  ```
  AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
  )
  ```

- torchinfo
  如果需要更更更詳細的模型資訊，可以使用 `torchinfo`，控制格式和詳細程度 (col_names、verbose)

  ```python
  import torchinfo
  torchinfo.summary(model, (3, 224, 224), batch_dim=0, col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"), verbose=0)
  ```

### TensorFlow

TensorFlow 有很多儲存 Model 的格式，這些都可以用 Tensorflow-onnx 工具轉換成 ONNX 格式

- Checkpoint (.ckpt)
  TensorFlow 會用 `checkpoint` 儲存 Model 訓練的權重，但是不包含 Computation Graph，之後可以繼續訓練

- Frozen Graph (.pb)
  把 Graph 和權重都儲存起來，可以直接用來做 Inference，但是權重是 `Frozen Weight`，不能再繼續訓練 (TensorFlow 2.x 不推薦這種方式)

- SavedModel (包含 saved_model.pb、variables、assets)
  是 TensorFlow 推薦的完整儲存格式，封裝了 Model 架構、Graph、權重和 Inference 的方式，不用額外指定輸入輸出，更容易導入 ONNX

- TFLite (.tflite)
  TensorFlow Lite 是為了在行動裝置或嵌入式上做 Inference 而設計的，可以將 SavedModel 轉換成 TFLite，並且可以做 Quantization 來減少 Model 的大小

#### 把 TensorFlow Model 轉換成 ONNX

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# 定義一個簡單的 CNN 模型
def create_simple_cnn_model(input_shape=(224, 224, 3), num_classes=10):
    model = tf.keras.Sequential([  # 使用 Keras Sequential 模型
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # 第一層 Convolution Layer，32 個 3x3 Filter
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),  # Fully Connected Layer，128 個 Neurons
        Dense(num_classes, activation='softmax')  # 輸出層，10 個 Neurons（10 個類別）
    ])
    return model

directory = "models/tf_cnn_models"
if not os.path.exists(directory):
    os.makedirs(directory)

simple_cnn_model = create_simple_cnn_model()

# 儲存模型為 TensorFlow 的 SavedModel 格式
tf.saved_model.save(simple_cnn_model, directory)
```

最後可以用以下指令把 SavedModel 轉換成 ONNX

```shell
!python3 -m tf2onnx.convert --saved-model models/tf_cnn_models --output models/tf2onnx_cnn_model.onnx
```

可以用 `--opset` (Operator Set Version) 來指定使用的 ONNX Operator 版本，越新的支援越多，預設是 `9`

| opset | ONNX Version |                            Description                            |
| :---: | :----------: | :---------------------------------------------------------------: |
|   9   |     1.4      |             Default version, includes basic operators             |
|  10   |     1.5      |   Supports `Slice` improvements, new `QuantizeLinear` operator    |
|  11   |     1.6      |     Supports `Loop` and `Range`, optimized for dynamic length     |
|  12   |     1.7      |           Adds `GatherND`, improved `Reshape` operator            |
|  13   |     1.8      |  `Softmax`, `ReduceMean`, etc. support more flexible dimensions   |
|  14   |     1.9      |       Improved `ConvTranspose`, supports more data formats        |
|  15+  |    1.10+     | Adds more new features (e.g., `Reshape` with variable dimensions) |

```shell
python -m tf2onnx.convert --saved-model <tensorflow_model_name> --opset 13 --output <onnx_model_name>
```

### ONNX

#### Inferencing

```python
import onnxruntime as ort
import numpy as np

# 載入 ONNX 模型
onnx_session = ort.InferenceSession("models/tf2onnx_cnn_model.onnx")

# Test data
test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Inference
onnx_input_name = onnx_session.get_inputs()[0].name
onnx_output = onnx_session.run(None, {onnx_input_name: test_input})
```

### Netron

Netron 是一個功能強大的機器學習模型的 **可視化** 工具，支援各種模型格式，包括 ONNX、TensorFlow、PyTorch、Keras 等等，可以用來看模型每一層的結構、輸入輸出、權重等等。可以用指令或 [官方網站](https://netron.app/) 來開啟模型

```shell
netron models/lenet.onnx
```

![LetNet Architecture](./images/ai-computing-system/LeNet.png)

### Protobuf

ONNX 格式將模型儲存為 Protobuf（Protocol Buffers）的結構，其中 Protobuf 是一種 Google 開發的 **Data Serialization Format**，可以將 ONNX 模型的 Graph、Layer 結構、權重等資訊儲存在裡面

```python
import onnx

onnx_model = onnx.load('./models/lenet.onnx')

# The model is represented as a protobuf structure and it can be accessed
# using the standard python-for-protobuf methods

## list all the operator types in the model
node_list = []
count = []
for i in onnx_model.graph.node:
    if (i.op_type not in node_list):
        node_list.append(i.op_type)
        count.append(1)
    else:
        idx = node_list.index(i.op_type)
        count[idx] = count[idx]+1
print(node_list)
print(count)
```

```
['Reshape', 'Conv', 'Add', 'Relu', 'MaxPool', 'Identity']
[4, 4, 4, 3, 2, 1]
```

這樣可以看到這個 ONNX Model 有哪些 Operator Type，以及每個 Operator Type 的數量。

可以參考 [onnx.proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto) 和 [Protocol Buffers Documentation](https://protobuf.dev/programming-guides/proto3/)，能看到 ONNX 最外層的結構是 ModelProto，裡面包含 GraphProto，其由 NodeProto 組成，包含 Graph 的許多資訊

```python
# find the IR version
print(onnx_model.ir_version)
## find the computation graph
print(onnx_model.graph)
## find the number of inputs
print(len(onnx_model.graph.input))
## find the number of nodes in the graph
print(len(onnx_model.graph.node))
```

以下方式可以印出這個 ONNX Model 的 Convolution Layer 輸入輸出的 Size，其中

- input_nlist: 模型所有的輸入，包含 Placeholder 和 Initializer
- initializer_nlist: 模型所有的權重，是 input_nlist 的子集
- value_info_nlist: 模型中間的計算結果

```python
## parse_model.py
import onnx

onnx_model = onnx.load('./models/lenet.onnx')

## need to run shape inference in order to get a full value_info list
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

## List all tensor names in the graph
input_nlist = [k.name for k in onnx_model.graph.input]
initializer_nlist = [k.name for k in onnx_model.graph.initializer]
value_info_nlist = [k.name for k in onnx_model.graph.value_info]

print('\ninput list: {}'.format(input_nlist))
print('\ninitializer list: {}'.format(initializer_nlist))
print('\nvalue_info list: {}'.format(value_info_nlist))

## a simple function to calculate the tensor size and extract dimension information
def get_size(shape):
    dims = []
    ndim = len(shape.dim)
    size = 1;
    for i in range(ndim):
        size = size * shape.dim[i].dim_value
        dims.append(shape.dim[i].dim_value)
    return dims, size

## find all `Conv` operators and print its input information
for i in onnx_model.graph.node:
    if (i.op_type == 'Conv'):
        print('\n-- Conv "{}" --'.format(i.name))
        for j in i.input:
            if j in input_nlist:
                idx = input_nlist.index(j)
                (dims, size) = get_size(onnx_model.graph.input[idx].type.tensor_type.shape)
                print('input {} has {} elements dims = {}'.format(j, size, dims  ))
            elif j in initializer_nlist:
                idx = initializer_nlist.index(j)
                (dims, size) = get_size(onnx_model.graph.initializer[idx].type.tensor_type.shape)
                print('input {} has {} elements dims = {}'.format(j, size, dims))
            elif j in value_info_nlist:
                idx = value_info_nlist.index(j)
                (dims, size) = get_size(onnx_model.graph.value_info[idx].type.tensor_type.shape)
                print('input {} has {} elements dims = {}'.format(j, size, dims))
```

這樣可以除了可以印出 input、initializer、value_info 的名稱，還可以印出所有 Convolution Layer 需要的輸入輸出大小

```txt
-- Conv "import/conv1first/Conv2D" --
input import/Placeholder:0 has 784 elements dims = [1, 1, 28, 28]
input import/conv1first/Variable:0 has 800 elements dims = [32, 1, 5, 5]

-- Conv "import/conv2/Conv2D" --
input import/pool1/MaxPool:0 has 6272 elements dims = [1, 32, 14, 14]
input import/conv2/Variable:0 has 51200 elements dims = [64, 32, 5, 5]

-- Conv "import/conv3/Conv2D" --
input import/pool2/MaxPool:0 has 3136 elements dims = [1, 64, 7, 7]
input import/conv3/Variable:0 has 3211264 elements dims = [1024, 64, 7, 7]

-- Conv "import/conv4last/Conv2D" --
input import/conv3/Relu:0 has 1024 elements dims = [1, 1024, 1, 1]
input import/conv4last/Variable:0 has 10240 elements dims = [10, 1024, 1, 1]
```

### Hooks

Hooks 是 PyTorch 中的一個功能，可以註冊在 Forward Pass 或 Backward Pass 中，用來觀察模型的中間過程，例如輸入、輸出、權重、梯度等等，用來檢視每一層算完後到底發生什麼事

```python
import torchvision.models as models
import torch
activation = {}

# Define a hook function
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = models.alexnet(pretrained=True)
model.eval()

# Register hook to each linear layer
for layer_name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Linear):
        # Register forward hook
        layer.register_forward_hook(get_activation(layer_name))

# Run model inference
data = torch.randn(1, 3, 224, 224)
output = model(data)

# Access the saved activations
for layer in activation:
    print(f"Activation from layer {layer}: {activation[layer].shape}")
```

這樣可以印出每個 Fully-connected Layer 的 Activation Shape，在 inference 的過程每一層的 Activation 都會被存起來，用來觀察每一層的輸出

```txt
Activation from layer classifier.1: torch.Size([1, 4096])
Activation from layer classifier.4: torch.Size([1, 4096])
Activation from layer classifier.6: torch.Size([1, 1000])
```

輸出的大小分別是 `[1, 4096]`、`[1, 4096]`、`[1, 1000]`。

### MACs and FLOPs

這是計算 AlexNet MACs 的例子，計算裡面的 Convolutions 和 Fully-connected Layers 總共的 MACs 數量

```python
import torch
import torchvision.models as models
import torch.nn as nn

def calculate_output_shape(input_shape, layer):
    # Calculate the output shape for Conv2d, MaxPool2d, and Linear layers
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        kernel_size = (
            layer.kernel_size
            if isinstance(layer.kernel_size, tuple)
            else (layer.kernel_size, layer.kernel_size)
        )
        stride = (
            layer.stride
            if isinstance(layer.stride, tuple)
            else (layer.stride, layer.stride)
        )
        padding = (
            layer.padding
            if isinstance(layer.padding, tuple)
            else (layer.padding, layer.padding)
        )
        dilation = (
            layer.dilation
            if isinstance(layer.dilation, tuple)
            else (layer.dilation, layer.dilation)
        )

        output_height = (
            input_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1
        output_width = (
            input_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1
        return (
            layer.out_channels if hasattr(layer, "out_channels") else input_shape[0],
            output_height,
            output_width,
        )
    elif isinstance(layer, nn.Linear):
        # For Linear layers, the output shape is simply the layer's output features
        return (layer.out_features,)
    else:
        return input_shape


def calculate_macs(layer, input_shape, output_shape):
    # Calculate MACs for Conv2d and Linear layers
    if isinstance(layer, nn.Conv2d):
        kernel_ops = (
            layer.kernel_size[0]
            * layer.kernel_size[1]
            * (layer.in_channels / layer.groups)
        )
        output_elements = output_shape[1] * output_shape[2]
        macs = int(kernel_ops * output_elements * layer.out_channels)
        return macs
    elif isinstance(layer, nn.Linear):
        # For Linear layers, MACs are the product of input features and output features
        macs = int(layer.in_features * layer.out_features)
        return macs
    else:
        return 0

model = models.alexnet(pretrained=True)

# Initial input shape
input_shape = (3, 224, 224)
total_macs = 0

# Iterate through the layers of the model
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Linear)):
        output_shape = calculate_output_shape(input_shape, layer)
        macs = calculate_macs(layer, input_shape, output_shape)
        total_macs += macs
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            print(
                f"Layer: {name}, Type: {type(layer).__name__}, Input Shape: {input_shape}, Output Shape: {output_shape}, MACs: {macs}"
            )
        elif isinstance(layer, nn.MaxPool2d):
            # Also print shape transformation for MaxPool2d layers (no MACs calculated)
            print(
                f"Layer: {name}, Type: {type(layer).__name__}, Input Shape: {input_shape}, Output Shape: {output_shape}, MACs: N/A"
            )
        input_shape = output_shape  # Update the input shape for the next layer

print(f"Total MACs: {total_macs}")
```

輸出如下

```txt
Layer: features.0, Type: Conv2d, Input Shape: (3, 224, 224), Output Shape: (64, 55, 55), MACs: 70276800
Layer: features.2, Type: MaxPool2d, Input Shape: (64, 55, 55), Output Shape: (64, 27, 27), MACs: N/A
Layer: features.3, Type: Conv2d, Input Shape: (64, 27, 27), Output Shape: (192, 27, 27), MACs: 223948800
Layer: features.5, Type: MaxPool2d, Input Shape: (192, 27, 27), Output Shape: (192, 13, 13), MACs: N/A
Layer: features.6, Type: Conv2d, Input Shape: (192, 13, 13), Output Shape: (384, 13, 13), MACs: 112140288
Layer: features.8, Type: Conv2d, Input Shape: (384, 13, 13), Output Shape: (256, 13, 13), MACs: 149520384
Layer: features.10, Type: Conv2d, Input Shape: (256, 13, 13), Output Shape: (256, 13, 13), MACs: 99680256
Layer: features.12, Type: MaxPool2d, Input Shape: (256, 13, 13), Output Shape: (256, 6, 6), MACs: N/A
Layer: classifier.1, Type: Linear, Input Shape: (256, 6, 6), Output Shape: (4096,), MACs: 37748736
Layer: classifier.4, Type: Linear, Input Shape: (4096,), Output Shape: (4096,), MACs: 16777216
Layer: classifier.6, Type: Linear, Input Shape: (4096,), Output Shape: (1000,), MACs: 4096000
Total MACs: 714188480
```

> Some layers in GoogleNet enable ceil_mode, which will affect the calculation formula for output_shape.

### Profiling

```python
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.alexnet(pretrained=True)
model.eval()  # Set the model to evaluation mode
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

```

這樣可以印出這個 Model 的 Profiling 結果，包含每一個 Operator 的 CPU Time、Memory Usage、Input Shape、Output Shape 等等

```txt
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                      aten::empty         0.11%      60.000us         0.11%      60.000us       5.455us       9.25 Mb       9.25 Mb            11
    aten::max_pool2d_with_indices         3.52%       1.838ms         3.52%       1.838ms     612.667us       5.05 Mb       5.05 Mb             3
                    aten::resize_         0.01%       6.000us         0.01%       6.000us       1.000us     180.00 Kb     180.00 Kb             6
                      aten::addmm        36.23%      18.939ms        36.32%      18.984ms       6.328ms     179.53 Kb     179.53 Kb             3
                     aten::conv2d         0.07%      36.000us        58.54%      30.601ms       6.120ms       9.25 Mb           0 b             5
                aten::convolution         0.21%     111.000us        58.47%      30.565ms       6.113ms       9.25 Mb           0 b             5
               aten::_convolution         0.10%      54.000us        58.26%      30.454ms       6.091ms       9.25 Mb           0 b             5
         aten::mkldnn_convolution        57.99%      30.313ms        58.15%      30.400ms       6.080ms       9.25 Mb           0 b             5
                aten::as_strided_         0.05%      24.000us         0.05%      24.000us       4.800us           0 b           0 b             5
                      aten::relu_         0.34%     178.000us         0.86%     447.000us      63.857us           0 b           0 b             7
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 52.275ms
```

### TensorBoard

TensorBoard 可以監控和分析機器學習模型的訓練過程，監測 CPU 和 GPU 的使用情況，看看 Bottleneck 在哪個地方

下面程式碼是用 CIFAR-10 Dataset 進行 ResNet-18 訓練，用 torch.profiler 儲存 Profile 結果

```python
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T


transform = T.Compose(
    [T.Resize(224),
     T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

device = torch.device("cpu")
model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()

def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step, batch_data in enumerate(train_loader):
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        if step >= 1 + 1 + 3:
            break
        train(batch_data)
```

接下來可以用以下指令啟動 TensorBoard，監測 Profile 的結果

```shell
tensorboard --logdir='~/projects/lab02/lab2-3/log/' --bind_all --port=10000 > tensorboard.stdout.log &> tensorboard.stderr.log & # Start TensorBoard
kill $(ps -e | grep 'tensorboard' | awk '{print $1}') # Stop TensorBoard
```

![TensorBoard](./images/ai-computing-system/TensorBoard.png)

### Python C++ Frontend

#### TorchScript

TorchScript 是 PyTorch 的一種 IR (Intermediate Representation)，可以在其他環境執行，像是 C++、嵌入式設備或伺服器，有兩種方法可以把 PyTorch Model 轉換成 TorchScript

- Tracing
  用 `torch.jit.trace`，把 Example Input 丟進去模型跑，將這個 Data 在模型的流向全部記錄起來，藉此捕捉模型的結構

  ```python
  import torch

  class MyModel(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.fc1 = torch.nn.Linear(3, 3)
          self.fc2 = torch.nn.Linear(3, 2)

      def forward(self, x):
          x = self.fc1(x)
          x = self.fc2(x)
          return x

  model = MyModel()

  # Example Input
  example_input = torch.randn(1, 3)

  traced_model = torch.jit.trace(model, example_input)

  traced_model.save("traced_model.pt")
  ```

- Scripting
  用 `torch.jit.scrip` 直接解析 Python Code，轉換成 TorchScript

  ```python
  import torch

  class MyModel(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.fc1 = torch.nn.Linear(3, 3)
          self.fc2 = torch.nn.Linear(3, 2)

      def forward(self, x):
          x = self.fc1(x)
          x = self.fc2(x)
          return x

  scripted_model = torch.jit.script(MyModel())

  scripted_model.save("scripted_model.pt")
  ```

#### LibTorch

[LibTorch](https://pytorch.org/get-started/locally/) 可以用來在 C++ 環境中執行 TorchScript，可以用 CMake 來建置 C++ 程式，並且連結 LibTorch 的 Library 來執行 TorchScript

```cpp
#include <torch/script.h> // One-stop header.
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/ir/ir.h>

#include <iostream>
#include <memory>

using namespace torch::jit;

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  set_jit_logging_levels("GRAPH_DUMP");

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "load the torchscript model, " + std::string(argv[1]) + ", successfully \n";

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  //dump the model information
  // source code can be found in
  // https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/api/module.cpp
  std::cout << module.dump_to_str(true,false,false) << " (\n";
}

```

- Build & Compile

  ```shell
  ## create config
  $ cmake -DCMAKE_PREFIX_PATH=./libtorch ..

  ## compile
  $ cmake --build . --config Release -j ${nproc}
  ```

- Build 完後，可以跑 C++ 程式，並輸入剛剛轉出來的 TorchScript (.pt) 檔案

  ```shell
  ./analyzer ../../traced_resnet18.pt
  ```

可以參考 [PyTorch C++ API](https://pytorch.org/cppdocs/index.html) 了解更多 C++ API 的使用方式
