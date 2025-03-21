---
title: Chisel
date: 2025-03-22 00:01:21
tags: [hardware, chisel]
category: hardware
math: true
---

Chisel 是基於 Scala 語言設計的 HDL，用設計數位電路。

## Scala 語法

#### 變數、常數

- var: 變數
- val: 常數

#### If else

- if else: 會有 **回傳值** (最後一行)
  ```scala
  val x = if (a > b) a else b
  ```

#### 函數

- def: **函數名(參數名: 參數型別): 回傳型別**

  ```scala
  def max(a: Int, b: Int): Int = if (a > b) a else b

  def max(a: Int, b: Int): Int = {
    if (a > b) a
    else b
  }
  ```

- 函數裡面可以有其他子函數

  ```scala
    def max(a: Int, b: Int): Int = {
        def min(a: Int, b: Int): Int = if (a < b) a else b
        min(a, b)
    }
  ```

#### List

```scala
val list1 = List(1, 2, 3, 4, 5)
val list2 = 1 :: 2 :: 3 :: 4 :: 5 :: Nil // Nil: 代表 List 的結尾

var list3 = list1 ++ list2 // 合併 list
var m = list1.length // 長度
var n = list1.size // 長度
var x = list1.head // 取第一個元素
var z = list1(2) // 取第三個元素

```

#### for loop

```scala
for (i <- 1 to 10) { // 包含 10
    println(i)
}
for (i <- 1 until 10) { // 不包含 10
    println(i)
}
for (i <- 1 to 10 by 2) { // 間隔 2
    println(i)
}
for (i <- list1) { // iterator
    println(i)
}
```

#### class

```scala
class Point(xc: Int, yc: Int) {
    var x: Int = xc
    var y: Int = yc

    def move(dx: Int, dy: Int): Unit = {
        x = x + dx
        y = y + dy
    }

    override def toString: String = "(" + x + ", " + y + ")"
    println("Point x: " + x + ", y: " + y) // 每次 new 一個 Point 就會執行
}

val pt = new Point(10, 20)
val pt2 = new Point(xc = 10, yc = 20)
pt.move(10, 10)
```

## Chisel 語法

### Signal Types and Constants

- 有三種主要的資料型態：Bits、UInt、SInt

  - Bits: 用來表示固定長度的二進位數字
  - UInt: 用來表示無號整數
  - SInt: 用來表示有號整數

    ```scala
    val x = Bits(8.W) // 8-bit
    val y = UInt(8.W) // unsigned 8-bit
    val z = SInt(8.W) // signed 8-bit (two's complement)
    ```

- 可以看到整數的後面有 .W，是用來表示整數的長度

  - 8.W: 8-bit 整數，W 是 width 的縮寫
  - 8.U: 值為 8 的 UInt，U 是 unsigned 的縮寫，會自動推論有幾個 bit
  - 8.S: 值為 8 的 SInt，S 是 signed 的縮寫，會自動推論有幾個 bit
  - 3.U(4.W): 值為 3 的 UInt，原本只需要 2-bit，但是括號裡面指定他有 4-bit (3.U(4.W) = 0011)

- 如果括號裡面的數字沒有加 `.W`，會變成 bit position，且是 0 從 LSB 開始

  ```scala
  1.U(0) // 1
  1.U(1) // 0
  ```

- 如果不是十進位，宣告的時候要用 `"` 框住，並且加上 `b`、`h`、`o` 來表示二進位、十六進位、八進位

  ```scala
  "hff".U // 255 in hex
  "o377".U // 255 in octal
  "b1111_1111".U // 255 in binary
  ```

- Boolean 型態

  ```scala
  val a = true.B
  val b = false.B
  ```

### Combinational Circuits

- 可以宣告一個空的 Wire，他的寬度會在之後自動推論

  ```scala
  val w = Wire(UInt())
  w := a & b
  ```

- 提取 Single Bit

  ```scala
  val sign = x(31)
  ```

- 提取 Bit Range

  ```scala
  val sign = x(31, 30)
  ```

- Concatenation

  ```scala
  val y = Cat(x, z)
  val z = x ## y // same as Cat(x, y)
  ```

- Mux
  Chisel 提供 Mux 函數，用來實現多路選擇器，第一個參數是 Bool 型態
  ```scala
  val result = Mux(sel, a, b)
  ```

### Registers

在 Chisel 中，register 默認是連到一個 global clock 上的，且是 rising edge-triggered，還會連到一個同步的 reset signal 上。

- RegInit: 用來初始化 register 的值

  ```scala
  val reg = RegInit(0.U(8.W)) // 8-bit register with initial value 0
  ```

- 更新 register 的值，有多種方法

  ```scala
  reg := d // d is the new value of reg
  val q = reg // q is the value of reg

  val nextReg = RegNext(d) // nextReg is the value of d in the next cycle

  val bothReg = RegNext(d, 0.U) // 0.U is the reset value
  ```

### Structure with Bundle and Vec

- Bundle: 用來封裝多個訊號，繼承 Bundle Class，定義自己的結構

  ```scala
  class MyBundle extends Bundle {
      val a = UInt(32.W)
      val b = Bool()
  }

  val bundle = Wire(new MyBundle)
  bundle.a := 1.U
  bundle.b := true.B
  val a = bundle.a
  val anotherBundle = bundle // anotherBundle is a reference to bundle
  ```

  在接電線的時候會用 `:=`，像是 `a`、`anotherBundle` 都是 reference，所以是用 `=`

- Vec: 用來封裝多個相同型態的訊號

  ```scala
  val vec = Wire(Vec(4, UInt(8.W)))
  vec(0) := 1.U
  vec(1) := 2.U
  vec(2) := 3.U
  vec(3) := 4.U
  val x = vec(2) // 像是 Mux
  ```

  register 型態的 vec init，Seq.fill(4)(0.U(8.W)) 會產生一個 4 個 8-bit 0 的 Vec

  ```scala
  val regVec = RegInit(VecInit(Seq.fill(4)(0.U(8.W))))
  ```

  Vec 不支援範圍 Assignment，可以用 Bundle 自己拆，最後再合併

  ```scala
  val assignWord = Wire(UInt(16.W))
  class Split extends Bundle {
      val high = UInt(8.W)
      val low = UInt(8.W)
  }
  val split = Wire(new Split())
  split.low := lowByte
  split.high := highByte
  assignWord := split.asUInt() // asUInt() is used to convert a Bundle to a UInt
  ```

### 參考資料

[Chisel 学习笔记 - JamesDYX](https://www.cnblogs.com/JamesDYX/p/10072885.html)
[Digital Design with Chisel](https://www.imm.dtu.dk/~masca/chisel-book.pdf)
[偉棻老師的筆記]()
