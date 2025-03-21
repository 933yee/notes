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

有三種主要的資料型態：Bits、UInt、SInt

- Bits: 用來表示固定長度的二進位數字
- UInt: 用來表示無號整數
- SInt: 用來表示有號整數

```scala
val x = Bits(8.W) // 8-bit
val y = UInt(8.W) // unsigned 8-bit
val z = SInt(8.W) // signed 8-bit (two's complement)
```

可以看到整數的後面有 .W，是用來表示整數的長度

- 8.W: 8-bit 整數，W 是 width 的縮寫
- 8.U: 值為 8 的 UInt，U 是 unsigned 的縮寫，會自動推論有幾個 bit
- 8.S: 值為 8 的 SInt，S 是 signed 的縮寫，會自動推論有幾個 bit
- 3.U(4.W): 值為 3 的 UInt，原本只需要 2-bit，但是括號裡面指定他有 4-bit (3.U(4.W) = 0011)

### 參考資料

[Chisel 学习笔记 - JamesDYX](https://www.cnblogs.com/JamesDYX/p/10072885.html)
[Digital Design with Chisel](https://www.imm.dtu.dk/~masca/chisel-book.pdf)
[偉棻老師的筆記]()

```

```
