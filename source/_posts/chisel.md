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

## Examples

### Full Adder

```scala
package acal_lab04.Lab

import chisel3._

class HalfAdder extends Module{
  val io = IO(new Bundle{
    val A = Input(UInt(1.W))
    val B = Input(UInt(1.W))
    val Sum = Output(UInt(1.W))
    val Carry = Output(UInt(1.W))
  })
  //the behavior of circuit
  io.Sum := io.A ^ io.B
  io.Carry := io.A & io.B
}

class FullAdder extends Module{
  val io = IO(new Bundle{
    val A = Input(UInt(1.W))
    val B = Input(UInt(1.W))
    val Cin = Input(UInt(1.W))
    val Sum = Output(UInt(1.W))
    val Cout = Output(UInt(1.W))
  })

  //Module Declaration
  val ha1 = Module(new HalfAdder())
  val ha2 = Module(new HalfAdder())

  //Wiring
  ha1.io.A := io.A
  ha1.io.B := io.B

  ha2.io.A := ha1.io.Sum
  ha2.io.B := io.Cin

  io.Sum := ha2.io.Sum
  io.Cout := ha1.io.Carry | ha2.io.Carry
}
```

#### Testbench

```scala
package acal_lab04.Lab

import chisel3.iotesters.{PeekPokeTester,Driver}

class FullAdderTest (fa : FullAdder) extends PeekPokeTester(fa){
  for(a <- 0 until 2){
    for(b <- 0 until 2){
	  for(c <- 0 until 2){
	    poke(fa.io.A,a)
		poke(fa.io.B,b)
		poke(fa.io.Cin,c)

		var x = c & (a^b)
		var y = a & b

		expect(fa.io.Sum,(a^b^c))
		expect(fa.io.Cout,(x|y))
		step(1)
	  }
	}
  }
  println("FullAdder test completed!!!")
}

object FullAdderTest extends App{
	Driver.execute(Array("-td","./generated","-tbn","verilator"),() => new FullAdder()){
		c => new FullAdderTest(c)
	}
}
```

#### Execute

```shell
sbt 'Test/runMain acal_lab04.Lab.FullAdderTest'
```

#### waveform

![FullAdder](./images/chisel/FullAdder.png)

### 32-bits Ripple Carry Adder

```scala
package acal_lab04.Lab

import chisel3._

class RCAdder (n:Int) extends Module{
  val io = IO(new Bundle{
      val Cin = Input(UInt(1.W))
      val In1 = Input(UInt(n.W))
      val In2 = Input(UInt(n.W))
      val Sum = Output(UInt(n.W))
      val Cout = Output(UInt(1.W))
  })

  //FullAdder ports: A B Cin Sum Cout
  val FA_Array = Array.fill(n)(Module(new FullAdder()).io)
  val carry = Wire(Vec(n+1, UInt(1.W)))
  val sum   = Wire(Vec(n, Bool()))

  carry(0) := io.Cin

  for (i <- 0 until n) {
    FA_Array(i).A := io.In1(i)
    FA_Array(i).B := io.In2(i)
    FA_Array(i).Cin := carry(i)
    carry(i+1) := FA_Array(i).Cout
    sum(i) := FA_Array(i).Sum
  }

  io.Sum := sum.asUInt
  io.Cout := carry(n)
}
```

不能直接在 for loop 裡面寫 `io.Sum(i) = sum(i)`，因為 Chisel3 不支援 **subword assignment**

#### Testbench

```scala
package acal_lab04.Lab

import chisel3.iotesters.{Driver,PeekPokeTester}

class RCAdderTest (dut:RCAdder) extends PeekPokeTester(dut){

    val in1 = Array(5,32,1,77,34,55,12)
    val in2 = Array(3456,89489,78,5216,4744,8,321)

    //in1.zip(in2)
    (in1 zip in2).foreach{
      case(i,j)=>
          poke(dut.io.In1,i)
          poke(dut.io.In2,j)
          expect(dut.io.Sum,i+j)
          step(1)
    }


    // for(i <- in1){
    //     for(j <- in2){
    //         poke(dut.io.In1,i)
    //         poke(dut.io.In2,j)
    //         expect(dut.io.Sum,i+j)
    //     }
    // }

    println("RCAdder test completed!!!!!")
}

object RCAdderTest extends App{
    Driver.execute(args,()=>new RCAdder(32)){
        c => new RCAdderTest(c)
    }
}
```

#### Execute

```shell
sbt 'Test/runMain acal_lab04.Lab.RCAdderTest -tbn verilator -td ./generated'
```

#### waveform

![RCAdder](./images/chisel/RCAdder.png)

### Carry Lookahead Adder

```scala
package acal_lab04.Lab

import chisel3._

class CLAdder extends Module{
  val io = IO(new Bundle{
      val in1 = Input(UInt(4.W))
      val in2 = Input(UInt(4.W))
      val Cin = Input(UInt(1.W))
      val Sum = Output(UInt(4.W))
      val Cout = Output(UInt(1.W))
  })

  val P = Wire(Vec(4,UInt()))
  val G = Wire(Vec(4,UInt()))
  val C = Wire(Vec(4,UInt()))
  val S = Wire(Vec(4,UInt()))

  for(i <- 0 until 4){
      G(i) := io.in1(i) & io.in2(i)
      P(i) := io.in1(i) | io.in2(i)
  }

  C(0) := io.Cin
  C(1) := G(0)|(P(0)&C(0))
  C(2) := G(1)|(P(1)&G(0))|(P(1)&P(0)&C(0))
  C(3) := G(2)|(P(2)&G(1))|(P(2)&P(1)&G(0))|(P(2)&P(1)&P(0)&C(0))

  val FA_Array = Array.fill(4)(Module(new FullAdder).io)

  for(i <- 0 until 4){
      FA_Array(i).A := io.in1(i)
      FA_Array(i).B := io.in2(i)
      FA_Array(i).Cin := C(i)
      S(i) := FA_Array(i).Sum
  }

  io.Sum := S.asUInt
  io.Cout := FA_Array(3).Cout
}
```

#### Testbench

```scala
package acal_lab04.Lab

import chisel3.iotesters.{Driver,PeekPokeTester}

class CLAdderTest (dut:CLAdder) extends PeekPokeTester(dut){
    for(i <- 0 to  15){
        for(j <- 0 to 15){
            poke(dut.io.in1,i)
            poke(dut.io.in2,j)
            if(peek(dut.io.Cout)*16+peek(dut.io.Sum)!=(i+j)){
                println("Oh No!!")
            }
            step(1)
        }
    }
    println("CLAdder test completed!!!")
}

object CLAdderTest extends App{
    Driver.execute(args,()=>new CLAdder){
        c => new CLAdderTest(c)
    }
}
```

#### Execute

```shell
sbt 'Test/runMain acal_lab04.Lab.CLAdderTest -tbn verilator -td ./generated'
```

### Stack

#### Testbench

```scala
// See LICENSE from https://github.com/ucb-bar/chisel-tutorial/blob/release/src/test/scala/examples/StackTests.scala
package acal_lab04.Lab

import chisel3.iotesters.{PeekPokeTester, Driver, ChiselFlatSpec}
import scala.collection.mutable.{ArrayStack => ScalaStack}

class StackTestsOrig(c: Stack) extends PeekPokeTester(c) {
  var nxtDataOut = 0
  var dataOut = 0
  val stack = new ScalaStack[Int]()

  for (t <- 0 until 16) {
    println(s"Tick $t")
    val enable  = rnd.nextInt(2)
    val push    = rnd.nextInt(2)
    val pop     = rnd.nextInt(2)
    val top     = rnd.nextInt(2)
    val dataIn  = rnd.nextInt(256)
    val empty   = stack.isEmpty
    val full    = stack.length == c.depth
    println(s"enable $enable push $push pop $pop dataIn $dataIn top $top isempty $empty isfull $full")
    if (enable == 1) {
      dataOut = nxtDataOut
      if (push == 1 && stack.length < c.depth) {
        stack.push(dataIn)
      } else if (pop == 1 && stack.length > 0) {
        stack.pop()
      }
      if (stack.length > 0) {
        nxtDataOut = stack.top
      }
    }

    poke(c.io.pop,    pop)
    poke(c.io.push,   push)
    poke(c.io.en,     enable)
    poke(c.io.dataIn, dataIn)
    poke(c.io.peek,   top)
    step(1)
    expect(c.io.dataOut, dataOut)
    expect(c.io.empty, stack.isEmpty)
    expect(c.io.full, stack.length == c.depth)
  }
}

object StackTest extends App {
  Driver.execute(Array("-td","./generated","-tbn","verilator"),() => new Stack(8)) {
    c => new StackTestsOrig(c)
  }
}
```

#### Execute

```shell
sbt 'Test/runMain acal_lab04.Lab.StackTest'
```

### 32-bit Mux2

```scala
class Mux2_32b extends Module{
//Port Declaration
val io = IO(new Bundle{
    val sel = Input(Bool())
    val in1 = Input(UInt(32.W))
    val in2 = Input(UInt(32.W))

    val out = Output(UInt(32.W))
})

//Method One
when(~io.sel){
    io.out := io.in1
}.otherwise{
    io.out := io.in2
}

//Method Two
def myMux[T <: Data](sel:Bool,in1:T,in2:T):T={
    val x = WireDefault(in1)
    when(sel){x := in2}
    x
}
io.out := myMux(io.sel,io.in1,io.in2)

//Method Three
io.out := Mux(io.sel,io.in1,io.in2)
}
```

### 32xINT32 Register File (2R1W)

```scala
class RegisterFile (readPorts: Int) extends Module{
  val io = IO(new Bundle{
  val wen = Input(Bool())
  val waddr = Input(UInt(5.W))
  val wdata = Input(UInt(32.W))
  val raddr = Input(Vec(readPorts, UInt(5.W)))
  val rdata = Output(Vec(readPorts, UInt(32.W)))
  })
  val regs = RegInit(VecInit(Seq.fill(32)(0.U(32.W))))

  when(io.wen){regs(io.waddr) := io.wdata}
  regs(0) := 0.U

  for(i<-0 until readPorts){
      io.rdata(i) := regs(io.raddr(i))
  }
}
```

### On-chip SRAM

```scala
import chisel3.util.experimental.loadMemoryFromFile

class SyncMem extends Module{
  val io = IO(new Bundle{
    val raddr = Input(UInt(5.W))
    val rdata = Output(UInt(32.W))

    val wen   = Input(Bool())
    val waddr = Input(UInt(5.W))
    val wdata = Input(UInt(32.W))
  })

  //這個是async的宣告方式：Mem(depth,dtype(width))
  //val memory = Mem(32, UInt(32.W))

  //這個是sync的宣告方式
  val memory = SyncReadMem(32, UInt(32.W))

  //Load Memory From File
  loadMemoryFromFile(memory, "./src/main/resource/value.txt")

  //Wiring
  io.rdata := memory(io.raddr)
  when(io.wen) { memory(io.waddr) := io.wdata }
}
```

### 閹割版 INT32 2-Input ALU

```scala
class ALUIO extends Bundle{
  val src1    = Input(UInt(32.W))
  val src2    = Input(UInt(32.W))
  val funct3   = Input(UInt(3.W))
  val funct7   = Input(UInt(7.W))
  val ALUout  = Output(UInt(32.W))
}

class ALU extends Module{
  val io = IO(new ALUIO)

  io.ALUout := MuxLookup(io.funct3,0.U,Seq(
    ADD_SUB -> (Mux(io.funct7===SUB_SRA, io.src1-io.src2, io.src1+io.src2)),
    SLL     -> (io.src1 << io.src2(4,0)) ,
    SLT     ->(Mux(io.src1.asSInt<io.src2.asSInt,1.U,0.U)),
    SLTU    ->(Mux(io.src1<io.src2,1.U,0.U)),
    XOR     -> (io.src1 ^ io.src2)   ,
    SRL_SRA -> (Mux(io.funct7===SUB_SRA,(io.src1.asSInt >> io.src2(4,0)).asUInt,io.src1 >> io.src2(4,0)))
  ))
}
```

### 參考資料

[Chisel 学习笔记 - JamesDYX](https://www.cnblogs.com/JamesDYX/p/10072885.html)
[Digital Design with Chisel](https://www.imm.dtu.dk/~masca/chisel-book.pdf)
[偉棻老師的筆記]()

```

```
