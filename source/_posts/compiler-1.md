---
title: Compiler 筆記 (1)
date: 2024-03-07 15:51:12
tags: compiler
category: 
math: true
---

> 參考 清華大學 李政崑老師 編譯器設計講義


## compilers and Assemblers
High-level language program (C) 
⇒ C compiler 
⇒ Assembly language program (for MIPS) 
⇒ Assembler 
⇒ Binary machine language program (for MIPS)

## Analysis-Synthesis Model
Compilation 可以分成兩個部分
- Analysis (front end)
  - Breaks up the source program into constituent pieces
  - Creates an Intermediate Representation (IR)
- Synthesis (back end)
  - Constructs the desired target program from the IR
  - (Optionally) performs optimizations

![Analysis-Synthesis Model](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjy5NjmBdRfsrfg96b3KYpevQqRBAygBtuvsszgZpXmPhyy7M9VH81zkvqd-uhdJBBtWL_u0_iaHC8nhSWK1gW7_0DNHxSofoPpj5CD76mp5wCRN7fSz5cDkzZmns_HZ12pRbBD37q9jYY/s1600/aandSmodel.bmp)

## Phase of a Compiler
![Phase of a compiler](https://cdn1.byjus.com/wp-content/uploads/2022/03/phase-of-compiler.png)

### Symbol-Table Management
- Essential function of a compiler
  - To **record the identifier** used in the source program and collect information about various attributes of each identifier
    - allocate storage, type, scope, etc
- Symbol Table
  - A data structure containing a record for each identifiers, with fields for the attributes
  - When a identifier is detected by the **lexical analysis(詞法分析)** , it is entered into the symbol table
  - The attributes are determined during **syntax analysis(語法分析)** and **semanic analysis(語義分析)**

### Analysis Phases
- Lexical Analysis
- Syntax Analysis
- Semantic Analysis

![Analysis Phases Example](./images/compiler-1/AnalysisPhasesExample.png)

### Intermediate Code Generation
- Two properties
  - Easy to produce
  - Easy to translate into the target program
- Examples
  - Graph representations
  - Postfix notation
  - Three-address code
    - 每條指令最多有三個 operands

![Intermediate Code Generation Example](./images/compiler-1/IntermediateCodeGenerationExample.png)

### Code Optimization
- Attempts to improve the intermediate code
  - So the faster-running machine code will result
  
![Code Optimization Example](./images/compiler-1/CodeOptimizationExample.png)

### Code Generation
- Generates target code
  - Consisting of reocatable machine code or assembly code
  
![Code Generation Example](./images/compiler-1/CodeGenerationExample.png)

### Counsins of the compiler
- Preprocessors
  - Produce input to compilers
  - Macro processing
  - File inclusion
- Assemblers
- Loaders and Link-Editors

## Evolution of Programming Languages
### Imperative language
- 命令式語言
- 指定程式該執行的確切操作
- Ex: C, C++, Java, Python
  
### Declarative language
- 宣告式語言
- 只要所需的結果，而不是詳細指定要執行的步驟
- Ex: SQL, HTML, CSS, Prolog

### Von Neumann language
- 基於 Von Neumann 電腦架構
- Many widely used programming languages such as C, C++ and Java have **ceased** to be strictly von Neumann by adding support for parallel processing, in the form of threads.
- Before C++ 11 added threads, C++ was strictly a Von Neumann language

- [Is C++ considered a Von Neumann programming language?](https://stackoverflow.com/questions/58312638/is-c-considered-a-von-neumann-programming-language)
- [高效能需求應用興起記憶體內運算的新戰場](https://ictjournal.itri.org.tw/xcdoc/cont?xsmsid=0M236556470056558161&sid=0M250379986616668141)

### Object-oriented language
- 繼承、封裝、多型
- Ex: Java, C++

### Functional language
- 在一般常見的命令式語言中，要執行操作的話是給電腦一組命令，而狀態會隨著命令的執行而改變。例如你指派變數 a 的值為 5，而隨後做了其它一些事情之後 a 就可能變成的其它值。有控制流程 (control flow)，你就可以重複執行操作
- 然而在純粹函數式程式語言中，你不是像命令式語言那樣命令電腦「要做什麼」，而是通過用函數來描述出問題「是什麼」，如「階乘是指從 1 到某個數的乘積」，「一個串列中數字的和」是指把第一個數字跟剩餘數字的和相加。你用宣告函數是什麼的形式來寫程式
- 另外，變數 (variable) 一旦被指定，就不可以更改了，你已經說了 a 就是 5，就不能再說 a 是別的什麼數
- Ex: Haskell、Scala、Clojure
  
```Haskell
add x y = x + y
result = add 5 10
```

- [Haskell 趣學指南](https://learnyouahaskell.mno2.org/zh-tw/ch01/introduction)

### Assignment-oriented language
- 賦值操作來實現程式邏輯的語言
- Ex: C, Java, 
- 反例: Haskell

### Third-generation language
- 相對於機器語言和組合語言而言的高階程式語言
- Ex: C, Java, Python
  
### Fourth-generation language
- 更高級、更抽象的程式語言，旨在簡化特定領域的應用程式開發
- 提供了更高程度的自動化和巨集
- Ex: SQL, MATLAB

### Scripting language
- 一個指令碼通常是直譯執行而非編譯
- Ex: JavaScript、Perl、PHP、Python、Ruby 和 Tcl，

### 補充
如果是說 imperative programming 和 declarative programming，我查到的都是程式碼的寫法
 
- Imperative language
  - 命令式編程
  - 著重在 **HOW**，具體表達程式碼該做什麼才能達到目標，程式一步一步按著順序照著你給他指示執行。
  - Imperative 比較常運用 Statement ，像是是 if, while, for, switch 等。
  - You tell the compiler what you want to happen, step by step.
  
- Delcarative language
  - 宣告式編程
  - 著重在該做什麼 **WHAT** ，採取抽象化流程。Declarative 比較常運用表達式 expression，
  - Delcarative 特色是單純運算並一定會有返回值

Example: choose the odd numbers

``` cs
List<int> collection = new List<int> { 1, 2, 3, 4, 5 };
```
With **imperative programming**, we'd step through this, and decide what we want:

``` cs
List<int> results = new List<int>();
foreach(var num in collection)
{
    if (num % 2 != 0)
          results.Add(num);
}
```
With **declarative programming**, on the other hand, you declare your desired results, but not the step-by-step:
  
```cs
var results = collection.Where( num => num % 2 != 0);
```

Source
- [Buzz Word 1 : Declarative vs. Imperative](https://ithelp.ithome.com.tw/articles/10233761)
- [What is the difference between declarative and imperative paradigm in programming?](https://stackoverflow.com/questions/1784664/what-is-the-difference-between-declarative-and-imperative-paradigm-in-programmin)

## Lambda Parameters
C++ 中，Lambda 架構是
```c++
[capture clause](parameters) -> return_type { body }
```

**[ ]** 空的捕獲列表，表示不捕獲任何外部變數
  
```cpp
[]() { return 42; }
```

**[&]** 按引用捕獲所有外部變數
  
```cpp
int x = 10;
[&]() { x++; } // 修改外部變數 x
```

**[=]** 按值捕獲所有外部變數

```cpp
int x = 10;
[=]() { return x * 2; } // 訪問但不修改外部變數 x
```

**[=, &foo]** 按值捕獲所有外部變數，但特別引用了 foo 變數

```cpp
int x = 10;
int foo = 20;
[=, &foo]() { return x + foo; } // 訪問 x 按值，訪問 foo 按引用
```

**[bar]** 按值捕獲 bar 變數

```cpp
int bar = 30;
[bar]() { return bar * 3; }
```

**[this]** 按值捕獲當前物件的指標（常用於 lambda 在 class 內部的情況）

```cpp
class MyClass {
public:
    MyClass(int value) : value(value) {}

    void printValue() {
        // Lambda 捕獲當前物件的指標，可以使用內部的變數和函式
        auto lambda = [this]() {
            std::cout << "Value inside lambda: " << value << std::endl; // 訪問物件變數
            someMethod(); // 呼叫物件函式
        };

        lambda(); 
    }

private:
    int value;

    void someMethod() {
        std::cout << "Inside someMethod" << std::endl;
    }
};
```

## First Class Object
- Entity can be stored into a variable
- Something can be passed around as parameters
- In C language, structure and function are consider first class objects

## Bindings
### Static Binding
- 在編譯時期或者早期階段就確定呼叫哪個方法或函式
- Ex: C 的函式呼叫，它在編譯時期就將函式內容綁定到識別符上，而無法在執行時期變更。

### Dynamic Binding
- 綁定發生在 runtime，而不是在編譯時
- Ex: C++ 的虛擬方法呼叫，由於多型的機制，物件的型別無法在編譯時期得知，所以綁定會在執行時期處理。
  
### Fluid Binding
- AKA dynamic assignment
- Assignments with dynamic extent to bindings that have lexical scope
- Syntax
  - var := expr during stmt-body
  
```scheme
(define x 10)  ; 定義變數 x，並賦值為 10

(displayln "在動態賦值之前：")
(displayln x)  ; 輸出當前變數 x 的值

; 在這個程式碼塊內，我們動態地改變變數 x 的值
(let ((x := 20 during
       (begin
         (displayln "在動態賦值內部：")
         (displayln x)  ; 輸出改變後的 x 的值
         (set! x (* x 2))  ; 修改 x 的值
         (displayln "在動態賦值內部修改後：")
         (displayln x)  ; 輸出修改後的 x 的值
       )))
  ; 這裡的 x 仍然是原始值
  (displayln "在動態賦值之後：")
  (displayln x)  ; 輸出原始值
)
```

### Example
```
program parameter-passing;
  var i: integer;
  a: array [1..3] of integer;

  procedure mess;
    begin
      var y: integer;
      y = a[i] + 5;
      writeln('y=', y);
    end;

  procedure sub1;
    var i: integer;
    a: array [1..3] of integer;
    begin
      i := 2;
      a[1] := 5 ; a[2] := 7; a[3] := 9;
    mess;
  end;

  begin 
    i := 1;
    a[1] :=1 ; a[2] := 4; a[3] :=8 ;
    sub1;
  end
```
- Suppose static binding (also known as lexical binding) is used for variable scopes. What's the printout value of y?
> 我猜 mess 裡面 i 是 1、a[i] 是 1，所以 y = 1 + 5 = 6

- Suppose dynamic binding is used for variable scopes. What's the printout value of y?
> 我猜 mess 裡面 i 是 2、a[i] 是 7，所以 y = 7 + 5 = 12

## Parameter Passing Schemes
### Call-by-reference
- Call 的瞬間就看 caller
- 在呼叫 function 的當下就已經決定好 parameter 的值
 
### Call-by-name
- 用的時候重新看 caller
- 在呼叫的 function 當中每次使用到 parameter 就重新去檢查 caller 當中當下的值
 
### Call-by-need
- 跟 call by name 很像，一樣去算 caller 的，但是第一次算完就存起來，不用每次都算
- 第一次使用時重新看 caller
- 在呼叫的 function 當中第一次使用到 parameter 的時候才去決定後續的值
 
### Call-by-text
- 用的時候重新看 callee
- 在呼叫的 function 當中每次使用到 parameter 就重新去檢查 callee 當中當下的值
- 更好的理解方式是從名稱 "call by text"，意即 parameter 會以 text 的型態傳遞，因此需要把所有的 parameter 都視為傳遞前的原貌
- 例如定義 `f(v: integer)` ，呼叫 `f(a[i])` ，則 `f` 當中的每個 `v` 都需要替換成 `a[i]`

#### Example
```
program parameter-passing;
  var i: integer;
  a: array [1..3] of integer;

  procedure mess(v : integer);
    begin
      v := v + 1;
      a[i] := 5;
      i := 3;
      v := v + 1;
    end;

  begin
    for i:= 1 to 3 do a[i] := 0;
    a[2] := 10;
    i := 2;
    mess(a[i]);
  end
```
- If by assuming **Call-by-Text**, what's the value in the array a and the variable i?
  
|                   | a[1]  | a[2]  | a[3]  |   i   |      v      |
| :---------------: | :---: | :---: | :---: | :---: | :---------: |
|    before mess    |   0   |  10   |   0   |   2   |      -      |
|    v := v + 1;    |   0   |  11   |   0   |   2   | a[i] (a[2]) |
|    a[i] := 5;     |   0   |   5   |   0   |   2   | a[i] (a[2]) |
|      i := 3       |   0   |   5   |   0   |   3   | a[i] (a[3]) |
|    v := v + 1     |   0   |   5   |   1   |   3   | a[i] (a[3]) |
| observation point |   0   |   5   |   1   |   3   |      -      |

- If by assuming **Call-by-Reference**, what's the value in the array a and the variable i?
  
|                   | a[1]  | a[2]  | a[3]  |   i   |   v   |
| :---------------: | :---: | :---: | :---: | :---: | :---: |
|    before mess    |   0   |  10   |   0   |   2   |   -   |
|    v := v + 1;    |   0   |  11   |   0   |   2   | a[2]  |
|    a[i] := 5;     |   0   |   5   |   0   |   2   | a[2]  |
|      i := 3       |   0   |   5   |   0   |   3   | a[2]  |
|    v := v + 1     |   0   |   6   |   0   |   3   | a[2]  |
| observation point |   0   |   6   |   0   |   3   |   -   |

```
program parameter-passing;
  var i: integer;
  a: array [1..3] of integer;

  procedure mess(v : integer);
    var i: integer;
    begin
      i := 1;
      v := v + 1;
      a[i] := 5;
      i := 3;
      v := v + 1;
    end;

  begin
    for i:= 1 to 3 do a[i] := 0;
    a[2] := 10;
    i := 2;
    mess(a[i]);
  end
```

- If by assuming **Call-by-Name**, what's the value in the array a and the variable i?
  
|                   | a[1]  | a[2]  | a[3]  | i (caller) | i (callee) |   v   |
| :---------------: | :---: | :---: | :---: | :--------: | :--------: | :---: |
|    before mess    |   0   |  10   |   0   |     2      |     -      |   -   |
|      i := 1       |   0   |  10   |   0   |     2      |     1      | a[2]  |
|    v := v + 1;    |   0   |  11   |   0   |     2      |     1      | a[2]  |
|    a[i] := 5;     |   5   |  11   |   0   |     2      |     1      | a[2]  |
|      i := 3       |   5   |  11   |   0   |     2      |     3      | a[2]  |
|    v := v + 1     |   5   |  12   |   0   |     2      |     3      | a[2]  |
| observation point |   5   |  12   |   0   |     2      |     -      |   -   |

- If by assuming **Call-by-Text**, what's the value in the array a and the variable i?
  
|                   | a[1]  | a[2]  | a[3]  | i (caller) | i (callee) |          v          |
| :---------------: | :---: | :---: | :---: | :--------: | :--------: | :-----------------: |
|    before mess    |   0   |  10   |   0   |     2      |     -      |          -          |
|      i := 1       |   0   |  10   |   0   |     2      |     1      | a[i(callee)] (a[1]) |
|    v := v + 1;    |   1   |  10   |   0   |     2      |     1      | a[i(callee)] (a[1]) |
|    a[i] := 5;     |   5   |  10   |   0   |     2      |     1      | a[i(callee)] (a[1]) |
|      i := 3       |   5   |  10   |   0   |     2      |     3      | a[i(callee)] (a[3]) |
|    v := v + 1     |   5   |  10   |   1   |     2      |     3      | a[i(callee)] (a[3]) |
| observation point |   5   |  10   |   1   |     2      |     -      |          -          |

- If by assuming **Call-by-Need**, what's the value in the array a and the variable i?
  
|                   | a[1]  | a[2]  | a[3]  | i (caller) | i (callee) |          v          |
| :---------------: | :---: | :---: | :---: | :--------: | :--------: | :-----------------: |
|    before mess    |   0   |  10   |   0   |     2      |     -      |          -          |
|      i := 1       |   0   |  10   |   0   |     2      |     1      |          -          |
|    v := v + 1;    |   0   |  11   |   0   |     2      |     1      | a[i(caller)] (a[2]) |
|    a[i] := 5;     |   5   |  11   |   0   |     2      |     1      |        a[2]         |
|      i := 3       |   5   |  11   |   0   |     2      |     3      |        a[2]         |
|    v := v + 1     |   5   |  12   |   0   |     2      |     3      |        a[2]         |
| observation point |   5   |  12   |   0   |     2      |     -      |          -          |

```
program parameter-passing;
  var i: integer;
  a: array [1..3] of integer;

  procedure mess(v : integer);
    begin
      i := 1;
      v := v + 1;
      a[i] := 5;
      i := 3;
      v := v + 1;
    end;

  begin
    for i:= 1 to 3 do a[i] := 0;
    a[2] := 10;
    i := 2;
    mess(a[i]);
  end
```

- If by assuming **Call-by-Name**, what's the value in the array a and the variable i?
  
|                   | a[1]  | a[2]  | a[3]  |   i   |      v      |
| :---------------: | :---: | :---: | :---: | :---: | :---------: |
|    before mess    |   0   |  10   |   0   |   2   |      -      |
|      i := 1       |   0   |  10   |   0   |   1   |      -      |
|    v := v + 1;    |   1   |  10   |   0   |   1   | a[i] (a[1]) |
|    a[i] := 5;     |   5   |  10   |   0   |   1   | a[i] (a[1]) |
|      i := 3       |   5   |  10   |   0   |   3   | a[i] (a[3]) |
|    v := v + 1     |   5   |  10   |   1   |   3   | a[i] (a[3]) |
| observation point |   5   |  10   |   1   |   3   |      -      |

- If by assuming **Call-by-Need**, what's the value in the array a and the variable i?
  
|                   | a[1]  | a[2]  | a[3]  |   i   |      v      |
| :---------------: | :---: | :---: | :---: | :---: | :---------: |
|    before mess    |   0   |  10   |   0   |   2   |      -      |
|      i := 1       |   0   |  10   |   0   |   1   |      -      |
|    v := v + 1;    |   1   |  10   |   0   |   1   | a[i] (a[1]) |
|    a[i] := 5;     |   5   |  10   |   0   |   1   |    a[1]     |
|      i := 3       |   5   |  10   |   0   |   3   |    a[1]     |
|    v := v + 1     |   6   |  10   |   0   |   3   |    a[1]     |
| observation point |   6   |  10   |   0   |   3   |      -      |


