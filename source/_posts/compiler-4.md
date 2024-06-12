---
title: Compiler 筆記 (4)
date: 2024-05-16 19:43:15
tags: Compilier
category: 
math: true
---

> 參考 清華大學 李政崑老師 編譯器設計講義

## leftmost and rightmost derivations

[leftmost and rightmost derivations](https://www.youtube.com/watch?v=K_aMajzrKF4)
- 介紹 leftmost 和 rightmost derivations
 
- A left-recursive grammar might cause a recursive-decent parser, even one with back-tracking, into an infinite loop.
  -  That is, when we try to expand A, we may eventually find ourselves again trying to expand A without having consumed any input.

## Push Down Automata

[Pushdown Automata (Introduction)](https://www.youtube.com/watch?app=desktop&v=4ejIAmp_Atw)
- PDA = Finite State Machine + A Stack
- PDA = A input tape + A finite control unit + A stack with infinite size
  
[Pushdown Automata (Formal Definition)](https://www.youtube.com/watch?v=JtRyd7Svlew)
- 介紹 $M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$
- 介紹 $\delta$ 的 input 和 output

[Pushdown Automata (Graphical Notation)](https://www.youtube.com/watch?v=eY7fwj5jvC4)
- 介紹 PDA 的 Graph 
- 介紹簡單範例: L = {$0^n 1^n$ | n $\ge$ 0}
- 一個 language 會被 accept 一旦它能達到 final state 或讓 stack 變空

[Pushdown Automata Example (Even Palindrome) PART-1](https://www.youtube.com/watch?v=TEQcJybMMFU)
- 介紹 Palindrome 的範例

[Pushdown Automata Example (Even Palindrome) PART-2](https://www.youtube.com/watch?v=BxA-aI2dyRo)
- 繼續上一部的 Palindrome 範例，詳細介紹 epsilon 是怎麼運作的

## Parsers
[Introduction to Parsers](https://www.youtube.com/watch?v=OIKL6wFjFOo)
- 介紹 Bottom-up Parser vs. Top-Down Parser
- 整個 Parser 的生態結構

### Top Down Parsers
[Top Down Parsers - Recursive Descent Parsers](https://www.youtube.com/watch?v=iddRD8tJi44)
- 介紹 Recursive Descent Parsers

[Top Down Parsers - LL(1) Parsers](https://www.youtube.com/watch?v=v_wvcuJ6mGY)
- 介紹 Recursive Descent Parsers 的名稱由來
- 介紹 LL(1) 的名稱由來
- 簡單介紹 FIRST() 和 FOLLOW()

[FIRST() and FOLLOW() Functions](https://www.youtube.com/watch?v=oOCromcWnfc)
- 非常重要的影片，多看幾次
- 計算 FIRST() 從下往上，計算 FOLLOW() 從上往下
- FISRT() 要包含 epsilon，FOLLOW() 不用
- 計算 FOLLOW() 前最好把 FIRST() 都列好，比較好算
- FOLLOW() 大概可以分成三種 case，就算遇到 epsilon 也一樣方法：
  1. The **following terminal symbol** will be selected as FOLLOW
  2. The **FIRST of the following non-terminal** will be selected as FOLLOW
  3. If it is the right most in the RHS, the **FOLLOW of the LHS** will be selected


[FIRST() and FOLLOW() Functions – Solved Problems (Set 1)](https://www.youtube.com/watch?v=jv4dwxukVvU)
- 更多 FIRST FOLLOW 的範例
- 不確定 Q2 Q3 的 FIRST(S) 要不要有 epsilon
  - 不用，如果全部產生的 non-terminals FIRST 都有 epsilon 才要

[FIRST() and FOLLOW() Functions – Solved Problems (Set 2)](https://www.youtube.com/watch?v=Wo4bafMawFA)
- 更多 FIRST FOLLOW 的範例

[LL(1) Parsing Table](https://www.youtube.com/watch?v=DT-cbznw9aY)

[LL(1) Parsing](https://www.youtube.com/watch?v=clkHOgZUGWU)
