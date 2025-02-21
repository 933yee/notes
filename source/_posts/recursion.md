---
date: 2023-09-13 09:21:41
title: Recursion
categories: [Algorithm, Recursion]
tags: [Recursion, Algorithm]
math: true
---

## Substitution Method

#### 1. $T(n) = 2T(\lfloor n/2 \rfloor) + n$, $T(1) = 1$

- Guess $T(n) = O(n lgn)$
- Show it by **induction**
  - for $n = 2$, $T(2) = 4$
  - for $c = 2$, $T(2) \le c n lgn$
- Base case: $n_0 = 2$hold
- Induction case
  - Assume the guess is true for all $n = 2, 3, ..., k$
  - For $n = k + 1$, we have
    $$
    \begin{align}
      T(n) &= 2T(\lfloor n/2 \rfloor) + n \newline
      &\le 2c\lfloor n/2 \rfloor lg \lfloor n/2 \rfloor + n \newline
      &\le c n lg n/2 + n = c n lgn - c n + n \newline
      &\le c n lg n
    \end{align}
    $$

#### 2. $T(n) = T(\lfloor n/2 \rfloor) + T(\lceil n/2 \rceil) + 1$, $T(1) = 1$

- 可以發現當 $n = 16 $時
  $$
    \begin{align}
      T(16) &= 2T(8) + 1 \newline
      &= 4T(4) + 2 + 1 \newline
      &= 8T(2) + 4 + 2 + 1 \newline
      &= 16T(1) + 8 + 4 + 2 + 1
    \end{align}
  $$
  - 當 n 夠大時， $T(1)$項可以被省略，所以可以猜 $T(n) = O(n)$
- Base case: for $c = 1$, $T(1) = 1 \le cn = 1$
- Inductive case:
  $$
    \begin{align}
      T(n) &= T(\lfloor n/2 \rfloor) + T(\lceil n/2 \rceil) + 1 \newline
      &= cn + 1 \newline
      &\not\le cn
    \end{align}
  $$
- Solution: prove a **stronger** statement
  - $T(n) \le cn - b$
- Base case: for $c = 2, \; b = 1$, $T(2) = 3 \le cn - b = 3$
- Improved Inductive case:
  $$
    \begin{align}
      T(n) &= T(\lfloor n/2 \rfloor) + T(\lceil n/2 \rceil) + 1 \newline
      &= c\lfloor n/2 \rfloor - b + c \lceil n/2 \rceil - b + 1 \newline
      &= cn - b \newline
      &\le cn ,\quad (b \ge 1)
    \end{align}
  $$

#### 3. $T(n) = 2T(\sqrt{n}) + lgn$

- Set $m = lgn$, we get $T(2^m) = 2T(2^{m/2}) + m$
- Rename $S(m) = T(2^m) = T(n)$, $S(m) = 2S(m/2) + m$
- We solve $S(m) = O(mlgm)$, $T(n) = O(lgn \cdot lg(lgn))$

## Recursion Tree Method

#### 1. $T(n) = 2T(n/2) + n^2$, with $T(1) = 1$

- Expanding the terms
  $$
    \begin{align}
      T(n) &= 2T(n/2) + n^2 \newline
      &= n^2 + n^2/2 + 4T(n/4) \newline
      &= n^2 + n^2/2 + n^2/4 + 8T(n/8) \newline
      &= ... \newline
      &= \sum_{k=0}^{lgn - 1} (1/2)^k n^2 + 2^{lgn}T(1) \newline
      &= \Theta(n^2) + \Theta(n) = \Theta(n^2) \newline
    \end{align}
  $$

#### 2. $T(n) = T(n/3) + T({2n}/3) + n$, with $T(1) = 1$

- 深度是 $log\_{3/2}n$，因為右邊項都是原本的 2/3

## Master Method

- When the **recurrence** is in a special form, we can apply the **Master Theorem** to solve the recurrence immediately
- $T(n) = aT(n/b) + f(n)$with $a \ge 1$and $b > 1$, where $n/b$is either $\lfloor n/b \rfloor$or $\lceil n/b \rceil$
- There are three cases

#### 1. Case 1

- $f(n) = O(n^{log_b^{a} - \epsilon})$for some constant $\epsilon > 0$
- 這代表的意義是，recursion 通常最後可以分成兩項
  1.  **最後一層的數量**，也就是 Divide
  2.  **每一層要做的計算**，也就是 Conquer
- 比較兩者，Case 1 代表 Divide 的計算量比 Conquer 大，所以可以忽略 Conquer 的時間複雜度
- 方程式中的 $n^{log_b{a}}$代表最後一層有幾個 node，也可以看成 $a^{log_b{n}}$，代表每一層 **會增加 a** 倍的 node，且總共有 $log_b{n}$層
- Example
  1.  $T(n) = 9T(n/3) + n$, T(1) = 1
      - We have $a = 9, \; b = 3, \; f(n) = n$
      - Since $n^{log_b{a}} = n^{log_3{9}} = n^2$, $f(n) = n = O(n^{2-\epsilon})$, we have $T(n) = \Theta(n^2)$, where $\epsilon = 1$
  2.  $T(n) = 8T(n/2) + n^2$, T(1) = 1
      - We have $a = 8, b = 2 and f(n) = \Theta(n^2)$
      - Since $n^{log_b{a}} = n^{log_2{8}} = n^3$, $f(n) = n^2 = O(n^{3-\epsilon})$, we have $T(n) = \Theta(n^3)$, where $\epsilon = 1$
  3.  $T(n) = 7T(n/2) + n^2$
      - We have $a =7, b = 2$, $n^{log_b{a}} = n^{lg 7} \approx n^{2.81}$
      - Hence, $T(n) = \Theta({n^{2.81}})$

#### 2. Case 2

- Divide 和 Conquer 計算量一樣
- If $f(n) = O(n^{log_b^{a}})$, then $T(n) = \Theta(f(n) lg n)$
- Example
  1.  $T(n) = T(2n/3) + 1$
      - $a = 1, b = 3/2, f(n) = 1$, and $n^{log_b{a}} = n^{log_{3/2}{1}} = 1$
      - We have $f(n) = \Theta(n^{log_b{a}}) = \Theta(1)$
      - Thus $T(n) = \Theta(lg n)$

#### 3. Case 3

- Conquer 計算量比 Divide 大
- If $f(n) = \Omega(n^{log_b{a} + \epsilon})$for some constant $\epsilon > 0$
- And if $a f(n/b \le cf(n))$for some constant $c < 1$
- Then $T(n) = \Theta(f(n))$
- Example: $T(n) = 3T(n/4) + nlgn$
  - $a = 3, b = 4$, $f(n) = n lg n$, and $n^{log_4{3}} = O(n^{0.793})$
  - $f(n) = \Omega(n^{0.793 + \epsilon})$
  - $af(n/b) = 3f(n/4) = 3(n/4)lg(n/4) \le (3/4)n lgn = cf(n) = cf(n)$, for c = 3/4
  - Hence, $T(n) = \Theta(n lg n)$

#### 4. 不能用的情況

1.  $f(n)$is smaller than $n^{log_b{a}}$but **not polynomial smaller**

    - Example: $T(n) = 2T(n/2) + n/lgn$
      - $n^{log_b{a}} = n^{log_2{2}} = n$, **n/lgn** is smaller than **n** but **not polynomial smaller**
      - Hence you can't use Master theorem

2.  $f(n)$is larger than $n^{log_b{a}}$but **not polynomial larger**
    - Example: $T(n) = 2T(n/2) + nlgn$
      - $n^{log_b{a}} = n^{log_2{2}} = n$, **n/lgn** is larger than **n** but **not polynomial larger**
      - Hence you can't use Master theorem
