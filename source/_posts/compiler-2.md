---
title: Compiler 筆記 (2)
date: 2024-03-26 15:05:02
tags: Compilier
category: 
math: true
---

## Terminlogy
- Gramma
  - $X \in G $ iff $ G \rightarrow X$
- Language
  - $L(G) = $ { $X | X \in G$ }
- Alphabet
  - $\Sigma$ = {0, 1}
  - $L$ over $\Sigma$

## Context-Free Grammar (CFG)
#### Grammar G = (V, T, P, S)
- **V**: A set of non-terminals (variables)
- **T**: A set of terminals
- **P**: A set of production rules
- **S**: Starting symbol

#### Example 1:
Write a grammar to represent L = { $a^{n}b^{n}$ | $n\ge0$}
- G = (V, T, P, S)
- V = {S}
- T = {a, b}
- P = {S $\rightarrow$ aSb | $\epsilon$}

#### Example 2
Write a grammar representing a balanced expression with '(' and ')'
- G = (V, T, P, S)
- V = {S}
- T = {(, )}
- P = {S $\rightarrow$ (S) | SS | $\epsilon$}

#### Example 3
Write a grammar for palindrome, L = { $W W^{T}$ | $W \in (a, b)^{*}$ }
- G = (V, T, P, S)
- V = {S}
- T = {a, b}
- P = {S $\rightarrow$ aSa | bSb | a | b | $\epsilon$}

#### Example 4
L = { $WcW^T$ | $W \in (a, b)^{*}$}
- G = (V, T, P, S)
- V = {S}
- T = {a, b, c}
- P = {S $\rightarrow$ aSa | bSb | c}

#### Example 5
Write a grammar representing an expression with equal number of a, b
- G = (V, T, P, S)
- V = {S}
- T = {a, b}
- P = {S $\rightarrow$ aSb | bSa | SS | $\epsilon$}


### Ambiguous Grammar
- 一個 sentence 可以由某文法推導出兩個或兩個以上的剖析樹 (parse tree)

#### Example
$$
  \begin{align}
    E &\rightarrow E + E \newline
    &\rightarrow E * E \newline
    &\rightarrow \text{ID} \newline
    &\rightarrow \text{number} \newline
    &\rightarrow (E)
  \end{align} 
$$
  Target: 2 + 3 + 2

### Un-Ambiguous Grammar

#### Example 1
- 乘法在 lower level，因為 priority 比加法高
- 加法和乘法都是 left associative
$$
  \begin{align}
    E &\rightarrow E + \text{term} \newline 
    &\rightarrow \text{term} \newline\newline
    \text{term} &\rightarrow \text{term} * \text{factor} \newline 
    &\rightarrow \text{factor} \newline\newline
    \text{factor} &\rightarrow \text{number} \newline
    &\rightarrow (E)
  \end{align} 
$$

#### Example 2
- exponent 在 lower level，因為 priority 比加法和乘法高
- exponent 是 right associative
$$
  \begin{align}
    E &\rightarrow E + \text{term} \newline 
    &\rightarrow \text{term} \newline\newline
    \text{term} &\rightarrow \text{term} * \text{expo} \newline 
    &\rightarrow \text{expo} \newline\newline
    \text{expo} &\rightarrow \text{factor} ^ \text{expo} \newline 
    &\rightarrow \text{factor} \newline\newline
    \text{factor} &\rightarrow \text{number} \newline
    &\rightarrow (E)
  \end{align} 
$$


## Recursion
- **任何left recursion都可以用數學轉換成right recursion**
- With right recursion, no reduction takes place until the entire list of elements has been read; with left recursion, a reduction takes place as each new list element is encountered. Left recursion can therefore save a lot of stack space.
  - [Right Recursion versus Left Recursion](https://www.ibm.com/docs/en/zvm/7.2?topic=topics-right-recursion-versus-left-recursion)
- With a left-recursive grammar, the top-down parser can expand the frontier indefinitely without generating a leading terminal symbol that the parser can either match or reject. To fix this problem, a compiler writer can convert the left-recursive grammar so that it uses only right-recursion.
  - [Left-Recursion](https://www.sciencedirect.com/topics/computer-science/left-recursion)

### Example 1
#### Left-Recursion
$$
  \begin{align}
    S &\rightarrow S\alpha | \beta
  \end{align} 
$$

#### Right-Recursion
$$
  \begin{align}
    S &\rightarrow \beta S' \newline
    S' &\rightarrow \alpha S' | \epsilon
  \end{align} 
$$


### Example 2
#### Left-Recursion
$$
  \begin{align}
    E &\rightarrow E + \text{term} \newline 
    &\rightarrow \text{term} \newline\newline
    \text{term} &\rightarrow \text{term} * \text{factor} \newline 
    &\rightarrow \text{factor} \newline\newline
    \text{factor} &\rightarrow \text{number} \newline
    &\rightarrow (E)
  \end{align} 
$$

#### Right-Recursion
$$
  \begin{align}
    E &\rightarrow TE' \newline 
    E' &\rightarrow +TE' | \epsilon \newline
    T &\rightarrow FT' \newline
    T' &\rightarrow *FT' | \epsilon \newline
    F &\rightarrow (E) | id
  \end{align} 
$$

### Example 3
#### Left-Recursion
$$
  \begin{align}
    E &\rightarrow E + T \newline 
    &\rightarrow T \newline
    T &\rightarrow T * P \newline 
    &\rightarrow P \newline
    P &\rightarrow F ^ P\newline
    &\rightarrow F \newline
    F &\rightarrow id \newline
    &\rightarrow (E)
  \end{align} 
$$

#### Right-Recursion
$$
  \begin{align}
    E &\rightarrow TE' \newline 
    E' &\rightarrow +TE' | \epsilon \newline
    T &\rightarrow PT' \newline
    T' &\rightarrow *PT' | \epsilon \newline
    P &\rightarrow F ^ P \newline
    &\rightarrow F \newline
    F &\rightarrow (E) | id 
  \end{align} 
$$

[Problem of Left Recursion and Solution in CFGs](https://www.youtube.com/watch?v=IO5ie7GbJGI)


