---
title: 數學筆記
date: 2025-02-18 01:37:12
tags: 
category: 
math: true
---

> 我的數學好爛，忘光光，小補一下

## 泰勒展開式

出發點：希望能夠用多項式來近似任何函數，包含 $sin(x)$、$cos(x)$、$e^x$ 等等，這樣就可以用微積分的方法來解決問題。

$$
f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n + R_n(x)
$$

其中 $R_n(x)$ 是餘項，當 $x \to a$ 時，$R_n(x) \to 0$，所以可以省略。

### 推導

$$
f(x) = a_0 + a_1x + a_2x^2 + a_3x^3 + \cdots + a_nx^n
$$

以 $x = 0$ 為例，當兩邊微分 $n$ 次後，得到通式 $a_n$：

$$
f^{(n)}(0) = n! \cdot a_n \Rightarrow a_n = \frac{f^{(n)}(0)}{n!}
$$

把 $a_n$ 代入 $f(x)$ 得到 **馬克勞林級數 (Maclaurin Series)** ：

$$
f(x) = f(0) + \frac{f'(0)}{1!}x + \frac{f''(0)}{2!}x^2 + \frac{f'''(0)}{3!}x^3 + \cdots + \frac{f^{(n)}(0)}{n!}x^n
$$

現在為了適用於任意點 $a$，以函數平移的角度來看，從 $x = 0$ 平移到 $x = a$，就是 $x - a$，得到 **泰勒展開式 (Taylor Series)** ：
$$
f(x) = f(a) + \frac{f'(a)}{1!}(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n
$$

可以看到，**泰勒展開式** 是在 $x = a$ 的點展開，**馬克勞林級數** 是在 $x = 0$ 的點展開，**泰勒展開式** 是 **馬克勞林級數** 的一般化。

### 例子

#### $e^x$

在 $x = a$ 展開：

$$
e^x = e^a + e^a(x-a) + \frac{e^a}{2!}(x-a)^2 + \frac{e^a}{3!}(x-a)^3 + \cdots + \frac{e^a}{n!}(x-a)^n = \sum_{n=0}^{\infty} \frac{e^a}{n!}(x-a)^n
$$

在 $x = 0$ 展開：

$$
e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots + \frac{x^n}{n!} = \sum_{n=0}^{\infty} \frac{x^n}{n!}
$$


#### $sin(x)$

在 $x = 0$ 展開：

$$
sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n+1}}{(2n+1)!}
$$

#### $cos(x)$

在 $x = 0$ 展開：

$$
cos(x) = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n}}{(2n)!} 
$$

#### 參考資料

- [如何理解泰勒展开？它有何用途？高中生也能听懂的泰勒展开式](https://www.youtube.com/watch?v=ViRvw2Hfto4)