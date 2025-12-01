---
title: Deep Learning
date: 2025-11-06 12:09:09
tags: [deep learning, ai, machine learning]
category: AI
math: true
---

## Linear Algebra

### Span & Linear Dependence

在 Machine Learning 裡面，不管原本的 function 是不是 Linear，我們都會試著用 Linear function 來 Approximate 它。

- $\text{span}(A_{:,1}, A_{:,2}, \ldots, A_{:,n})$ 被稱為 column space of A，記作 $\text{Col}(A)$
- $\text{rank}(A)$ 是 $\text{Col}(A)$ 的維度 (dimension)
- 給定 $A$ 和 $y$，解方程式 $Ax = y$，其中 $A$ 是 $m \times n$ 矩陣，$x$ 是 $n \times 1$ 向量，$y$ 是 $m \times 1$ 向量

  - **一定至少有一個解**

    因為 $Ax = \Sigma_{i} x_i A_{:,i}$，所以 $Ax$ 只不過是 $A$ 的 column vectors 的 linear combination，只要 $y$ 能寫成 $A$ 的 column vectors 的 linear combination，就一定有解

    因此，$\text{span}(A_{:,1}, A_{:,2}, \ldots, A_{:,n}) \ni \mathbb{R}^m$，也代表 $n \geq m$

  - **唯一解**

    $A$ 最多只能有 $m$ 個 Column，也代表 $n = m$，且 $A$ 的 column vectors 必須是線性獨立 linearly independent 的

    在這種情況下，$A$ 是 full rank 的，且為 invertible，因此 $x = A^{-1}y$ 是唯一解

### Norms

norm 是一個 function ($\|\cdot\|$)，能夠把 vector 映射到非負實數 (non-negative real number)

#### $L^p$ norm:

$$
\|x\|_p = \left( \sum_i |x_i|^p \right)^{1/p}
$$

- $p=1$: Manhattan norm
- $p=2$: Euclidean norm (通常直接寫成 $\|x\|$)

  $$
   \|x\| = (x^Tx)^{1/2} = \sqrt{\sum_i x_i^2}
  $$

- $p \to \infty$: Maximum norm

  $$
   \|x\|_{\infty} = \max_i |x_i|
  $$

- $x^Ty = \|x\| \|y\| \cos \theta$，其中 $\theta$ 是 $x$ 和 $y$ 之間的夾角
  - 當 $x \perp y$ 時，$x^Ty = 0$，稱為 orthogonal。且如果 $\|x\| = \|y\| = 1$ (unit vectors)，則 $x$ 和 $y$ 是 orthonormal 的

#### Matrix Norms

- Frobenius norm:

  $$
  \|A\|_F = \sqrt{\sum_{i,j} A_{i,j}^2} = \sqrt{\operatorname{tr}(A^\top A)}
  $$

- Orthogonal matrix:

  每個 column vector 都是 unit vector，且彼此 orthogonal，通常不會特別去區分 orthogonal matrix 和 orthonormal matrix，都稱為 orthogonal matrix

  $$
   Q^TQ = QQ^T = I
  $$

  - 也代表 $Q^{-1} = Q^T$

### Eigendecomposition

Decomposition 可以幫助我們快速了解矩陣的性質

#### Eigen vectors & Eigen values

- 給定一個方陣 $A$，如果存在一個非零向量 $v$ 和一個純量 $\lambda$，使得

  $$
   Av = \lambda v
  $$

  則稱 $v$ 為 $A$ 的 **eigen vector**，$\lambda$ 為對應的 **eigen value**

  可以想像成，某個 function A 遇到 eigen vector 時，只會對其進行伸縮 (scaling)，不會改變方向

- 如果 $v$ 是一個 eigen vector，則 $cv$ (c 為非零常數) 也是 eigen vector，對應的 eigen value 不變，所以一般只會討論 unit eigen vectors

#### Eigendecomposition

- 任何 **real symmetric matrix** $A \in \mathbb{R}^{n \times n}$ 都可以被分解成

  $$
  A = Q \text{diag}(\lambda) Q^T
  $$

  - $\lambda \in \mathbb{R}^n$: eigen values of $A$，通常會由大到小排序
  - $Q = [v_1, v_2, \ldots, v_n]$: 由 $A$ 的 eigen vectors 組成的 orthogonal matrix

- Eigendecomposition 不是唯一的

![eigendecomposition](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT9qejWx2-LxS3zRwd4QcDk7cob7ZWCQJhRoxOjFTb5kmmPzmFDrMYrEEkpc5OwNYNJhPY&usqp=CAU)

可以想成把矩陣 A 當作一個線性轉換 (linear transformation)，先把空間旋轉到 eigen vector 的方向 (由 Q 決定)，再沿著各個 eigen vector 方向進行伸縮 (由 $\lambda$ 決定)，最後再把空間旋轉回來 (由 $Q^T$ 決定)

#### Rayleigh's Quotient

給定一個 real symmetric matrix $A \in \mathbb{R}^{n \times n}$，對任意非零向量 $x \in \mathbb{R}^n$，定義 Rayleigh's Quotient 為

$$
R(x) = \frac{x^T A x}{x^T x}
$$

且

$$
\lambda_{\min} \leq R(x) \leq \lambda_{\max}
$$

- $\lambda_{\min}$: $A$ 的最小 eigen value
- $\lambda_{\max}$: $A$ 的最大 eigen value
- 當且僅當 $x$ 是對應於 $\lambda_{\min}$ 或 $\lambda_{\max}$ 的 eigen vector 時，等號成立

#### Singularity

若 $A = Q \text{diag}(\lambda) Q^T$，則 $A^{-1} = Q \text{diag}(\lambda)^{-1} Q^T$

因為 Diagonal matrix 的反矩陣是把對角線元素取倒數，所以當 $\lambda_i = 0$ 時，$A$ 就沒有反矩陣，稱為 singular matrix

#### Positive Definiteness

- Positive Definite:

  對所有非零向量 $x$，都有 $x^T A x > 0$，等價於所有的 eigen values 都是正的

- Positive Semi-Definite:

  對所有非零向量 $x$，都有 $x^T A x \geq 0$，等價於所有的 eigen values 都是非負的

對於一個 Quadratic Function $f(x) = \frac{1}{2} x^T A x - b^T x + c$，可以藉由檢查 $A$ 是否為 `Positive Definite`、`Positive Semi-Definite`、`Negative Definite`、`Negative Semi-Definite` 來判斷其凸性 (convexity)：

| Definiteness           | Convexity        | Condition on Eigenvalues   |
| ---------------------- | ---------------- | -------------------------- |
| Positive Definite      | Strictly Convex  | All eigenvalues > 0        |
| Positive Semi-Definite | Convex           | All eigenvalues ≥ 0        |
| Negative Definite      | Strictly Concave | All eigenvalues < 0        |
| Negative Semi-Definite | Concave          | All eigenvalues ≤ 0        |
| Indefinite             | Neither          | Eigenvalues of mixed signs |

### Singular Value Decomposition (SVD)

Eigendecomposition 要求矩陣是方陣 (square matrix) 且為 symmetric matrix，但 SVD 不需要這些限制

任何一個矩陣 $A \in \mathbb{R}^{m \times n}$ 都可以被分解成

$$
A = U \Sigma V^T
$$

- $U \in \mathbb{R}^{m \times m}$: 左奇異向量 (left singular vectors)，為 $AA^T$ 的 eigen vectors 組成的 orthogonal matrix
- $\Sigma \in \mathbb{R}^{m \times n}$: 對角矩陣 (diagonal matrix)，對角線上的元素為 $AA^T$ 或 $A^TA$ 的 eigen values 的平方根，稱為 singular values，且通常會由大到小排序
- $V \in \mathbb{R}^{n \times n}$: 右奇異向量 (right singular vectors)，為 $A^TA$ 的 eigen vectors 組成的 orthogonal matrix

### Moore-Penrose Pseudoinverse

對於一般情況，一個非方陣 $A \in \mathbb{R}^{m \times n}$，我們無法計算其反矩陣 $A^{-1}$，但我們可以用另一個矩陣 $B \in \mathbb{R}^{n \times m}$ ，用來解出 $Ax = y$，即 $x = By$

Moore-Penrose Pseudoinverse 令 $B = A^+$，可以把任務拆成三種 Case：

1. **$m = n$**:

   - 有唯一解 (如果 $A$ 是 invertible 的)
   - 解為 $A^+ = A^{-1}$

2. **$m < n$**:

   - 有無限多解
   - 目標是找到一個 $x$，使得 $\|x\|_2$ 最小，找一個最小範數解

3. **$m > n$**:

   - 沒有精確解
   - 目標是找到一個 $x$，使得 $\|Ax - y\|_2$ 最小，找一個最佳近似解

Moore-Penrose Pseudoinverse 定義為

$$
A^+ = \text{lim}_{\lambda \to 0^+} (A^TA + \lambda I)^{-1} A^T
$$

其中 $\lambda$ 是一個非常小的正數，用來確保 $A^TA + \lambda I$ 是 Full Rank 的，因此可以被反轉

實際上計算時，可以直接用 SVD 來計算，先把 $A$ 分解成 $A = U \Sigma V^T$，則

$$
A^+ = V \Sigma^+ U^T
$$

其中 $\Sigma^+$ 是把 $\Sigma$ 的非零奇異值取倒數後，再轉置得到的矩陣

### Traces

- trace 的定義是矩陣對角線元素的和

  $$
   \text{trace}(A) = \sum_{i} A_{i,i}
  $$

- $\|A\|^2_F = \text{trace}(A^TA)$
- $\text{trace}(ABC) = \text{trace}(BCA) = \text{trace}(CAB)$ (cyclic property)

### Determinant

$$
\text{det}(A) = \sum_i (-1)^{i+1} A_{1,i} \text{det}(A_{-1,-i})
$$

其中 $A_{-1,-i}$: 去掉第 1 列和第 i 行後的子矩陣 (submatrix)

- $\text{det}(A^T) = \text{det}(A)$
- $\text{det}(AB) = \text{det}(A) \text{det}(B)$
- $\text{det}(A^{-1}) = \frac{1}{\text{det}(A)}$
- $\text{det}(A) = \prod_i \lambda_i$，其中 $\lambda_i$ 是 $A$ 的 eigen values

Determinant 可以被解釋為線性轉換後，空間體積的伸縮比例，也就是 Image of Unit Square (or Cube) 的體積。

- $\text{det}(A) = 0$，代表它在某一個維度被壓縮成 0，整個空間就會被壓縮成一個低維度的子空間 (subspace)

- $\text{det}(A) = 1$，代表形狀可能會被改變，但體積不變

## Probability & Information Theory

### Random Variables

- Discrete Random Variable: 只能取有限或可數無限個值的隨機變數
- Continuous Random Variable: 可以取無限多個值的隨機變數，通常是區間上的所有值

- 必須有一個對應的 Probability Measure 來描述 Random Variable 的分佈情況
  - Probability Mass Function (PMF)
  - Probability Density Function (PDF)

### Probability Mass and Density Functions

- PMF: 定義在離散隨機變數上，描述每個可能取值的機率

  $$
   P(X = x_i) = p_i
  $$

  - $\sum_i p_i = 1$

- PDF: 定義在連續隨機變數上，描述在某個區間內取值的機率密度

  $$
   P(a < X < b) = \int_{a}^{b} p(x) dx
  $$

  - $\int_{-\infty}^{\infty} p(x) dx = 1$
  - $p(x) \geq 0$，但 $p(x)$ 本身不代表機率
  - $p(x) > 1$ 是可能的，只要積分結果不超過 1 即可
    - 例如，$p(x) = 2$ 在區間 $[0, 0.5]$ 上是合法的 PDF，因為 $\int_{0}^{0.5} 2 dx = 1$

### Marginal Probability

$$
P(X = x) = \sum_y P(X = x, Y = y)
$$

or

$$
P(X = x) = \int p(X = x, Y = y) dy
$$

### Conditional Probability

$$
P(X = x | Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)}
$$

### Indenpendence & Conditional Independence

- Indenpendence:

  $X$ 和 $Y$ 是獨立的，當且僅當

  $$
   P(X = x, Y = y) = P(X = x) P(Y = y)
  $$

  $$
   P(X = x | Y = y) = P(X = x)
  $$

- Conditional Independence:

  $X$ 和 $Y$ 在給定 $Z$ 的條件下是獨立的，當且僅當

  $$
   P(X = x, Y = y | Z = z) = P(X = x | Z = z) P(Y = y | Z = z)
  $$

  $$
   P(X = x | Y = y, Z = z) = P(X = x | Z = z)
  $$

### Expectation

- Discrete Random Variable:

  $$
   \mathbb{E}[X] = \sum_i x_i P(X = x_i)
  $$

- Continuous Random Variable:

  $$
   \mathbb{E}[X] = \int_{-\infty}^{\infty} x p(x) dx
  $$

- 線性性質:

$$
\mathbb{E}[aX + bY] = a \mathbb{E}[X] + b \mathbb{E}[Y]
$$

$$
\mathbb{E}[\mathbb{E}[f(x)]] = \mathbb{E}[f(x)]
$$

$$
\mathbb{E}[f(x) g(y)] = \mathbb{E}[f(x)] \mathbb{E}[g(y)] \quad \text{if } x \perp y
$$

### Variance

- 定義: 隨機變數 $X$ 的方差是其期望值的平方差的期望值

  $$
   \text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
  $$

- 性質:

$$
\text{Var}(aX + b) = a^2 \text{Var}(X)
$$

### Covariance

- 定義: 隨機變數 $X$ 和 $Y$ 的協方差是它們偏離各自期望值的乘積的期望值

  $$
   \text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X] \mathbb{E}[Y]
  $$

可以用來衡量兩個隨機變數之間的線性關係，如果 $\text{Cov}(X, Y) > 0$，表示 $X$ 和 $Y$ 傾向於同時增加或減少；如果 $\text{Cov}(X, Y) < 0$，表示當 $X$ 增加時，$Y$ 傾向於減少，反之亦然

- 如果 $X$ 和 $Y$ 是獨立的，則 $\text{Cov}(X, Y) = 0$，但反之不一定成立

- 性質:

$$
\text{Var}(aX + bY) = a^2 \text{Var}(X) + b^2 \text{Var}(Y) + 2ab \text{Cov}(X, Y)
$$

$$
\text{Cov}(aX + b, cY + d) = ac \text{Cov}(X, Y)
$$

$$
\text{Cov}(aX + bZ, cY + dW) = ac \text{Cov}(X, Y) + ad \text{Cov}(X, W) + bc \text{Cov}(Z, Y) + bd \text{Cov}(Z, W)
$$

### Multivariate Random Variables

$x = [x_1, x_2, \ldots, x_n]^T$ 是一個 n 維隨機向量 (random vector)

- 每個 $x_i$ 都是隨機變數，且彼此是 dependent 的
- $P(x)$ 就是 $x_1, x_2, \ldots, x_n$ 的 joint distribution

- Expectation Vector:

  $$
   \mathbb{E}[x] = \begin{bmatrix}
        \mathbb{E}[x_1] \newline
        \mathbb{E}[x_2] \newline
        \vdots \newline
        \mathbb{E}[x_n]
   \end{bmatrix} = \begin{bmatrix}
        \mathbb{\mu}_{x_1} \newline
        \mathbb{\mu}_{x_2} \newline
        \vdots \newline
        \mathbb{\mu}_{x_n}
   \end{bmatrix} = \mu_x
  $$

- Covariance Matrix:

  $$
    \begin{aligned}
   \text{Cov}(x) &= \begin{bmatrix}
        \text{Var}(x_1) & \text{Cov}(x_1, x_2) & \ldots & \text{Cov}(x_1, x_n) \newline
        \text{Cov}(x_2, x_1) & \text{Var}(x_2) & \ldots & \text{Cov}(x_2, x_n) \newline
        \vdots & \vdots & \ddots & \vdots \newline
        \text{Cov}(x_n, x_1) & \text{Cov}(x_n, x_2) & \ldots & \text{Var}(x_n)
   \end{bmatrix}  \newline
   &= \Sigma_x \newline
   &= \mathbb{E}[(x - \mu_x)(x - \mu_x)^T] \newline
   &= \mathbb{E}[xx^T] - \mu_x \mu_x^T
    \end{aligned}
  $$

  - 性質:
    - $\Sigma_x$ 是 symmetric
    - $\Sigma_x$ 是 positive semi-definite

### Derived Random Variables

令 $y = f(x;w) = w^T x$，其中 $y$ 是一個由隨機向量 $x$ 經過線性轉換得到的隨機變數，稱為 derived random variable

- 性質
  - $\mathbb{E}[y] = w^T \mathbb{E}[x]$
  - $\text{Var}(y) = w^T \text{Cov}(x) w = w^T \Sigma_x w$
  - 如果 $x$ 是 centralized 的 (mean zero)，則 $\mathbb{E}[y] = 0$

### Bayes' Rule

$$
P(A | B) = \frac{P(B | A) P(A)}{P(B)} = \frac{P(B | A) P(A)}{\sum_i P(B | A_i) P(A_i)}
$$

Bayes' Rule 可以用來反轉條件機率，例如從 $P(B | A)$ 推斷 $P(A | B)$，每一項都有自己的名字：

$$
P(A | B): \text{Posterior} = \frac{P(B | A): \text{Likelihood} \times P(A): \text{Prior}}{P(B): \text{Evidence}}
$$

### Point Estimation

用樣本資料計算一個單一數值，用來估計未知的母體參數，例如：

- 取樣本平均數作為母體平均數的估計

  $$
  \hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} x_i
  $$

- 取樣本比例作為母體比例的估計

  $$
  \hat{p} = \frac{1}{N} \sum_{i=1}^{N} I(x_i \in A)
  $$

- 取樣本協方差作為母體協方差的估計

  $$
  \hat{\Sigma}_x = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{\mu})(x_i - \hat{\mu})^T
  $$

  - 如果 $x$ 是 centralized 的 (mean zero)，則 $\hat{\Sigma}_x = \frac{1}{N} \sum_{i=1}^{N} x_i x_i^T$

### Principal Component Analysis (PCA)

#### 壓縮資訊

- 給定資料集 $X = {x_1, x_2, \ldots, x_N}$，每個 $x_i \in \mathbb{R}^D$
- 找到一個 function $f: \mathbb{R}^D \to \mathbb{R}^K$，其中 $K < D$，使得 $f(x_i)$ 能夠保留 $x_i$ 的大部分資訊
- 目標是最小化重建誤差 (reconstruction error)，保留最大的資訊量

#### 方法

- 假設 $x^{(i)}$ 為 randome variable $x$ 的 i.i.d. sample
- 假設 $f$ 是 Linear function，即 $f(x) = W^T x$，其中 $W \in \mathbb{R}^{D \times K}$
- PCA 的目標是找到 $K$ 個 orthonormal vectors $w_1, w_2, \ldots, w_K$ (principal components)，使得投影後的資料變異數 (variance) 最大化
- 為什麼 $w_1, w_2, \ldots, w_K$ 要是 orthonormal 的？
  - 避免 redundant information
- 為什麼 $\|w_i\| = 1$？
  - 避免 scale 的影響，不然可以無限放大 $w_i$ 來增加變異數的值

#### Optimization Problem

為了簡化，先考慮 $K = 1$ 的情況，我們要求最大的 $\text{Var}(z_1)$，其中 $z_1 = w_1^T x$

而 $\text{Var}(z_1)$ 又可以寫成

$$
\text{Var}(z_1) = \sigma_{z_1}^2 = w_1^T \Sigma_x w_1
$$

如果先把 $x$ centralized (mean zero)，則

$$
\hat{\Sigma}_x = \frac{1}{N} \sum_{i=1}^{N} x_i x_i^T = \frac{1}{N} X^TX
$$

接著，PCA 的目標可以寫成以下的優化問題：

$$
\begin{aligned}
\text{arg max}_{w_1 \in \mathbb{R}^D} \quad & w_1^T X^T X w_1, \text{ subject to } \|w_1\|_2 = 1 \newline
\end{aligned}
$$

又因為 $X^TX$ 是一個 real symmetric matrix，可以被 eigendecomposition 分解成

$$
X^TX = W \Lambda W^T
$$

套用 Rayleigh's Quotient 的結果，知道最大值會出現在最大的 eigen value 上，$w_1$ 就是其對應的 eigen vector

再來考慮 $w_2$，可以寫成以下的優化問題：

$$
\begin{aligned}
\text{arg max}_{w_2 \in \mathbb{R}^D} \quad & w_2^T X^T X w_2, \newline
\text{ subject to } & \|w_2\|_2 = 1, & w_2^T w_1 = 0 \newline
\end{aligned}
$$

所以，對於一般情況， $w_1, w_2, \ldots, w_K$ 就是 $X^TX$ 前 $K$ 個最大的 eigen values 所對應的 eigen vectors

### Technical Details

#### Sure & Almost Sure Events

- Sure Event: 發生機率為 1 的事件，沒有任何例外情況
  - 擲一個公平的六面骰子，得到的點數一定在 1 到 6 之間
- Almost Sure Event: 發生機率為 1 的事件，但可能包含一些 measure zero 的例外情況
  - 在連續隨機變數中，取到某個特定值的機率為 0，但這並不代表該事件不可能發生

#### Equality of Random Variables

感覺沒那麼重要，之後再補充

- Equality in Distribution
- Almost Sure Equality
- Equality

#### Convergence of Random Variables

感覺沒那麼重要，之後再補充

- Convergence in Distribution
- Convergence in Probability
- Almost Sure Convergence

#### Distribution of Derived Variables

令 $y = f(x)$ 且 $f^{-1}$ 存在，那 $P(y = y) = P(x = f^{-1}(y))$ 一定成立嗎？

如果 $x$ 和 $y$ 都是連續的，**不成立**

令 $x \sim \text{Uniform}(0, 1)$ 是連續的，且 $p(x) = c$ for $x \in [0, 1]$

令 $y = \frac{x}{2}$，則 $y \sim \text{Uniform}(0, 0.5)$

如果 $p_y(y) = p_x(f^{-1}(y)) = p_x(2y) = c$，則

$$
\int_{0}^{0.5} p_y(y) dy = \int_{0}^{0.5} c dy = 0.5c =  0.5 \ne 1
$$

不符合 PDF 的定義，所以不成立

#### Jacobian Adjustment

我們知道 $Pr(y = y) = p_y(y) dy$，且 $Pr(x = x) = p_x(x) dx$ 且

$$
\|p_y(y) dy\| = \|p_x(x) dx\|
$$

可以得到

$$
p_y(y) = p_x(f^{-1}(y)) \left| \frac{d f^{-1}(y)}{dy} \right|
$$

而在 multivariate 的情況下，Jacobian matrix $J_{f^{-1}}(y)$ 的行列式 (determinant) 就是我們要的調整因子

$$
p_y(y) = p_x(f^{-1}(y)) \left| \det(J_{f^{-1}}(y)) \right|
$$

其中

$$
J_{f^{-1}}(y) = \begin{bmatrix}
    \frac{\partial f_1^{-1}(y)}{\partial y_1} & \frac{\partial f_1^{-1}(y)}{\partial y_2} & \ldots & \frac{\partial f_1^{-1}(y)}{\partial y_n} \newline
    \frac{\partial f_2^{-1}(y)}{\partial y_1} & \frac{\partial f_2^{-1}(y)}{\partial y_2} & \ldots & \frac{\partial f_2^{-1}(y)}{\partial y_n} \newline
    \vdots & \vdots & \ddots & \vdots \newline
    \frac{\partial f_n^{-1}(y)}{\partial y_1} & \frac{\partial f_n^{-1}(y)}{\partial y_2} & \ldots & \frac{\partial f_n^{-1}(y)}{\partial y_n} \newline
\end{bmatrix}
$$

因為 function $f$ 在計算時可能會 distort 空間的體積，Jacobian determinant 就是用來調整這個體積變化的因子，在一維的情況下，就是導數的絕對值

### Probability Distributions

#### Bernoulli Distribution (Discrete)

給定一個參數 $p$，表示事件發生的機率，則 Bernoulli 分佈的 PMF 為

$$
P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
$$

- Expectation: $\mathbb{E}[X] = p$

  $$
  \begin{aligned}
  \mathbb{E}[X] &= \sum_{x \in \{0, 1\}} x P(X = x) \newline
  &= 0 \cdot P(X = 0) + 1 \cdot P(X = 1) \newline
  &= 0 \cdot (1-p) + 1 \cdot p \newline
  &= p \newline
  \end{aligned}
  $$

- Variance: $\text{Var}(X) = p(1-p)$

  $$
  \begin{aligned}
  \text{Var}(X) &= \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \newline
  &= \sum_{x \in \{0, 1\}} x^2 P(X = x) - p^2 \newline
  &= 0^2 \cdot P(X = 0) + 1^2 \cdot P(X = 1) - p^2 \newline
  &= 0^2 \cdot (1-p) + 1^2 \cdot p - p^2 \newline
  &= p - p^2 \newline
  &= p(1-p) \newline
  \end{aligned}
  $$

#### Categorical Distribution (Discrete)

給定一個參數向量 $p = [p_1, p_2, \ldots, p_K]$，表示 $K$ 個類別的機率，則 Categorical 分佈的 PMF 為

$$
P(X = k) = p_k, \quad k \in \{1, 2, \ldots, K\}
$$

or

$$
P(X = k) = \prod_{i=1}^{K} p_i^{I(X = i)}
$$

- $I(X = i)$: indicator function，當 $X = i$ 時為 1，否則為 0

#### Multinomial Distribution (Discrete)

給定一個參數向量 $p = [p_1, p_2, \ldots, p_K]$，表示 $K$ 個類別的機率，且有 $n$ 次獨立試驗，則 Multinomial 分佈的 PMF 為

$$
P(X_1 = x_1, X_2 = x_2, \ldots, X_K = x_K) = \frac{n!}{x_1! x_2! \ldots x_K!} p_1^{x_1} p_2^{x_2} \ldots p_K^{x_K}
$$

- $x_i$: 第 $i$ 類別的次數，且 $\sum_{i=1}^{K} x_i = n$

#### Normal/Gaussian Distribution (Continuous)

給定一個參數 $\mu$ 和 $\sigma^2$，表示平均數和變異數，則 Normal 分佈的 PDF 為

$$
p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{(x - \mu)^2}{2 \sigma^2} \right)
$$

- Expectation: $\mathbb{E}[X] = \mu$
- Variance: $\text{Var}(X) = \sigma^2$

好處：

- 提供最多的 Uncertainty (最大熵)
- 數學上很好處理 (continuous, differentiable)

性質

- 如果 $X \sim \mathcal{N}(\mu, \sigma^2)$，則 $Y = aX + b \sim \mathcal{N}(a\mu + b, a^2 \sigma^2)$

  $$
  \begin{aligned}
  \mathbb{E}[Y] &= \mathbb{E}[aX + b] = a \mathbb{E}[X] + b = a\mu + b \newline
  \text{Var}(Y) &= \text{Var}(aX + b) = a^2 \text{Var}(X) = a^2 \sigma^2 \newline
  \end{aligned}
  $$

  - 所以做了 `z-normalization (standardization)` 後，$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$

- 如果 $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$，$X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$，且 $X_1$ 和 $X_2$ 獨立，則 $Y = X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$
  $$
  \begin{aligned}
  \mathbb{E}[Y] &= \mathbb{E}[X_1 + X_2] = \mathbb{E}[X_1] + \mathbb{E}[X_2] = \mu_1 + \mu_2 \newline
  \text{Var}(Y) &= \text{Var}(X_1 + X_2) = \text{Var}(X_1) + \text{Var}(X_2) = \sigma_1^2 + \sigma_2^2 \newline
  \end{aligned}
  $$

#### Multivariate Normal/Gaussian Distribution

給定一個參數向量 $\mu \in \mathbb{R}^D$ 和協方差矩陣 $\Sigma \in \mathbb{R}^{D \times D}$，則 Multivariate Normal 分佈的 PDF 為

$$
p(x) = \frac{1}{(2 \pi)^{D/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)
$$

有空補

### Parametrizing Functions

用參數化的函數來描述資料的生成過程，例如：

- 線性模型: $y = w^T x + b$
- 決策樹: $y = f(x; \theta)$，其中 $\theta$ 是樹的結構和分裂規則
- 神經網路: $y = f(x; \Theta)$，其中 $\Theta$ 是網路的權重和偏差

#### Logistic Function

$$
\sigma(z) = \frac{1}{1 + \exp(-z)}
$$

- 將實數映射到 (0, 1) 之間
- 常用於二元分類問題的輸出層

#### Softplus Function

$$
\zeta(z) = \log(1 + \exp(z))
$$

- soft 版本的 ReLU 函數

#### Softmax Function

$$
\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j} \exp(z_j)}
$$

- 將實數向量映射到機率分佈
- 常用於多元分類問題的輸出層

### Information Theory

**Probability** 能夠在不確定的情況下，量化我們對事件發生的信心程度

而 **Information Theory** 則是量化不確定性本身，以及我們從觀察事件中獲得的資訊量

#### Self-Information

事件 $x$ 發生所帶來的資訊量

$$
 I(x) = -\log P(x)
$$

- 如果 $P(x)$ 越小，則 $I(x)$ 越大，表示罕見事件帶來更多資訊
- 如果 $P(x) = 1$，則 $I(x) = 0$，表示確定事件不帶來任何資訊

#### Entropy

隨機變數 $X$ 的不確定性

##### Shannon Entropy:

$$
H(X) = -\sum_{x} P(x) \log P(x)
$$

##### Differential Entropy (Continuous):

$$
H(X) = -\int p(x) \log p(x) dx
$$

- Entropy 衡量一個隨機變數的不確定性或資訊量
- Entropy 越大，表示不確定性越高
- $0 \log 0$ 被定義為 $\text{lim}_{p \to 0^+} -p \log p = 0$

#### Average Code Length

Shannon Entropy 是最小平均編碼長度的下界，表示在最佳編碼方案下，平均每個符號所需的位元數

- Example:

  假設有四個符號 $A, B, C, D$，其機率分別為 $P(A) = 0.5, P(B) = 0.25, P(C) = 0.15, P(D) = 0.1$

  - Shannon Entropy 為

    $$
     H(X) = -\sum_{x \in \{A, B, C, D\}} P(x) \log_2 P(x) = 1.7427 \text{ bits}
    $$

  - 一種可能的編碼方案為:

    - $A$: 0
    - $B$: 10
    - $C$: 110
    - $D$: 111

  - 平均編碼長度為

    $$
     L = \sum_{x \in \{A, B, C, D\}} P(x) \cdot \text{length}(code(x)) = 1.75 \text{ bits}
    $$

  雖然平均編碼長度 $L = 1.75$ bits 大於 Shannon Entropy $H(X) = 1.7427$ bits，但已經非常接近，表示這是一個有效的編碼方案

#### Kullback-Leibler (KL) Divergence

用來衡量兩個機率分佈 $P$ 和 $Q$ 之間的差異

當 $P$ 是真實分佈，$Q$ 是近似分佈時，KL Divergence 衡量使用 $Q$ 來近似 $P$ 所帶來的資訊損失

$$
\begin{aligned}
D_{KL}(P || Q) &= \sum_{x} P(x) \log \frac{P(x)}{Q(x)} \newline
&= \mathbb{E}_{x \sim P} \left[ \log \frac{P(x)}{Q(x)} \right] \newline
&= \mathbb{E}_{x \sim P} [\log P(x) - \log Q(x)] \newline
&= -H(P) - \mathbb{E}_{x \sim P} [\log Q(x)] \newline
\end{aligned}
$$

其中 $\mathbb{E}_{x \sim P} [\log Q(x)]$ 是 **cross-entropy**

如果 $P$ 和 $Q$ 是 Independent 的，$H(P)$ 不依賴於 $Q$，所以可以忽略，則

$$
\begin{aligned}
\arg\min_Q D_{KL}(P || Q) &= \arg\min_Q -\mathbb{E}_{x \sim P} [\log Q(x)] \newline
&= \arg\max_Q \mathbb{E}_{x \sim P} [\log Q(x)] \newline
\end{aligned}
$$

- 性質
  - $D_{KL}(P || Q) \geq 0$，且當且僅當 $P = Q$ 時，等號成立
  - $D_{KL}(P || Q) \ne D_{KL}(Q || P)$，不對稱

##### Minimizer of KL Divergence

給定分布 $P$，尋找分布 $Q^*$ 使得 $D_{KL}(P || Q^*)$ 最小化

因為 KL Divergence 是不對稱的，我們有兩種方法：

1. Forward KL Divergence: $\arg\min_Q D_{KL}(P || Q)$
2. Reverse KL Divergence: $\arg\min_Q D_{KL}(Q || P)$

### Decision Trees

Decision Trees 是一種監督式學習演算法，用於分類和迴歸問題

- Information Gain

  $$
  IG(D, A) = H(D) - \sum_{v \in \text{Values}(A)} \frac{|D_v|}{|D|} H(D_v)
  $$

  - $D$: 資料集
  - $A$: 特徵
  - $D_v$: 在特徵 $A$ 上取值為 $v$ 的子集
  - $H(D)$: 資料集 $D$ 的熵
  - $\text{Values}(A)$: 特徵 $A$ 的所有可能取值

  選擇 Information Gain 最大的特徵來分裂節點，可以把 $H$ 看成是 Impurity 的量度，Impurity 越低，表示資料越純淨，$H$ 也越低

1. 選擇 Information Gain 最大的特徵 $A^*$ 來分裂節點
2. 對每個可能的取值 $v$，創建一個子節點 $D_v$
3. 重複以上步驟，直到滿足停止條件 (例如，節點中的樣本數小於某個閾值，或所有樣本屬於同一類別)

### Random Forests

Decision Tree 通常非常深，越深的節點越少經過訓練資料，容易 overfitting

Random Forests 是一種集成學習方法，通過結合多個 Decision Trees 來提高模型的泛化能力

1. Bootstrap Aggregating (Bagging): 從原始資料集中有放回地抽取多個子集，對每個子集訓練一棵 Decision Tree
2. Feature Randomness: 在每個節點分裂時，隨機選擇 $k$ 個特徵來考慮，找出 Information Gain 最大的特徵來分裂節點
3. 重複以上步驟，直到生成 $N$ 棵 Decision Trees
4. 預測時，對所有 Decision Trees 的預測結果進行投票 (分類) 或平均 (迴歸)

## Numerical Optimization

### Numerical Computation

在機器學習中，會有大量的浮點數計算，受限於浮點數儲存的精度，有時候會產生 **numeric errors**

#### Overflow & Underflow

對於 Softmax 函數來說

$$
\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j} \exp(z_j)}
$$

如果對於 $z_i = c \ \forall i$，如果 $|c|$ 很大

- 如果 $c$ 是正數，則 $\exp(c)$ 會 overflow
- 如果 $c$ 是負數，則 $\exp(c)$ 會 underflow，還有可能導致分母為 0

為了解決這個問題，可以對 $z$ 進行平移 (shift)：

$$
\text{softmax}(z_i) = \frac{\exp(z_i - z_{\max})}{\sum_{j} \exp(z_j - z_{\max})}
$$

這樣的計算結果會和原本的 softmax 函數相同，但可以避免 overflow 和 underflow 的問題

- 分子最多是 $\exp(0) = 1$，不會 overflow
- 分母至少有一項是 $\exp(0) = 1$，不會為 0

$$
\begin{aligned}
\text{softmax}(z_i) &= \frac{\exp(z_i - z_{\max})}{\sum_{j} \exp(z_j - z_{\max})} \newline
&= \frac{\exp(z_i) \exp(-z_{\max})}{\sum_{j} \exp(z_j) \exp(-z_{\max})} \newline
&= \frac{\exp(z_i)}{\sum_{j} \exp(z_j)} \newline
\end{aligned}
$$

#### Poor Conditioning

**Condition number** 是用來衡量一個函數對輸入變化的敏感度

像是我們有 $f(x) = Ax = y$，其中 $A$ 是一個矩陣，且 $A^{-1}$ 存在

那麼其 condition number 定義為

$$
\kappa(A) = \max_{i, j} \frac{|\lambda_i|}{|\lambda_j|}
$$

- $\lambda_i$: $A$ 的第 $i$ 個 eigen value，如前面所述，可以想成是對某個方向的伸縮最大值和最小值的比值
- 當 $\kappa(A)$ 很大時，對 $x$ 的微小變化會導致 $y$ 有很大的變化，這會影響到優化算法的收斂速度和穩定性，稱為 **ill-conditioned**
  - 在解 $x = A^{-1} y$ 時，會放大 $y$ 的 numeric errors，導致 $x$ 有很大的誤差

### Optimization Problems

Optimization problem 的目標是去最小化一個 **cost function** $f: \mathbb{R}^d \to \mathbb{R}$

$$
\text{argmin}_{x \in \mathbb{R}^d} f(x) \newline
\text{subject to } x \in \mathcal{C}
$$

$\mathcal{C} \subseteq \mathbb{R}^d$: constraint set，表示 $x$ 必須滿足的約束條件，又稱 為 feasible set，$x$ 稱為 feasible point

- 例如: $\mathcal{C} = \{x : g_i(x) \leq 0, h_j(x) = 0\}$
- 如果 $\mathcal{C} = \mathbb{R}^d$，稱為 unconstrained optimization problem
- 如果是最大化 objective function 問題，可以把目標改成最小化 $-f(x)$

#### Examples

- Critical Point: $\mathbf{C} = \{x : \nabla f(x) = 0\}$
  - Minima: $\mathbf{C} = \{x : \nabla f(x) = 0, H(f)(x) \succ 0\}$
  - Maxima: $\mathbf{C} = \{x : \nabla f(x) = 0, H(f)(x) \prec 0\}$
  - Plateau/Saddle Point: $\mathbf{C} = \{x : \nabla f(x) = 0, H(f)(x) = \mathbf{O} \ \text{or} \text{ indefinite}\}$
- Global Minimum: $\min_{x \in \mathcal{C}} f(x) \in \mathbf{R}$
- Optimal Point: $x^* \in \mathcal{C}$ such that $f(x^*) = \min_{x \in \mathcal{C}} f(x)$

#### Convex Optimization

滿足以下條件的 optimization problem，稱為 convex optimization problem

1. $H(f)(x) \succeq 0$ for all $x \in \mathcal{C}$
2. $g_i(x)$ 是 convex function for all $i$
3. $h_j(x)$ 是 affine function for all $j$
   - affine function: $h(x) = Ax + b$，就是 Linear + Constant shift

### Gradient Descent

根據 Taylor expansion，可以用函數 $\tilde{f}(x)$ 的多項式
近似來描述函數在某一點附近的行為，例如在點 $a$ 附近，可以用一階 Taylor expansion 來近似函數：

$$
f(x) \approx \tilde{f}(x) = f(a) + \nabla f(a)^T (x - a)
$$

當我們選擇 $x = a - \eta \nabla f(a)$ for some $\eta > 0$ 時，可以得到

$$
\tilde{f}(x) = f(a) - \eta \|\nabla f(a)\|^2 \leq \tilde{f}(a)
$$

#### Negative Gradient is the Direction of Steepest Descent

給定一個 function $f$，一個方向 $u$ 和點 $a$

Directional Derivative:

$$
D_u f(a) = \lim_{h \to 0} \frac{f(a + h u) - f(a)}{h} = \nabla f(a)^T u
$$

我們想找到一個單位向量 $u$，使得 $D_u f(a)$ 最小化

$$
\begin{aligned}
\arg\min_{u, \|u\| = 1} D_u f(a) &= \arg\min_{u, \|u\| = 1} \nabla f(a)^T u \newline
&= \arg\min_{u, \|u\| = 1} \|\nabla f(a)\| \|u\| \cos \theta \newline
&= \arg\min_{u} \cos \theta \newline
\end{aligned}
$$

所以，當 $u = -\frac{\nabla f(a)}{\|\nabla f(a)\|}$ 時，也就是負梯度方向，Directional Derivative 會達到最小值

#### Set Learning Rate

有空再補

#### Problems of Gradient Descent

沒有考慮到 $H(f)(x)$ 的 Conditioning 的問題，可能在某個方向下降很快，但在另一個方向下降很慢，導致整體收斂速度變慢

- Zig-Zags

### Newton's Method

根據二維 Taylor expansion，可以用函數 $\tilde{f}(x)$ 的多項式近似來描述函數在某一點附近的行為，例如在點 $a$ 附近，可以用二階 Taylor expansion 來近似函數：

$$
f(x) \approx \tilde{f}(x) = f(a) + \nabla f(a)^T (x - a) + \frac{1}{2} (x - a)^T H(f)(a) (x - a)
$$

當 $f$ 是 strictly convex 時，我們可以去解 $\nabla \tilde{f}(x) = 0$，來找到 $\tilde{f}(x)$ 的最小值，得到

$$
 a - H(f)(a)^{-1} \nabla f(a)
$$

- $H(f)(a)$ 就是 gradient 的 corrector

在這種情況下，只要不斷找

$$
x^{(k+1)} = x^{(k)} - \eta H(f)(x^{(k)})^{-1} \nabla f(x^{(k)})
$$

就可以快速收斂

當 $f$ 不是 strictly convex 時，$H(f)(x) \preceq 0$ 或 indefinite，會導致找出來的方向不是下降方向，因此會需要 **Levenberg-Marquardt adjustment**

$$
x^{(k+1)} = x^{(k)} - \eta (H(f)(x^{(k)}) + \lambda I)^{-1} \nabla f(x^{(k)})
$$

選一個足夠大的 $\lambda$，可以確保 $H(f)(x^{(k)}) + \lambda I \succ 0$，使得方向為下降方向

#### Problems of Newton's Method

- 計算 $H(f)(x)$ 和 $H(f)(x)^{-1}$ 的時間和空間複雜度都很高，尤其是在高維度的情況下
  - 時間複雜度: $O(d^3)$
  - 空間複雜度: $O(d^2)$
- $H(f)(x)$ 可能有很大的 Condition Number
  - 當 gradient 有 numeric errors 時，會被放大，導致方向不正確，越往後誤差越大
- 可能會收斂到 Saddle Point
  - 因為在 Saddle Point 處，$H(f)(x)$ 是 indefinite，可能會導致找出來的方向不是下降方向

### Optimization in Machine Learning

大部分的機器學習模型都有 convex function，e.g. Linear Regression, Logistic Regression, SVM...

但是在深度學習中，通常都是 non-convex function，e.g. Neural Networks

#### Lipschitz Continuity

我們通常會假設 Cost Function $C$ 是 Lipschitz Continuous 的，表示存在一個常數 $K > 0$，使得：

$$
\|C(w^1) - C(w^2)\| \leq K \|w^1 - w^2\|, \quad \forall w^1, w^2 \in \mathbb{R}^d
$$

這個條件保證了函數的變化不會太劇烈，有助於優化算法的收斂性分析

![Lipschitz Continuity](https://upload.wikimedia.org/wikipedia/commons/8/8d/Lipschitz_continuity.png)

#### Perceptron Learning Algorithm

Binary Classification 問題

假設資料集 $D = {(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)}$，其中 $x_i \in \mathbb{R}^d$ 是特徵向量，$y_i \in \{-1, 1\}$ 是標籤

定義線性分類器 $f(x) = sign(w^T x + b)$

- 其中 $w \in \mathbb{R}^d$ 是權重向量，$b \in \mathbb{R}$ 是偏差
- 也可以把 $b$ 合併到 $w$ 中，令 $\tilde{x} = [x; 1]$，$\tilde{w} = [w; b]$，則 $f(x) = \tilde{w}^T \tilde{x}$

訓練時，每個 Epoch 會根據每筆資料 $(x^t, y^t)$ 來更新權重 $w$

$$
w^{(t+1)} = w^{(t)} + \eta (y^t - \hat{y}^t) x^t
$$

- 如果 $\hat{y}^t = y^t$，則不更新
- 如果 $\hat{y}^t \ne y^t$，得到 $w^{(t+1)} = w^{(t)} + 2 \eta y^t x^t$

  - $y^t = 1$
    $$
    \begin{aligned}
    \text{sign}(w^{(t+1)T} x^t) &= \text{sign}((w^{(t)}  + 2 \eta x^t)^T x^t) \newline
    &= \text{sign}(w^{(t)T} x^t + 2 \eta \|x^t\|^2) \newline
    &= \text{sign}(w^{(t)T} x^t + c)
    \end{aligned}
    $$
    因為 $c > 0$，所以 $\text{sign}(w^{(t+1)T} x^t)$ 會傾向於變成 1

如果資料沒辦法線性可分，則 Perceptron Learning Algorithm 可能無法收斂

#### ADAptive LInear NEuron (Adaline)

Adaline 是一種線性分類器，與 Perceptron 類似，但使用連續的輸出值來進行學習

Cost Function:

$$
\arg\min_{w} \frac{1}{2} \sum_{i=1}^{N} (y_i - w^T x_i)^2
$$

訓練完之後，使用 $f(x) = sign(w^T x)$ 來進行分類

Update Rule:

$$
w^{(t+1)} = w^{(t)} + \eta \sum_{i=1}^{N} (y_i - w^{(t)T} x_i) x_i
$$

因為 Cost function 是 Convex 的，所以可以保證可以收斂到全域最小值

#### Stochastic Gradient Descent (SGD)

把資料集 $D$ 分成多個 mini-batch，每次只使用一個 mini-batch 來更新權重

$$
w^{(t+1)} = w^{(t)} - \eta \nabla C_{MB}(w^{(t)})
$$

- $C_{MB}(w)$: mini-batch 上的 Cost Function
- 每次更新只需要計算 mini-batch 的 gradient，計算量較小，適合大規模資料集
- 支援 Online Learning

### Constrained Optimization

Problem:

$$
\text{min}_{x} f(x) \newline
\text{subject to } x \in \{x : g_i(x) \leq 0, h_j(x) = 0\}
$$

#### Karush-Kuhn-Tucker (KKT) Methods

可以把問題轉成

$$
\text{min}_x\text{max}_{\alpha, \beta, \alpha \geq 0} \mathcal{L}(x, \alpha, \beta) = \newline
\text{min}_x\text{max}_{\alpha, \beta, \alpha \geq 0} f(x) + \sum_{i} \alpha_i g_i(x) + \sum_{j} \beta_j h_j(x) \newline
$$

當 $x$ 是 Feasible Point 時，$\mathcal{L}(x, \alpha, \beta) = f(x)$，因為 $g_i(x) \leq 0$ 和 $h_j(x) = 0$，會選 $\alpha_i = 0$ 和任意 $\beta_j$ 來 maximize $\mathcal{L}(x, \alpha, \beta)$

當 $x$ 不是 Feasible Point 時，會有某些 $g_i(x) > 0$ 或 $h_j(x) \ne 0$，這時候可以選擇很大的 $\alpha_i$ 或 $\beta_j$ 來讓 $\mathcal{L}(x, \alpha, \beta)$ 變無限大，這樣就不會選擇這些 $x$ 來 minimize $\mathcal{L}(x, \alpha, \beta)$

#### KKT Conditions

假設 $f, g_i, h_j$ 都是可微分的，且 $x^*$ 是問題的最優解，則存在 $\alpha^*, \beta^*$，使得以下條件成立：

1. Primal Feasibility: $g_i(x^*) \leq 0$ for all $i$, $h_j(x^*) = 0$ for all $j$
2. Dual Feasibility: $\alpha_i^* \geq 0$ for all $i$
3. Stationarity: $\nabla f(x^*) + \sum_{i} \alpha_i^* \nabla g_i(x^*) + \sum_{j} \beta_j^* \nabla h_j(x^*) = 0$
4. Complementary Slackness: $\alpha_i^* g_i(x^*) = 0$ for all $i$

#### Complementary Slackness

- 如果 $g_i(x^*) = 0$，稱為 active，$\alpha_i^* g_i(x^*) = 0$

- 如果 $g_i(x^*) < 0$，稱為 inactive，為了 maximize $\mathcal{L}(x^*, \alpha^*, \beta^*)$，必須有 $\alpha_i^* = 0$

今天我有一個 $\alpha_i^* > 0$，我就可以知道 $g_i(x^*)$ 一定是 0，可以快速找到 active constraints，就可以確定最優解真正被哪些限制所決定

### The Regression Problem

給定資料集 $D = {(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)}$，其中 $x_i \in \mathbb{R}^d$ 是特徵向量，$y_i \in \mathbb{R}$ 是目標值

目標是找到一個函數 $f: \mathbb{R}^d \to \mathbb{R}$，使得對於所有的 $(x_i, y_i)$，$f(x_i)$ 盡可能接近 $y_i$

#### Sum of Squared Errors (SSE):

$$
SSE(f; D) = \sum_{i=1}^{N} (y_i - f(x_i))^2
$$

#### Mean Squared Error (MSE):

$$
MSE(f; D) = \frac{1}{N} \sum_{i=1}^{N} (y_i - f(x_i))^2
$$

#### Relative Squared Error (RSE):

$$
RSE(f; D) = \frac{\sum_{i=1}^{N} (y_i - f(x_i))^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}
$$

- $\bar{y} = \frac{1}{N} \sum_{i=1}^{N} y_i$: 目標值的平均值
- 一般來說，RSE 越小表示模型越好，如果算出來大於 1，表示模型比直接用平均值來預測還差，你乾脆直接用平均值算就好
- R-squared ($R^2$) 是用來衡量模型解釋變異的比例
  $$
  R^2 = 1 - RSE
  $$
  - $R^2$ 越接近 1，表示模型解釋變異的能力越強

#### Data Augmentation

在機器學習中，可以把 Linear model 轉換成多項式特徵，或是加入其他變換，以增加模型的表達能力

[HW] How many variables to solve in $w$ for a polynomial regression problem with degree $P$

### Regularization

為了 Generalization ，而不是只在訓練資料上表現好，可以在 Cost Function 中加入 Regularization term，來懲罰過於複雜的模型

$$
\text{argmin}_{w \in \mathbb{R}^d} \frac{1}{2} \| y - (Xw - b)\|^2 \newline
\text{subject to } \|w\|^2 \leq R
$$

可以寫成

$$
\text{argmin}_{w \in \mathbb{R}^{d+1}} \frac{1}{2} \| y - (Xw )\|^2 \newline
\text{subject to } w^T S w \leq R
$$

其中 $S = \begin{bmatrix} 0 & 0 \\ 0 & I_d \end{bmatrix}$

By KKT，可以轉成 unconstrained problem

$$
\text{argmin}_w \text{max}_{\alpha, \alpha \ge 0} \frac{1}{2} (\| y - (Xw )\|^2 + \alpha(  w^T S w - R))
$$

### Dual Problem

給定 primal problem:

$$
p^* = \text{min}_x \text{max}_{\alpha, \beta, \alpha \geq 0}  \mathcal{L}(x, \alpha, \beta)
$$

dual problem 定義為:

$$
d^* = \text{max}_{\alpha, \beta, \alpha \geq 0} \text{min}_x  \mathcal{L}(x, \alpha, \beta)
$$

[HW] By max-min inequality, we have

$$
d^* \leq p^*
$$

當 primal problem 是 convex，且有解，則 strong duality 成立，$d^* = p^*$，有時候解問題會更有效率

#### Example

考慮

$$
\text{argmin}_{w \in \mathbb{R}^d} \frac{1}{2} \|x\|^2 \newline
\text{subject to } Ax \geq b, A \in \mathbb{R}^{n \times d}, b \in \mathbb{R}^n
$$

By KKT，可以寫成

$$
\text{argmin}_x \text{max}_{\alpha, \alpha \geq 0} \frac{1}{2} \|x\|^2 + \alpha^T (b - Ax)
$$

因為是 Convex problem，滿足 Strong Duality，可以交換 min 和 max

$$
\text{argmax}_{\alpha, \alpha \geq 0} \text{min}_x \frac{1}{2} \|x\|^2 + \alpha^T (b - Ax)
$$

對 $x$ 求導並設為 0，得到 $x = A^T \alpha$，帶回上式，得到

$$
\text{argmax}_{\alpha, \alpha \geq 0} -\frac{1}{2} \|A^T \alpha\|^2 + b^T \alpha
$$

現在只需要解 $n$ 維的問題，而不是 $d$ 維的問題，當 $n \ll d$ 時，會更有效率

## Learning Theory

#### Empirical Error / risk

Empirical Error 是在訓練資料集上計算的平均 Loss

$$
E_N[f] = \frac{1}{N} \sum_{i=1}^{N} \text{loss}(f(x_i; w), y_i)
$$

#### Generalization Error / risk

Generalization Error 是在整個資料分佈上計算的平均 Loss，代表模型在未見過的資料上的表現

$$
C[f] = \int \text{loss}(f(x; w), y) dP(x, y)
$$

Learining Theory 就是在理解怎麼去 Characterize

$$
C[f_N] = \int \text{loss}(f_N(x; w), y) dP(x, y)
$$

- $f_N$: model learned from training data of size $N$
- $C[f_N]$: generalization error

### Bounding Methods

$min_fC[f] = C[f^*]$ 稱為 Bayes Error，表示在給定資料分佈下，最佳模型的誤差下限

- 如果 $P(y|x)$ 有隨機性，則 Bayes Error > 0
- 目標是讓 $C[f_N]$ 越接近 $C[f^*]$ 越好

$\Epsilon = C(f_N) - C(f^*)$ 稱為 Excess Error，又可以寫成

$$
\Epsilon = (C(f_F^*) - C(f^*)) + (C(f_N) - C(f_F^*))
$$

其中前項稱為 Approximation Error，後項稱為 Estimation Error

- 選越複雜的 $F$，Approximation Error 越小，但 Estimation Error 越大

###

## Maximum Likelihood Estimation (MLE)

- 假設資料是獨立同分佈 (i.i.d)
- 給定資料集 $D = {x_1, x_2, \ldots, x_N}$，假設資料來自某個參數化的分佈 $p(x|\theta)$
- 目標是找到參數 $\theta$，使得在該參數下，資料出現的機率最大
- 定義似然函數 (Likelihood Function) 為資料在參數 $\theta$ 下的聯合機率
- $L(\theta; D) = p(D|\theta) = \prod_{i=1}^{N} p(x_i|\theta)$
- 最大化似然函數等價於最大化對數似然函數 (Log-Likelihood Function)
- $\ell(\theta; D) = \log L(\theta; D) = \sum_{i=1}^{N} \log p(x_i|\theta)$
- 最大化對數似然函數的參數 $\theta$ 即為 MLE
- $\hat{\theta}_{MLE} = \arg\max_{\theta} \ell(\theta; D)$

## Large-Scale Machine Learning

Solve problems by leveraging the posteriror knowledge learned from the big data.

- Characteristics of Big Data:
- Volume: 大量的資料
- Variety: 多樣化的資料類型和來源
- Velocity: 單位時間的資料量
- Veracity: 資料的真實性和可靠性

## Neural Networks: Design

- Feedforward neural networks (FNN) 又稱 multi-layer perceptron (MLP)

$$

\begin{aligned}
\hat{y} = f^{(L)}(\ldots f^{(2)}(f^{(1)}(x; \theta^{(1)}); \theta^{(2)}) \ldots; \theta^{(L)})
\end{aligned}


$$

- $L$: number of layers
- $f^{(l)}$: nonlinear function of layer $l$
- $\theta^{(l)}$: parameters of layer $l$
- $x$: input vector

- $f^{(k)}$ outputs value $a^{(k)}$, where

$$

\begin{aligned}
a^{(k)} &= f^{(k)}(a^{(k-1)}; \theta^{(k)}) \newline
&= \text{act}^{(k)}(W^{(k)\top} a^{(k-1)} + b^{(k)}) \newline
\end{aligned}


$$

- $W^{(k)}$: weight matrix of layer $k$
- $b^{(k)}$: bias vector of layer $k$
- $\text{act}^{(k)}$: activation function of layer $k$
- $a^{(0)} = x$

沒有非線性，模型就沒有深度，等價於單層線性模型而已。

### Training an NN

- Most NNs are trained using the **maximum likelihood** by default.

$$

\begin{aligned}
\text{argmax}_{\Theta} \text{log } P(X | \Theta) &= \text{argmin}_{\Theta} -\text{log } P(X | \Theta) \newline
&= \text{argmin}_{\Theta} \Sigma_i -\text{log } P(x_i, y_i | \Theta) \newline
&= \text{argmin}_{\Theta} \Sigma*i [-\text{log } P(y_i | x_i, \Theta) - \text{log } P(x_i | \Theta)] \newline
&= \text{argmin}*{\Theta} \Sigma*i -\text{log } P(y_i | x_i, \Theta) \quad \text{(if we ignore } P(x_i | \Theta)) \newline
&= \text{argmin}*{\Theta} \Sigma_i C_i (\Theta)
\end{aligned}


$$

- 通常 $P(x_i | \Theta)$ 不依賴於模型參數 $\Theta$，可以把他想像成常數，最小化時不會影響結果，所以可以忽略

- decision
- xgboost
- lightgbm
- rnn
- lstm
  https://blog.nbswords.com/2020/12/xgboostlightgbmcatboost.html
  $$
