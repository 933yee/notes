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
   \|A\|_F = \sqrt{\sum_{i,j} A_{i,j}^2} = \sqrt{\text{trace}(A^TA)}
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

### Maximum Likelihood Estimation (MLE)

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
    &= \text{argmin}_{\Theta} \Sigma_i [-\text{log } P(y_i | x_i, \Theta) - \text{log } P(x_i | \Theta)] \newline
    &= \text{argmin}_{\Theta} \Sigma_i -\text{log } P(y_i | x_i, \Theta) \quad \text{(if we ignore } P(x_i | \Theta)) \newline
    &= \text{argmin}_{\Theta} \Sigma_i C_i (\Theta)
\end{aligned}
$$

- 通常 $P(x_i | \Theta)$ 不依賴於模型參數 $\Theta$，可以把他想像成常數，最小化時不會影響結果，所以可以忽略

- decision
- xgboost
- lightgbm
- rnn
- lstm
  https://blog.nbswords.com/2020/12/xgboostlightgbmcatboost.html
