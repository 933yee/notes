---
title: Deep Learning
date: 2025-10-07 12:09:09
tags: [deep learning, ai, machine learning]
category: AI
math: true
---

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
