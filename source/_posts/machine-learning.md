---
title: 機器學習筆記
date: 2025-02-12 13:22:11
tags: ml
category:
math: true
---

> 惡補 ML https://www.youtube.com/watch?v=Ye018rCVvOo&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J

## 機器學習基本概念

機器學習的核心是讓機器根據資料學習一個模型（function），讓新的輸入做出相對的預測或決策。

### 分類

- Regression
  - ex: 預測東西
- Classification
  - ex: Alphago
- Structured Learning
  - 創造東西，ex: 生成式?

### 機器學習的 Framework

1. 找 Model ，ex:$y = b + wx$

   - 帶有未知 parameter 的 function 稱為 `Model`
   - $x$ 稱為 `feature`，為 **輸入給模型的資料**
   - $w$ 稱為 `weight`
   - $b$ 稱為 `bias`

2. 定義 `Loss Function`，ex: $L(b, w)$

   - 假設一開始是 $b = 0.5$、$w = 1$，就以 $y = 0.5 + 1x$ 來算出 `每一筆資料的計算結果` 與 `label` 之間的誤差 $e$
     - 實際上正確的 $y$ 稱為 `label`
       計算誤差的方法有很多，像是
       - MAE(Mean Absolute Error): $e = \lvert y - \hat{y} \rvert$
       - MSE(Mean Square Error): $e = (y - \hat{y})^2$
       - Cross-entropy
   - 算好後把每個 $e$ 套入 `Loss Function`
     - ex: $L = \frac{1}{N}\Sigma_{n}e_n$

3. Optimization，找到 $w$ 和 $b$ 能夠讓 `Loss Function` 算出最小的值
   - Gradient Descent
     1. 隨便選一個 $w_0$、$b_0$
     2. 計算斜率，以 $w$ 為例，$\eta \cdot \frac{\partial L}{\partial w}|_{w=w^0, b=b^0}$
        - $\eta$ 是 `learning rate`，需要自己設定
          - 這種要自己設定的值稱為 `hyperparameter`
     3. 更新 $w$ 和 $b$，以 $w$ 為例，$w^1 = w^0 - \eta \cdot \frac{\partial L}{\partial w}|_{w=w^0, b=b^0}$
     4. 持續更新，最後 $L$ 會接近 `local minimum` 或 `global minimum` 附近

實際上在算 Gradient 的時候，不會每次都拿所有訓練資料去算，會把所有訓練資料隨便分成很多 `Batch`，每次都拿一個 `Batch` 算出 Gradient，叫做一次 `update`，如果把所有 `Batch` 都看過一輪，叫做一個 `epoch`

此外，上面的 Model 是定為 $y = b + wx$，但實際上這種 `Linear Model` 可能不足以描述現實情況，也就是 Model 不夠力，稱為 `Model Bias`，這時候可以加入更多 `Feature`，定出一個更複雜的 Model

- Piecewise Linear

![Piecewise Linear](https://optimization.cbe.cornell.edu/images/thumb/9/93/Numerical_example.jpg/680px-Numerical_example.jpg)

現實中可能是 `Continuous Function`，可以用 `Piecewise Linear` 來模擬，`Piecewise Linear` 又可以用很多不同的 `Hard Sigmoid` 來組合出來，而 `Hard Sigmoid` 又可以由 `Sigmoid Function` 模擬出來

![Sigmoid vs Hard Sigmoid](https://www.researchgate.net/publication/368305965/figure/fig4/AS:11431281118383692@1675742756740/Comparison-diagram-of-sigmoid-and-hard-sigmoid.png)

- Sigmoid Function
  $c \cdot sigmoid(b+wx_1)$ = y = c \cdot \frac{1}{1+e^{-b+wx_1}}

`Hard Sigmoid` 也能用兩個 `ReLU (Rectified Linear Unit)` 算出

![ReLU](https://machinelearningmastery.com/wp-content/uploads/2018/10/Line-Plot-of-Rectified-Linear-Activation-for-Negative-and-Positive-Inputs.png)

## 機器學習的任務攻略

![機器學習的任務攻略](https://miro.medium.com/v2/resize:fit:1400/1*KmNACQdl5FvoXvQt5n41nQ.png)

- Model Bias
  模型不給力

- Optimization
  可能是 Optimization 的方式有問題，像是 `Gradient Descent` 不給力

透過比較不同模型的來區分是不是 Model 的問題

![Model Bias v.s. Optimization Issue](https://miro.medium.com/v2/resize:fit:1400/1*Rl5VIRnXCJbvR6e6cTsZcg.png)

像這個例子中， `20-layer` 在 `Traning Data` 上表現的還比 `56-layer` 好，就可以知道是 `Optimization` 的問題。可以先測比較淺的 Model 或是比較簡單的 Model，比較不會有 Optimization 的問題。

- Overfitting
  Traning Data 表現好，Testing Data 表現很爛
  - 增加 Traning Data
  - Data augmentation
    根據理解創造出新的資料，ex: 翻轉圖片、放大圖片等
  - 限制模型，不要讓他彈性那麼大，ex: 減少 `Feature`、減少 `Parameter`、Sharing Parameter、Early Stopping、Regularization、Dropout 等

## 類神經網路訓練不起來

###　卡在 `Critical Point`

- `Local Minimum`
- `Saddle Point`

![Saddle Point](https://media.geeksforgeeks.org/wp-content/uploads/20240829080037/Saddle-Points-02.webp)

### 泰勒展開式

對於 $L(\theta)$，在 $\theta = \theta^\prime$ 可以被近似為：

$$
L(\theta) \approx L(\theta^\prime) + (\theta - \theta^\prime)^T \cdot g + \frac{1}{2}(\theta - \theta^\prime)^T \cdot H \cdot (\theta - \theta^\prime)
$$

- $g$ 為 Gradient Vector，$g = \nabla L(\theta^\prime)$
- $H$ 為 Hessian Matrix，$H = \nabla^2 L(\theta^\prime)$

當走到 `Critical Point` 時，Gradient 會等於 0，所以可以得到：

$$
L(\theta) \approx L(\theta^\prime) + \frac{1}{2}(\theta - \theta^\prime)^T \cdot H \cdot (\theta - \theta^\prime)
$$

因此可以利用 `Hessian Matrix` 來判斷是 `Local Minimum` 、 `Local Maximum` 還是 `Saddle Point`

- 當 $\frac{1}{2}(\theta - \theta^\prime)^T \cdot H \cdot (\theta - \theta^\prime) > 0$ 時，可以知道任何在 $\theta^\prime$ 附近的 $\theta$，$L(\theta) > L(\theta^\prime)$ ，所以 $L(\theta^\prime)$ 是 `Local Minimum`
- 當 $\frac{1}{2}(\theta - \theta^\prime)^T \cdot H \cdot (\theta - \theta^\prime) < 0$ 時，可以知道任何在 $\theta^\prime$ 附近的 $\theta$，$L(\theta) < L(\theta^\prime)$ ，所以 $L(\theta^\prime)$ 是 `Local Maximum`
- 當 $\frac{1}{2}(\theta - \theta^\prime)^T \cdot H \cdot (\theta - \theta^\prime)$ 有時候大於 0 有時候小於 0 時， $\theta^\prime$ 是 `Saddle Point`

基於線性代數的知識，可以知道 `Hessian Matrix` 是 `Symmetric Matrix`，所以可以透過 `Eigenvalue` 來判斷是 `Local Minimum` 、 `Local Maximum` 還是 `Saddle Point`

- 當 `Hessian Matrix` 的 `Eigenvalue` 全部大於 0 時，是 `Local Minimum`
- 當 `Hessian Matrix` 的 `Eigenvalue` 全部小於 0 時，是 `Local Maximum`
- 當 `Hessian Matrix` 的 `Eigenvalue` 有正有負時，是 `Saddle Point`

如果是 `Saddle Point`，可以透過 Hessian Matrix 來判斷出 Loss 更小的方向，然後往那個方向走：

$$
L(\theta) \approx L(\theta^\prime) + \frac{1}{2}(\theta - \theta^\prime)^T \cdot H \cdot (\theta - \theta^\prime)
$$

假設 $u$ 是 `Hessian Matrix` 的 `Eigen vector`，$\lambda$ 是 `Eigen value`，可以得到：

$$
u^T \cdot H \cdot u = u^T \cdot (\lambda \cdot u) = \lambda \lVert u \rVert^2
$$

當 $(\theta - \theta^\prime) = u$ 且 $\lambda < 0$ 時：

$$
\frac{1}{2}(\theta - \theta^\prime)^T \cdot H \cdot (\theta - \theta^\prime) = \frac{1}{2}u^T \cdot H \cdot u = \frac{1}{2}\lambda \lVert u \rVert^2 < 0
$$

可以知道 $L(\theta) < L(\theta^\prime)$。因此如果讓 $\theta^\prime$ 往 $u$ 的方向走，$\theta^\prime + u = \theta$，可以得到更小 Loss 的 $\theta$。

這只是一種解法，在實作上計算量很大，沒有人會這樣做。

此外，實際上 `Local minimum` 並不常見，Loss 下不去常常是卡在 `Saddle Point`。

### 為什麼要用 Batch

- Batch size = N(Full Batch)

  - 一次拿所有資料去算 Gradient
  - 每一次的 Gradient 都很穩定
  - 理論上花的時間比較多，但考慮到平行運算，若以 epoch 為單位，實際上可能更快

- Batch size = 1
  - 一次只拿一筆資料去算 Gradient
  - 每一次的 Gradient 都很不穩定，可能會跳來跳去

![Different Batch Size](./images/machine-learning/diff-bacth-size.png)

既然時間差不多，乍看之下 Batch Size 大一點比較好，但實際上小的 Batch Size 可能會有更好的訓練效果

![Different Batch Size](./images/machine-learning/diff-bacth-size-2.png)

上圖可以看到小的 Batch Size Optimization 的效果會比較好

|                            | Batch Size 小 |       Batch Size 大       |
| :------------------------: | :-----------: | :-----------------------: |
| 每次 update (沒有平行運算) |      快       |            慢             |
|  每次 update (有平行運算)  |     一樣      | 一樣 (在不是超大的情況下) |
|         每個 epoch         |      慢       |            快             |
|     Gradient 的穩定性      |      差       |           穩定            |
|    Optimization 的效果     |      好       |            差             |

### Momentum

- 一般的 Gradient Descent

  - $\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t)$

- Momentum

  - $\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t) + \gamma \cdot (\theta_t - \theta_{t-1})$

    - $\gamma$ 為 `Momentum`，通常設為 0.9

![Momentum](https://i.imgur.com/DdabCqX.png)
