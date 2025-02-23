---
title: 機器學習筆記
date: 2025-02-22 13:22:11
tags: ml
category:
math: true
---

> 惡補 ML https://www.youtube.com/watch?v=Ye018rCVvOo&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J

# 課程筆記

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
  $$c \cdot sigmoid(b+wx_1) = y = c \cdot \frac{1}{1+e^{-b+wx_1}}$$

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

此外，實際上 `Local minimum` 並不常見，Loss 下不去常常是卡在 `Saddle Point`，但是用一般的 `Gradient Descent` 通常不會卡在 `Critical Point`。

### 為什麼要用 Batch

- Batch size = N(Full Batch)

  - 一次拿所有資料去算 Gradient
  - 每一次的 Gradient 都很穩定
  - 理論上花的時間比較多，但考慮到平行運算，若以 epoch 為單位，實際上可能更快

- Batch size = 1
  - 一次只拿一筆資料去算 Gradient
  - 每一次的 Gradient 都很不穩定，可能會跳來跳去

![Different Batch Size](./images/machine-learning/diff-batch-size.png)

既然時間差不多，乍看之下 Batch Size 大一點比較好，但實際上小的 Batch Size 可能會有更好的訓練效果

![Different Batch Size](./images/machine-learning/diff-batch-size-2.png)

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

  $$
  \begin{aligned}
  \texttt{Start at} \quad \theta^0 \newline
  \texttt{Movement} \quad m^0 &= 0 \newline
  \texttt{Compute gradient} \quad g^0 \newline
  \texttt{Movement} \quad m^1 &= \lambda \cdot m^0 - \eta \cdot g^0 \newline
  \texttt{Move to} \quad \theta^1 &= \theta^0 + m^1 \newline
  \texttt{Compute gradient} \quad g^1 \newline
  \texttt{Movement} \quad m^2 &= \lambda \cdot m^1 - \eta \cdot g^1 \newline
  \texttt{Move to} \quad \theta^2 &= \theta^1 + m^2 \newline
  \end{aligned}
  $$

  $m^i$ 就像是所有過去 `Weighted Gradient` 的總和，$g^0, g^1, \cdots g^{i-1}$

  $$
  \begin{aligned}
  m^0 &= 0 \newline
  m^1 &= \lambda \cdot m^0 - \eta \cdot g^0 \newline
  &= -\eta \cdot g^0 \newline
  m^2 &= \lambda \cdot m^1 - \eta \cdot g^1 \newline
  &= \lambda \cdot (-\eta \cdot g^0) - \eta \cdot g^1
  \end{aligned}
  $$

  ![Momentum](https://i.imgur.com/DdabCqX.png)

### Learning Rate

#### 一般情況下的 Learning Rate 造成的問題

![Learning Rate](./images/machine-learning/LearningRate.png)

- 當 `Learning Rate` 設定太大時，可能會造成 `Oscillation` 的問題，Loss 會一直在上下跳動，無法收斂
- 當 `Learning Rate` 設定太小時，可能會造成 `Convergence` 的問題，Loss 會一直往下收斂，但是收斂的速度很慢。就像上圖一樣，當 `Gradient` 很大時，沒什麼問題，但是當 `Gradient` 很小時，就會卡住

上述例子中，每個參數的 `Learning Rate` 都是一樣的，但是實際上每個參數的 `Gradient` 都不一樣，應該要為每個參數的 `Learning Rate` 客製化。

#### AdaGrad (Adaptive Gradient)

用 `Root Mean Square` 來調整 `Learning Rate`

$$
\begin{aligned}
\theta^1_i \leftarrow \theta^0_i - \frac{\eta}{\sigma^0_i} \cdot g^0_i &\quad \sigma^0_i = \sqrt{(g^0_i)^2} = \lvert g^0_i \rvert \newline
\theta^2_i \leftarrow \theta^1_i - \frac{\eta}{\sigma^1_i} \cdot g^1_i &\quad \sigma^1_i = \sqrt{\frac{1}{2} \cdot [(g^0_i)^2 + (g^1_i)^2]} \newline
\theta^3_i \leftarrow \theta^2_i - \frac{\eta}{\sigma^2_i} \cdot g^2_i &\quad \sigma^2_i = \sqrt{\frac{1}{3} \cdot [(g^0_i)^2 + (g^1_i)^2 + (g^2_i)^2]} \newline
\vdots \newline
\theta^{t+1}_i \leftarrow \theta^t_i - \frac{\eta}{\sigma^t_i} \cdot g^t_i &\quad \sigma^t_i = \sqrt{\frac{1}{t} \cdot \sum^t_k (g^k_i)^2}
\end{aligned}
$$

從公式可以觀察到，當坡度小的時候，`Gradient` 會比較小，算出來的 $\sigma$ 也會比較小，所以 `Learning Rate` 會比較大，反之亦然。

然而，當 `t` 很大時，當前的 `Gradient` 可能會被過去累積的 `Gradient` 稀釋掉，導致收斂速度變慢，不能實時考慮梯度的變化情況。

#### RMSprop

`RMSprop` 增加了 `Decay Rate` $\alpha$，可以控制 **當前的 `Gradient`** 和 **過去累積的 `Gradient`** 的重要程度

$$
\begin{aligned}
\theta^1_i \leftarrow \theta^0_i - \frac{\eta}{\sigma^0_i} \cdot g^0_i &\quad \sigma^0_i = \sqrt{(g^0_i)^2} = \lvert g^0_i \rvert \newline
\theta^2_i \leftarrow \theta^1_i - \frac{\eta}{\sigma^1_i} \cdot g^1_i &\quad \sigma^1_i = \sqrt{\alpha \cdot (\sigma^0_i)^2 + (1 - \alpha) \cdot (g^1_i)^2} \newline
\theta^3_i \leftarrow \theta^2_i - \frac{\eta}{\sigma^2_i} \cdot g^2_i &\quad \sigma^2_i = \sqrt{\alpha \cdot (\sigma^1_i)^2 + (1 - \alpha) \cdot (g^2_i)^2} \newline
\vdots \newline
\theta^{t+1}_i \leftarrow \theta^t_i - \frac{\eta}{\sigma^t_i} \cdot g^t_i &\quad \sigma^t_i = \sqrt{\alpha \cdot (\sigma^{t-1}_i)^2 + (1 - \alpha) \cdot (g^t_i)^2}
\end{aligned}
$$

- $\alpha$ 為 `Decay Rate`，通常設為 0.9

最常用的 `Optimizer` 是 `Adam`，他就是 `RMSprop` 和 `Momentum` 的結合

#### Learning Rate Scheduling

- Learning Rate Decay

  隨著參數的更新，`Learning Rate` 逐漸變小

  ![Learning Rate Decay](https://miro.medium.com/v2/resize:fit:1400/1*iFCd4c6Bq8vQgFHpxTXFUA.png)

  左邊是 `AdaGrad`，當縱軸的 `Gradient` 一直都是很小的值時，會導致 $\sigma$ 變得很小，造成 `Learning Rate` 放的很大，因此會飛出去。右邊是加上 `Learning Rate Decay`。

- Warm Up

  在一開始的時候，`Learning Rate` 會比較小，然後逐漸變大，最後再變小

  ![Warm Up](./images/machine-learning/WarmUp.png)

### Loss Function

`Loss Function` 也會影響到 `Optimization` 的效果，這邊以分類問題為例。

#### Classification

在 Classification 的問題中，通常會用 one-hot encoding 來表示 label，例如：

$$
\begin{aligned}
\hat{y} &= [1, 0, 0] \newline
\hat{y} &= [0, 1, 0] \newline
\hat{y} &= [0, 0, 1] \newline
\end{aligned}
$$

最後在 Output 的時候，通常會把輸出 `y` 通過 `Softmax` 函數，再讓他和 `one-hot encoding` 的 `label` 做比較，計算出 `Loss`

$$
\begin{aligned}
\text{Softmax}(z)_i &= \frac{e^{z_i}}{\sum^n_j e^{z_j}} \newline
\hat{y} \leftrightarrow y^\prime &= \text{Softmax}(y) \newline
\end{aligned}
$$

- 這裡的 `y` 稱為 `logit`
- `Softmax` 可以把 `Output` 變成 `Probability`，讓 `Output` 的值在 0 到 1 之間，並且總和為 1
- 當只有兩個 Class 時，`Softmax` 和 `Sigmoid` 的作用是一樣的

#### Loss Function of Classification

可以直接用 `MSE`，但是 `Cross-entropy` 通常表現的比較好

- `Cross-entropy`
  $$L = -\sum_{i=1}^{n} \hat{y_i} \cdot \ln(\hat{y}_i)$$
  - Minimize `Cross-entropy` 就是最大化 `Likelihood`

![Cross-entropy](https://miro.medium.com/v2/resize:fit:1400/1*nvX_2FTKK6-e2L-XlxZMrw.png)

上圖可以看到，在 `Classfication` 問題用 `MSE` 可能會 train 不起來

#### Batch Normalization

有時候在 `Error Surface` 中，不同維度的輸入值可能差很多，導致 `Error Surface` 很扭曲，斜率、坡度都不同

![Error Surface](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*XSpgDZ9r7FG6vlE9Rv9kCA.png)

對於一個 `Batch` 的資料，對每一個 `Feature` 做 `Feature Normalization`，讓他的 `Mean` 為 0，`Variance` 為 1，稱為 `Batch Normalization`

$$
\begin{aligned}
\mu &= \frac{1}{m} \sum_{i=1}^{m} x_i \newline
\sigma^2 &= \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2 \newline
\hat{x}_i &= \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \newline
\end{aligned}
$$

### Regularization

最常使用 `L1` 和 `L2` Regularization，在 `Loss Function` 中加上 `Regularization Term`，讓 `Model` 不要太複雜

公式來源可以參考 Lagrange Multiplier。

#### L1 Regularization

$$
Loss\_Fn = original\_Loss\_Fn + \lambda \cdot \sum_{i=1}^{n} \lvert w_i \rvert
$$

#### L2 Regularization

$$
Loss\_Fn = original\_Loss\_Fn + \lambda \cdot \sum_{i=1}^{n} w_i^2
$$

![Regularization](./images/machine-learning/Regularization.png)

#### 參考資料

- [L1 , L2 Regularization 到底正則化了什麼 ?](https://allen108108.github.io/blog/2019/10/22/L1%20,%20L2%20Regularization%20%E5%88%B0%E5%BA%95%E6%AD%A3%E5%89%87%E5%8C%96%E4%BA%86%E4%BB%80%E9%BA%BC%20_/)

## Convolutional Neural Network (CNN)

如果用 `Fully Connected Network` 的方式來做圖片的分類，會有很多參數，雖然可以增加 Model 的彈性，但也會增加 `Overfitting` 的風險

![Fully Connected Network](./images/machine-learning/CNN-1.png)

像上圖這個例子，圖片大小是 `100 x 100`，算上 `RGB` 三個 Channel，就有 `100 x 100 x 3` 個 `Feature`，第一層有 `1000` 個 `Neuron`，每個 `Neuron` 對於這 `100 x 100 x 3` 個 `Feature` 都有一個 `Weight`，所以總共有 `100 x 100 x 3 x 1000` 個 `Weight`

### Receptive Field

然而對於圖片辨識來說，他只在乎圖片有沒有重要的 `Pattern`，因此這些 `Neuron` 其實不用把整張圖片當作輸入，只要關心自己的 `Receptive Field` 就好

![Receptive Field](./images/machine-learning/ReceptiveField.png)

典型的設置方式是像下圖

- 會檢查所有 `Channel`
- `Kernel Size`: `3 x 3`
- `Stride`: 通常是 `1` 或 `2`，避免有些 `Pattern` 被忽略
- 超出去的部分要補 `Padding`
- 每個 `Receptive Field` 會有一組 `Neuron` 看著

![CNN Typical Setting](./images/machine-learning/CNNTypicalSetting.png)

雖然 `Kernel Size` 只有 `3 x 3`，但當 Model 疊的越深，每個 `Receptive Field` 就會看到更大的 `Pattern`，不用擔心太大的 `Pattern` 偵測不到

### Shared Parameter

有時候同樣的 `Pattern` 會在不同圖片的不同位置出現，這些 `Neuron` 做的事情其實是一樣的

![Same Pattern Different Regions](./images/machine-learning/SamePatternDifferentRegions.png)

這時候可以用 `Shared Parameter` 來解決，讓不同的 `Receptive Field` 的不同 `Neuron` 用同樣的 `Weight`，減少參數。(在實作上，其實就是一個 `Filter` 掃過整張圖片)

![Shared Patameters](./images/machine-learning/SharedPatameters.png)

`Fully Connected Network` 很彈性，可以做各式各樣的事情，但可能沒辦法在任何特定的任務上做好。`CNN` 則是專注在圖片辨識上，即使 `Model Bias` 比較大，比較不會 `Overfitting`

![CNN Benefit](./images/machine-learning/CNNBenefit.png)

### Pooling

有時候為了減少運算量，會用 `Pooling` 來做 `Subsampling`，通常是 `Max Pooling` 或 `Average Pooling`

![Max Pooling & Average Pooling](https://www.researchgate.net/publication/333593451/figure/fig2/AS:765890261966848@1559613876098/llustration-of-Max-Pooling-and-Average-Pooling-Figure-2-above-shows-an-example-of-max.png)

一般都是在 `Convolutional Layer` 後面接 `Pooling Layer`，交替使用

![Max Pooling](./images/machine-learning/MaxPooling.png)

不過 `Pooling` 可能會造成 `Information Loss`，有些比較細微的特徵會偵測不到，因此也有人從頭到尾都只用 `Convolution`，像是 `AlphaGo`

### CNN Structure

在 `CNN` 中 `Convolutional Layer` 和 `Pooling Layer` 的最後，`Flatten` 過後會再接幾層 `Fully Connected Layer`，再接一個 `Softmax` 來做分類

![CNN Structure](./images/machine-learning/CNNStructure.png)

## Self-Attention

先前提到的 `Input` 都只是一個 `Vector`，然而很多時候，模型吃的是 **一組 `Vector`**，又稱 `Vector Set`、`Sequence`，又可以分成三類

- 每個 `Vector` 有一個 `Label`，輸入的數量等於輸出的數量，稱為 `Sequence Labeling`
  - ex: `Pos-Tagging`
- 整個 `Sequence` 只有一個 `Label`
  - ex: `Sentiment Analysis`
- 機器自己決定要有幾個 `Label`
  - ex: `Sequence-to-Sequence`、`Machine Translation`

對於 `Sequence Labeling`，如果像前面提到的 `CNN` 一樣，每個 `Vector` 都是獨立的，可能會忽略掉 `Vector` 之間的關係 (Context)。你也可以把整個 `Sequence` 丟到 `CNN` 裡面，但參數量、計算量都會超大，又久又容易 `Overfitting`，因此有了 `Self-Attention`

![https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html](https://sebastianraschka.com/images/blog/2023/self-attention-from-scratch/summary.png)

`Self-Attention` 的概念是，對於每個 `Vector`，都會有一個 `Query`、`Key`、`Value`，然後透過 `Query` 和 `Key` 的 `Dot Product` 來計算 `Attention Score`，再透過 `Softmax` 來計算 `Attention Weight`，最後再把 `Value` 乘上 `Attention Weight` 來得到 `Output`

- `Softmax` 是最常見的，不過也可以用別的 `Activation Function`
- `Attention Weight` 會讓 `Model` 知道哪些 `Vector` 是重要的，哪些是不重要的

### Multi-Head Self-Attention

相關這件事情可能有很多種形式，為了要找到資料中不同種類的相關性，可以用 `Multi-Head Self-Attention`

- Head 的數量也是 `Hyperparameter`

![https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff406d55e-990a-4e3b-be82-d966eb74a3e7_1766x1154.png)

### Positional Encoding

`Self-Attention` 並不會考慮到 `Position`，因此需要加上 `Positional Encoding`，讓 `Model` 知道 `Vector` 的位置

![Positional Encoding](https://i.imgur.com/QYpn2J3.png)

### Truncated Self-Attention

有時候 `Sequence` 會超長，造成 `Attention Matrix` 太大，計算量太大，甚至 Train 不起來，因此可以用 `Truncated Self-Attention`，只考慮某距離以內的 `Vector`，不考慮太遠的 `Vector`

![Truncated Self-Attention](https://lh5.googleusercontent.com/E-BGMJXwdWbYORmuf9OFvIDZ9ciriH88oWI8otaIjJDNuCyYvdFMtAeR7HqbbhK_WwHSLhMGyr77wBh7W1_kB1AQ9XAdouBsONFfqWltGXxlqtclXC7uRUU5NUxwFU80JHhIHDE)

### Self-Attention vs CNN

一般圖片都是用 `CNN` 來處理，但其實 `CNN` 是 `Self-Attention` 的一種，只是 `CNN` 會考慮到 `Local Pattern`，而 `Self-Attention` 會考慮到 `Global Pattern`

![Self-Attention vs CNN](https://miro.medium.com/v2/resize:fit:1400/1*IRvXQeATmX0JxJxz45_XUA.png)

`Self-Attention` 就是一種更彈性的 `CNN`，因此在訓練資料很大的時候，`Self-Attention` 可能比 `CNN` 更好

![Self-Attention vs CNN](https://miro.medium.com/v2/resize:fit:1400/1*2xKkjuDVe8zTUa6lVf3QGw.png)

### Self-Attention vs RNN

現在 `RNN` 幾乎被 `Self-Attention` 取代，因為 `RNN` 有兩大缺陷

- `Long-Term Dependency` 的問題，當 `Sequence` 很長時，很容易忘記越早輸入進來的資料
- 只能 `Sequential` 計算，無法平行運算

![Self-Attention vs RNN](https://miro.medium.com/v2/resize:fit:1400/1*qATp1B0W4BK0J4IL-sIsig.png)

# PyTorch

- PyTorch 是一個開源的機器學習框架
  - 可以做 `Tensor Computation`，像是 `Numpy`，但可以用 `GPU` 加速
  - 可以幫你算 `Gradient`

#### Tensor

- Tensor: 高維度的矩陣或陣列
- 常用的 `Data Type` 有 `torch.float`、`torch.long`、`torch.FloatTensor`、`torch.LongTensor` 等
- PyTorch 的 `dim` 跟 `Numpy` 的 `axis` 一樣

##### Example

- Constructor

  ```python
  x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
  # tensor([1., 2., 3., 4.])

  x = torch.from_numpy(np.array([1, 2, 3, 4]))

  x = toch.zeros(2, 3)
  # tensor([[0., 0., 0.],
  #         [0., 0., 0.]])

  x = torch.ones(2, 3)
  # tensor([[1., 1., 1.],
  #         [1., 1., 1.]])
  ```

- Operators

  - `squeeze`
    ```python
    x = torch.zeros([1, 2, 3])
    print(x.shape)
    # torch.Size([1, 2, 3])
    x = x.squeeze(0) # 把第 0 維度的 1 拿掉
    print(x.shape)
    # torch.Size([2, 3])
    ```

  ````
  - `unsqueeze`
    ```python
    x = torch.zeros([2, 3])
    print(x.shape)
    # torch.Size([2, 3])
    x = x.unsqueeze(1) # 在第 1 維度插入 1
    print(x.shape)
    # torch.Size([2, 1, 3])
  ````

  - `cat`
    ```python
    x = torch.zeros([2, 1, 3])
    y = torch.zeros([2, 3, 3])
    z = torch.zeros([2, 2, 3])
    w = torch.cat([x, y, z], dim=1) # 合併第 1 維度
    print(w.shape)
    # torch.Size([2, 6, 3])
    ```
  - `pow`
    ```python
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    x = x.pow(2)
    # tensor([ 1.,  4.,  9., 16.])
    ```
  - `sum`
    ```python
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    x = x.sum()
    # tensor(10.)
    ```
  - `mean`
    ```python
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    x = x.mean()
    # tensor(2.5000)
    ```

##### vs Numpy

|        PyTorch         |        Numpy         |
| :--------------------: | :------------------: |
|        x.shape         |       x.shape        |
|        x.dtype         |       x.dtype        |
| x.reshape() / x.view() |     x.reshape()      |
|      x.squeeze()       |     x.squeeze()      |
|     x.unsqueeze(1)     | np.expand_dims(x, 1) |

##### Device

可以用 `toruch.cuda.is_available()` 檢查有沒有 `GPU`，然後用 `torch.cuda.device_count()` 來看有幾個 `GPU`

```python
x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
x = x.to('cuda') # 把 tensor 放到 GPU 上計算
```

## DNN 的架構

![PyTorch DNN](./images/machine-learning/PytorchDNN.png)

### Gradient

```python
x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True) # 設定 requires_grad=True 會記錄 Gradient
z = x.pow(2).sum()
z.backward() # 計算 Gradient
print(x.grad) # tensor([[ 2.,  0.],
              #         [-2.,  2.]])
```

### Dataset & DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    # Read data and preprocess
    def __init__(self):
        self.data = ...

    # Returns one sample at a time
    def __getitem__(self, idx):
        return self.data[idx]

    # Returns the size of the dataset
    def __len__(self):
        return len(self.data)

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

![Dataset & DataLoader](./images/machine-learning/DatasetDataLoader.png)

### Neural Network Layers

Linear Layer (Fully-connected Layer)

```python
layer = torch.nn.Linear(32, 64) # 32 -> 64
print(layer.weight.shape)
# torch.Size([64, 32])
print(layer.bias.shape)
# torch.Size([64])
```

### Activation Function

```python
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax()
```

### Loss Function

```python
nn.MSELoss()
nn.CrossEntropyLoss()
```

### Optimizer

```python
torch.optim.SGD(model.parameters(), lr, momentum) # Stoachastic Gradient Descent
torch.optim.Adam(model.parameters(), lr) # Adam
```

### Build Model

```python
import torch.nn as nn

class MyModel(nn.Module):
    # Initialize the model & define the layers
    def __init__(self):
        super(MyModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(10, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )

    # Compute the output of NN
    def forward(self, x):
        return self.nn(x)
```

### Training

```python
dataset = MyDataset(filename)
training_set = DataLoader(dataset, batch_size=16, shuffle=True)
model = MyModel().to('cuda')
critertion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train() # Set the model to training mode
    for x, y in training_set:
        x, y = x.to('cuda'), y.to('cuda') # Move the data to GPU
        optimizer.zero_grad() # Clear the gradient
        y_pred = model(x) # Forward pass
        loss = criterion(y_pred, y) # Compute the loss
        loss.backward() # Compute the gradient
        optimizer.step() # Update the parameters
```

### Evaluation (Validation)

```python
model.eval() # Set the model to evaluation mode
total_loss = 0
for x, y in validation_set:
    x, y = x.to('cuda'), y.to('cuda')
    with torch.no_grad(): # Disable gradient computation
        y_pred = model(x)
        loss = criterion(y_pred, y)

    total_loss += loss.cpu().item() * x.size(0) # Accumulate the loss
    avg_loss = total_loss / len(validation_set.dataset) # Compute the average loss
```

### Evaluation (Testing)

```python
model.eval() # Set the model to evaluation mode
predictions = []
for x in test_set:
    x = x.to('cuda')
    with torch.no_grad(): # Disable gradient computation
        y_pred = model(x)
        predictions.append(y_pred.cpu())
```

### Save & Load Model

```python
torch.save(model.state_dict(), path) # Save the model

checkpoint = torch.load(path) # Load the model
model.load_state_dict(checkpoint)
```

## 可重現性 (Reproducibility)

為了讓每次執行的結果都保持一樣，不然很難確定效果改進是因為調整 `hyperparameter`，還是來自隨機性的變異。

```python
myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
```

- `myseed = 42069`: 設定固定的 `seed`
- `torch.backends.cudnn.deterministic = True`: `cuDNN` 為了加速計算，會默認選擇最快的方法，可能導致 non-deterministic 的行為發生，改成 `True` 可以確保每次結果都一樣
- `torch.backends.cudnn.benchmark = False`: 禁止 `cuDNN` 自己選擇最佳的計算方法，使用固定的方法來保證每次計算方式相同

如果為了 Performance 而不在乎 Reproducibility，可以把上面設成相反的
