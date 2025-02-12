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

### 步驟

1. 找 Model ，ex:$y = b + wx$

   - 帶有未知 parameter 的 function 稱為 `Model`
   - $x$ 稱為 `feature`
   - $w$ 稱為 `weight`
   - $b$ 稱為 `bias`

2. 定義 `Loss Function`，ex: $L(b, w)$

   - 假設一開始是 $b = 0.5$、$w = 1$，就以 $y = 0.5 + 1x$ 來算出 `每一筆資料的計算結果` 與 `label` 之間的誤差 $e$
     - 實際上正確的 $y$ 稱為 `label`
     - 計算誤差的方法有很多，像是
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
     4. 持續更新，最後 $L$ 可能停在 `local minimum ` 或 `global minimum `

上面的 Model 是定為 $y = b + wx$，但實際上這種 `Linear Model` 可能不足以描述現實情況，也就是 Model 不夠力，稱為 `Model Bias`，這時候就要定一個更複雜的 Model

- Piecewise Linear
  ![Piecewise Linear](https://dr282zn36sxxg.cloudfront.net/datastreams/f-d%3A1100872fa13e5fef130e4074ccb4500cb570c129053fedaebc290981%2BIMAGE_THUMB_POSTCARD_TINY%2BIMAGE_THUMB_POSTCARD_TINY.1)
