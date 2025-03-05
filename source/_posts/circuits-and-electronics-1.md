---
title: Circuits & Electronics - Semiconductor Material & Diode
date: 2024-09-07 22:54:32
tags:
category:
math: true
---

# 半導體 Semiconductor

- 自由電子數量介於 **導體(Conductor)** 與 **絕緣體(Insulator)** 之間

### 導體 Conductor

- 材料上有許多自由電子 (free electron)
- 給予外加電壓後，產生電場，自由電子開始移動
- 自由電子移動，且數量夠多時，量測的到電流，我們稱它為導體
- 在室溫下，自由電子濃度大概是 $10^22 / cm^3$

### 絕緣體 (Insulator)

- 每個原子幾乎都沒辦法貢獻自由電子，因此自由電子數量極少，沒辦法導電
- 在室溫下，自由電子濃度大概是 $10^1 / cm^3$

## 半導體有哪些種類？

- 通常在元素週期表上的第四族，像是矽(Si)、鍺(Ge) 又稱為 **元素半導體 (elemental semi)**
- 有時候會是一個 **第三族** 和 **第五族** 元素混合形成化合物，像是砷化鎵 (GaAs)，又稱 **化合物半導體 (compound semi)**

#### 矽 Si

- 礦產佔地球比例 1/4
- 跟 Ge 比起來比較穩定

## 本質半導體 (Intrinsic Semi)

- 第四族的所有元素，**價電子 (valence electron)** 數量都是 4 個
  - 價電子：原子最外的電子層中的電子
  - 其中 C 是絕緣體，Si、Ge 是半導體，Sn、Pb 是導體
- Si 會與鄰近的其他 Si 形成 **共價鍵 (covalence bond)**
  - 在室溫下，自由電子濃度大概是 $10^10 / cm^3$
  - 外界給予能量時 (ex: 溫度上升)，價電子會脫離共價鍵
    - 價電子脫離共價鍵結構所需的最小能量稱為 **Bandgap Energy (Eg)**
    - $e^-$ 脫離後會產生電洞 $h^+$，又稱為 Generate $e^-$ $h^+$ 對，與之相反的是 Recomine
  - 在溫度上升的過程中，Generation 速率會比 Recombination 速率快
  - 當熱平衡 (Thermal Equilibrium) 時，Generation 速率跟 Recombination 速率一樣快，$e^-$、$h^+$ 數量不再改變

### 電子濃度公式

- $ n_i(T) = BT^{3/2} e^{(-Eg/2kT)}$
  - i: intrinsic
  - B: Semi Constant for Si, B = 5.23 \* $10^{15}$
  - Eg: Bandgap energy (eV) for Si, Eg = 1.1 eV
  - T: 絕對溫度 (K), K = 273.15 + $^\circ C$
  - k: Boltzmann's Constant, k = 86 \* $10^{-6}$ eV/K
- 可以發現本質載子濃度對於溫度 T 非常敏感

## 外質半導體 (Extrinsic Semi)

- 摻少量 (比例大約 1/10000000) 雜質到本質半導體 (Si)，導電性上升

### N 型半導體

- 這邊以摻雜五價元素為例，稱為 **N 型半導體**
  - 原先 Si 濃度是 5 \* $10^22$ 原子/$cm^3$
  - 摻雜的 P 濃度是 5 \* $10^15$ 原子/$cn^3$
  - 摻雜過後的 Si 濃度是 5 _ $10^22$ - 5 _ $10^15$，還是 5 \* $10^22$ 原子/$cm^3$
    - 可以發現 Si 濃度幾乎沒有改變，因此熔點等物理特性、化學特性不會變，但是 **導電性** 會改變
- 摻雜元素要是均勻分布的
- 摻雜後，每個雜質都多貢獻一個自由電子，稱其為 **施體或施子 (donor)**
  - 原先電子濃度是 $10^10$
  - 摻雜後的電子濃度是 $10^10$ + $10^15$ = $10^15$，因此電子濃度是由 donor 濃度決定，稱其為 $N_d$

### P 型半導體

- 摻雜三價元素 (B)，原理跟 N 型半導體 差不多
- 電洞變成多數載子 (Majority Carriers)，電子變成少數載子 (Minority Carriers)
- 上述的 donor 變成 acceptor (受體、受子)，因此 $N_d$ 也變成 $N_a$

### 質量作用定律 (mass-action law)

- $ n p = n_i^2$
  - n: 電子濃度
  - p: 電洞濃度
  - n_i: 本質載子濃度
- 電子濃度和電動濃度乘積固定，當電子濃度上升後，電洞濃度會下降
- 在 N 型半導體裡面，n 替換成 $n_{n0}$，p 替換成 $p_{n0}$
- 在 P 型半導體裡面，n 替換成 $n_{p0}$，p 替換成 $p_{p0}$

## 飄移與擴散 Drift and Diffusion

### 飄移

- 外力，像是施予電壓產生電場
- 電場產生力給載子，移動裡面的電子與電洞
  - 雖然電子與電洞方向不同，但他們貢獻的方向是相同的，不會抵銷
  - 所有載子的平均速度，就是 **飄移速度**，會隨著電壓變大而變快
    - 當飄移速度變快時，電流也會越大
    - 可以發現跟歐姆定律很像，但是在半導體中要多考慮 **電洞**
- 雖然電場與飄移速度成正相關，但是當電場大到一定程度 ($10^4$)，飄移速度幾乎不會再增加
  - 電子: $v_{dn}$ = - $\mu_n$ E
  - 電洞: $v_{dp}$ = + $\mu_p$ E
    - $\mu$: 移動率，mobility
    - $v_d$: 飄移速度
    - E: 電場
    - 電子多個負號，因為它與電場方向相反
    - $\mu_n$ 大約是 1350，$\mu_p$ 大約是 480 $\frac{cm^3}{V \cdot s}$

### 半導體電流公式

![Semiconductor Current Equation](./images/circuits-and-electronics-1/SemiconductorCurrentEquation.png)

#### 電子 n

我們有 $I_n$ = $\frac{N \cdot q}{t}$ 、 $V_{dn}$ = $\frac{-L}{t}$ (- 為方向)，因此 $I_n = qN \cdot \frac{(-V_{dn})}{L}$
電流密度 $J_n = \frac{I_n}{A} = \frac{qN \dot (-V_{dn})}{L\cdot A}$ = $q \cdot n (-V_{dn})$ (n 為電子濃度)
結合前面的飄移速度公式，可以得到 $J_n = q \cdot n \cdot (-V_{dn}) =  q \cdot n \cdot (-) (-\mu_n) \cdot E = q \cdot n \mu_n \cdot E$

#### 電洞 p

基本上跟電子類似，可以得到 $J_p = q \cdot p \mu_p \cdot E$

可以知道 **半導體的電流密度** $J = J_n + J_p = (q \cdot n \mu_n + q \cdot p \mu_p) \cdot E$

#### 導電係數 $\sigma$

半導體的電流密度 中的 $(q \cdot n \mu_n + q \cdot p \mu_p)$ 又稱為 **導電係數 conductivity**

在探討 **歐姆定律 (Ohm's laws)** 時，我們有 $I = \frac{V}{R} = \frac{V \cdot A}{\rho \cdot L} = J \cdot A$

- $\rho$ 為 **電阻係數**
- $L$ 為長度
- $A$ 為截面積

可以發現 $J = \frac{1}{\rho} \cdot \frac{V}{L} = (\frac{1}{\rho}) \cdot E$ 且 $J = \sigma \cdot E$
所以 $\sigma = \frac{1}{\rho}$

##### 例題

本質半導體 Si 在 300 K 的溫度下， $\mu_n = 1350 \frac{cm^3}{V \cdot s}$，$\mu_p = 480 \frac{cm^3}{V \cdot s}$，$N_d = 1 \cdot 10^{16} 1/cm^3$，求導電係數 $\sigma$？

$n = N_d = 1 \cdot 10^{16}$
$p = \frac{n_i^2}{N_d} = \frac{(1.5 \cdot 10^{10})^2}{10^{16}} = 2.25 \cdot 10^4$
可以發現 $q \cdot p \cdot \mu_p$ 跟 $q \cdot n \cdot \mu_n$ 比起來極小，可以省略
因此 $J = q \cdot n \cdot \mu_n = 1.6 \cdot 10^{-19} \cdot 10^{16} \cdot 1350 = 2.16$

- 可以觀察到，如果沒有參雜雜質，$p$ 或 $n$ 都遠不及有參雜雜質的，因此 **導電性** 會差很多倍

### 導電係數之溫度係數

![Impurity Scattering & Lattice Scattering](https://i.sstatic.net/BO3H1.png)

#### Impurity Scattering

溫度上升，電子能有更多能量，移動速度較快，電子移動過程中更難受質子影響 (吸引)
因此，$T$ 上升，$\mu$ 上升

#### Lattice Scattering

溫度上升，離子更容易震動，離子的 **有效面積** 上升，電子移動過程中更容易撞到離子
因此，$T$ 上升，$\mu$ 下降

一般而言，我們討論的室溫 (300 K)，大概都在 **Lattice Scattering** 的範圍內

##### 本質半導體

在本質半導體中，$\sigma = (q \cdot n \mu_n + q \cdot p \mu_p)$，其中 $n = p = n_i$，因此 $\sigma = q \cdot n_i \cdot (\mu_n + p \mu_p)$
前面有提到，**本質載子濃度** $n_i$ 對於溫度非常敏感，因此當溫度上升時，雖然 $(\mu_n + p \mu_p)$ 下降，但是遠不及 $n_i$ 的上升程度，所以 $\sigma$ 還是上升

##### N 型半島體

$\sigma = q \cdot N_d \cdot \mu_n$
溫度上升，$N_d$ 不變，$\mu_n$ 下降，所以 $\sigma$ 下降，P 型半導體同理

### 擴散

- 透過濃度的差異，自然的流動，不需要外力
