---
title: VLSI
date: 2024-10-13 16:22:14
tags: 
math: true
---


Assert-High Swtich: 輸入邏輯 1 電路會 close，導通 
Assert-Low Swtich: 輸入邏輯 0 電路會 close，導通 

# MOSFET
Metal + Oxide-Semiconductor Field Effect Transitor

N channel MOSFET (nFET) 是 assert-high switch，也稱 N-MOS
P channel MOSFET (pFET) 是 assert-low switch，也稱 P-MOS


### 電壓對應 Boolean 值
當電壓大於某個值，一律判斷成邏輯 1，小於某個值判斷成邏輯 0，中間的區域叫做 noise margin。noise margin 越小越容易誤判


### P-N Junctions
p-type 半導體電洞多，n-type 半導體電子多。兩者接在一起會讓電子電洞傾向互相吸引，然而實際上受到電廠影響，僅有少數電子電洞相吸。

P-N Junctions 的特性是，施加外加電壓後，**電流只能從 P 端流向 N 端**。當給予 P 端正電、給予 N 端負電，N 端的電子會受到正電吸引、P 端的電動會受到負電吸引，兩者都往反方向衝，產生 **P 流向 N 的電流**。然而如果我們是給 P 端負電、N 端正電，就不會有電流產生。

## nFET (nMOS)
Source 接地，Drain 給高電壓 ($V_{DD}$)
Source 和 Drain 都接到一個帶很多電子 (n) 的東西，然後整個 body 是 p 型半導體
Strong Logic 0, Weak Logic 1

## pFET (pMOS)
都跟 nFET 相反


## Gate Level

### Not Gate
![not gate](https://i.sstatic.net/DULlo.png)


### 我可以先把 Nand、Nor 之類的 用 pMOS、nMOS 的作法做出，記熟一點
一端是串聯、一端並聯

# Physics Structure
## Add more layers
先加 insulating glass
再 CMP (chemical-mechanical planarization)，磨平
最後加上金屬層

low-k: 電容很快就充滿
critical path: 最長的那條，會用 low-k 來加速

## 絕緣層
t_ox 如果越小，代表絕緣層厚度越小，因此上下的吸引力會越大，導致電容變大
A_g: 面積，w * L
1:15:20 current voltage equation 推導

active contact: metal to drain/source
gate contact : metal to gate 
via: metal to metal

### 會考 IR drop 是啥
### 會考 semantic 轉 layout、layout 轉 semantic
要注意 n-well 的位置
先試試看畫出 inverter
Nor2 或 Nand2，然後要注意 3 個 input 版本的 **並聯** 部分
AOI

### 要會看 stick layout

### 要看的懂 Euler graph (WTFㄑ


# Fabrication
把矽晶柱切成一片一片，然後 polish
產線會需要好幾個禮拜，沒辦法一次做完

#### 為什麼晶圓是圓的
1. 方形的話邊邊角角會撞到，良率 (yeild) 降低
2. 邊邊角角其實不太能用，之後會講

#### 良率 Yeild 
如果做出來完全不能用，直接丟掉
如果做出來不符合規格，可以以更低價錢賣出
為了提升良率，有時候會把導線之間做寬一點，提升容錯率，但會犧牲 Area 或 Power，讓整體面積更大


$Y = e^{-\sqrt{DA}}$
- A: area，晶圓的面積
- D: defect density，根據過往的統計資料


## Silicon Dioxide
使用在 substrate 和 gate 之間


Thermal oxide，加熱讓 Si 和 O2 變成 SiO2，但會很花時間，還會吃掉原本的 Si
Wet oxidation Si + 2H2O -> SiO2 + 2H2，比較快但也很花時間
Deposited on the top (CVD oxide)，省時間，並且是拿另外的 Si 來用，附著在 substrate 上

## Silicon Nitride (氮)
用於 Final protective

## Polysilicon
結晶都是一塊一塊的，所以叫做 Polycrystal silicon 或 polysilicon
加入 Ti、Pt 降低 sheet resistance

## Metal
通常用 Aluminium，附著性很好且很好做 pattern
會有 electromigration (EM) 的問題，也就是 Wire 的老化

### EM
因為電子不斷從同個方向往另一端流，導致電子離開那端的導線產生 voids，越來越細甚至斷掉，充飽所需的電量可能有差
另一端會產生 Hillocks，導線會變大，可能會碰到別的導線導致 short
要控制 Current density J 不能太大，因此導線寬度不能太小

後來大部分改用 Copper，比較不會有 EM 的問題