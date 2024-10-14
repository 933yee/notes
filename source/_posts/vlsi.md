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


# 我可以先把 Nand、Nor 之類的 用 pMOS、nMOS 的作法做出，記熟一點
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

# 會考 IR drop 是啥
# 會考 semantic 轉 layout、layout 轉 semantic
要注意 n-well 的位置
先試試看畫出 inverter
Nor2 或 Nand2，然後要注意 3 個 input 版本的 **並聯** 部分
AOI

# 要換看 stick layout

# 要看的懂 Euler graph (幹