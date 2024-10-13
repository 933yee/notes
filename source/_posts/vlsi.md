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