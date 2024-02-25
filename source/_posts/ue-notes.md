---
title: Unreal Engine Notes
date: 2024-02-24 21:45:31
tags: UE5
categories: ['UE5']
---

### Texture Object 
- 有點像一個單獨的 Texture 的容器，包含很多資料，像是顏色、光澤度、粗糙度
  
### Texture Sample 
- 把 Texture Object 做 sample，得到某個數據，像是顏色
- 要傳入 UVs 和 Tex，這裡的 Tex 就是 Texture Object

### Dynamic Material Instances
- 可以在遊戲中動態改變的 Material
- 如何使用
  - [UE5 Blueprint Tutorial - How to Create Dynamic Material Instances](https://www.youtube.com/watch?v=uSKzkg0dQpY&ab_channel=UnrealDevHub)
  - [Unreal Engine 4 Materials Tutorial——虚幻4引擎教程——材质](https://zhuanlan.zhihu.com/p/377411777) 

### Set Scalar Parameter Value
- 有三個輸入，Target、Parameter Name 和 Value
- 改變 Target 的 Parameter 的值
