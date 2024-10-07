---
title: Graphcis Programming and Application
date: 2024-09-08 13:28:01
tags: 
category: 
math: true
---

## OpenGL
- Application Programming Interface (API)
- 寫指令與底層顯卡溝通，把資料餵到 Render Pipeline 裡面，繪出結果
- Primitive-based Rendering
  - 用最小單元，點、線、面組成

## Rasterization Rendering Pipeline
- 有很多 Stage，一路從 CPU 流到 GPU，有些專門處理頂點，有些專門處理三角片，有些處理 Pixel...
- 早期的 Rendering Pipeline 是 Fiexed Function，Stage 裡面大部分的邏輯是寫死的
  - 只能傳一些自己的參數進去，不能改變裡面的運行邏輯
- 現在是 Programmable shader，可以自己寫一些東西進去，更為靈活
  - 在 OpenGL 3.0 成為正式的核心，捨去 Fixed Function Pipeline
  - 在 OpenGL 裡面使用 GLSL 撰寫 
![Rasterization Rendering Pipeline](https://www.khronos.org/opengl/wiki_opengl/images/RenderingPipeline.png)

# Projection & Transformation
- Modeling Transformation
World Coordinate
- Viewing Transformation
Camera Coordinate
- Projection Transformation
Window Coordinate
- Viewport Transformation
Screen Coordinate

## Coordinate System
![Coordinate System](https://i.sstatic.net/w7bKr.jpg)


### Homogeneous Coordinates
- 用四維 vector (x, y, z, w) 來描述，能夠區分它是 **點** 還是 **向量**
  - w = 0: 向量
  - w = 1: 點
