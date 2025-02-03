---
title: React DOM
date: 2025-02-03 22:15:37
tags: [react, DOM, frontend, website]
category:
---

> 從來沒去搞懂 react 的一些機制

## 瀏覽器怎麼渲染網頁

瀏覽器會遵照一些既有的規則，去解析我們的 HTML 和 CSS 檔案，分別做成 DOM 和 CSSOM 等樹狀結構，最後再合併成 Render Tree。

![Web Browser Rendering](https://www.lumin.tech/articles/browser-reflow-repaint/reflow_repaint.png)

#### Reflow

當 DOM 結構、元素的大小、位置或其他與 layout 的屬性改變時，瀏覽器會重新計算所有元素的位置

#### Repaint

當 DOM 變更影響了元素的外觀（顏色、背景、陰影等），瀏覽器只需要重新繪製該元素，成本沒有 Reflow 那麼高

## React Virtual DOM

傳統的方法是直接更改 Real DOM，像是如果我有 `document.getElementById().innerHTML = "new content"`，他會直接去更新 Real DOM，更新完後 Reflow 或 Repaint 可能會再做一遍，成本很高。如果我有數個相似的操作，就會很慢。

React 會在記憶體中建立一個 JavaScript 物件（Virtual DOM），這個物件是一個與 Real DOM 結構相似的樹。當 State 改變時， React 比較新舊版本的 Virtual DOM 的樹上的每一個節點，然後更新，最後批次更改 Real DOM，減少花費的成本。

### React 的 re-rendering

上面提到，React 會比較新舊 Virtual DOM 的差異，最後更新到 Real DOM 上面。其中他會檢查 Virtual DOM 更新的部分，稱為 re-render，而非指 repaint。

### React 中可以減少 re-rendering 的方式

- memo
- useCallback
- babel-plugin-react-compiler

### 參考資料

[浏览器的重绘 (repaint) 和重排 (reflow)](https://www.lumin.tech/articles/browser-reflow-repaint/)
[Why is every React site so slow?](https://www.youtube.com/watch?v=INLq9RPAYUw)
