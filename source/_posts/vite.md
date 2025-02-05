---
title: Vite
date: 2025-02-05 10:48:03
tags: [frontend, vite, website]
category:
---

> 原本的專案都是用 webpack 打包，現在想換成 vite 看看

### 安裝 Vite 和相關的 Plugins

```sh
npm install vite @vitejs/plugin-react --save-dev
```

### Restructure Project

Webpack 通常用 `index.js` 或 `index.jsx` 當作 entry point，但是 Vite 是用 `index.html`

### 新增 vite.config.js

```js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// Detect environment mode
const isDev = process.env.NODE_ENV === "development";
console.log("isDev: ", isDev);
export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [["babel-plugin-react-compiler"]],
      },
    }),
  ],
  root: "src", // Set project root to "src"
  resolve: {
    alias: {
      States: path.resolve(__dirname, "src/States"),
      Components: path.resolve(__dirname, "src/Components"),
      Global: path.resolve(__dirname, "src/Components/Global"),
      Home: path.resolve(__dirname, "src/Components/Home"),
      About: path.resolve(__dirname, "src/Components/About"),
      Contact: path.resolve(__dirname, "src/Components/Contact"),
      Price: path.resolve(__dirname, "src/Components/Price"),
      News: path.resolve(__dirname, "src/Components/News"),
      Products: path.resolve(__dirname, "src/Components/Products"),
    },
  },
  server: {
    port: 7070,
    strictPort: true,
    open: true, // Open browser automatically
    historyApiFallback: true,
  },
  build: {
    outDir: path.resolve(__dirname, "public"),
    sourcemap: isDev ? "inline-source-map" : false,
  },
  define: {
    "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV),
  },
  base: isDev ? "/" : "https://carbon-walker.com/",
});
```

### package.json

這部分要改成

```json
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "predeploy": "npm run build",
    "deploy": "gh-pages -d public",
  },
```

### 圖片問題

原本用 webpack 加載圖片時是用，他就能夠讀取到 `public` 資料夾裡面的 `images` 裡面的圖片

```html
<img
  className="h-[2.5rem] object-contain lg:h-[2rem] md:h-[2.2rem]"
  src="./images/logo.png"
  alt="logo"
/>
```

但換成 Vite 我發現路徑會有問題，要把原本放在先前的資料夾 `images` 放到 `src` 裡面的 `assets` 資料夾中，這樣跑 `npm run build` 的時候他會自動處理，複製一份到 `public`

### 刪掉 Webpack-Specific 的 Packages

```sh
npm remove webpack webpack-cli webpack-dev-server html-webpack-plugin babel-loader style-loader css-loader postcss-loader
```

雖然搞很久，但有感覺 Vite 速度有快很多，之後再看有沒有更酷的功能
