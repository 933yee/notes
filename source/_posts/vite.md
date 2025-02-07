---
title: Webpack 換成 Vite
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
  plugins: [react()],
  root: "src",
  publicDir: "../public", // 因為這裡 root 設為 "src"，靜態資源放在 public 裡面，要加個 ../
  resolve: {
    alias: {
      States: path.resolve(__dirname, "src/States"),
      Utilities: path.resolve(__dirname, "src/Utilities"),
      Components: path.resolve(__dirname, "src/Components"),
    },
  },
  server: {
    port: 7070,
    strictPort: true,
    open: true, // Open browser automatically
  },
  build: {
    outDir: path.resolve(__dirname, "public"),
    sourcemap: isDev ? "inline-source-map" : false,
    // rollupOptions: {
    //   output: {
    //     manualChunks(id) {
    //       if (id.includes("node_modules")) {
    //         return id
    //           .toString()
    //           .split("node_modules/")[1]
    //           .split("/")[0]
    //           .toString();
    //       }
    //     },
    //   },
    // },
  },
  define: {
    "process.env.NODE_ENV": JSON.stringify(process.env.NODE_ENV),
  },
  base: isDev ? "/" : "https://bms.carbon-walker.com/",
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

### 刪掉 Webpack-Specific 的 Packages

```sh
npm remove webpack webpack-cli webpack-dev-server html-webpack-plugin babel-loader style-loader css-loader postcss-loader
```

雖然 Vite 速度感覺快很多，但是花了一堆時間在 debug，遇到一堆怪怪的問題，之後再看有沒有更酷的功能
![金句](https://media2.dev.to/dynamic/image/width=1000,height=500,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fd4u62vy3h2bptsynmuu6.png)
