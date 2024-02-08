---
date: 2024-02-08 21:05:13
title: Push React App to Github Pages
category: [React]
tags: [React, Gtihub]
---

- 安裝套件
``` shell
npm install gh-pages --save-dev
yarn add -D gh-pages
```

- 在 package.json 裡面新增
```json
{
  "homepage" : "{url to your website}"
  "scripts": {
    "predeploy": "npm run build",
    "deploy": "gh-pages -d build"
  }
}
```

- 要 push 的時候就打以下指令，就可以自動 push 到 branch gh-pages
``` shell
npm run deploy
```
