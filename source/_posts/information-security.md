---
title: Frontend Information Security
date: 2025-02-04 13:59:12
tags: [frontend]
category:
---

- `npm init -y`

  - `-y` 代表接下來詢問的 `package name`、`version`、`description`...之類的東西一律回答 yes，快速建立 `package.json`
    ```json
    {
      "name": "infosec",
      "version": "1.0.0",
      "main": "index.js",
      "scripts": {
        "test": "echo \"Error: no test specified\" && exit 1"
      },
      "keywords": [],
      "author": "",
      "license": "ISC",
      "description": ""
    }
    ```

- `FQDN (Fully Qualified Domain Name)` 是指 `hostname` + `domain name`

  - 比方說 `bms.carbon-walker.com` 是 `FQDN`，其 `hostname` 是 bms，`domain name` 是 `carbon-walker.com`

- HTTP methods

  | Method  |                               作用                               |
  | :-----: | :--------------------------------------------------------------: |
  |   GET   |                             取得資源                             |
  |  HEAD   |                  取得 HTTP header，不包含 body                   |
  |  POST   |                        註冊資料、建立資源                        |
  |   PUT   |                更新資源，若該資源不存在就新建資源                |
  | DELETE  |                             刪除資源                             |
  | CONNECT |       建立到目標資源的 tunnel，用於 SSL/TLS 加密的代理連線       |
  | OPTIONS |                 查詢 server 可支援的 HTTP method                 |
  |  TRACE  | 伺服器直接回傳收到的資料，用於偵錯 (好像沒人在用，瀏覽器也不支援 |

  - 其中 `GET`、`HEAD`、`OPTIONS`、`TRACE` 不會影響到伺服器的資源，因此被 `RFC 7231` 視為安全的 HTTP method

- Status Code

  |          Method           |                  作用                   |
  | :-----------------------: | :-------------------------------------: |
  |       100 Continue        |     通知瀏覽器，伺服器處理尚未完成      |
  |          200 OK           |                順利完成                 |
  |        201 Created        |           順利完成新建立資源            |
  |   301 Moved Permanently   |          將指定資料移動到別處           |
  |         302 Found         | 暫時移動資源位置 (可能臨時要維修之類的) |
  |      400 Bad Request      |              請求資訊有誤               |
  |       404 Not Found       |                找不到啦                 |
  | 500 Internal Server Error |            伺服器內部有錯誤             |
  |  503 Service Unavailable  |             通知伺服器當機              |

- TLS/SSL

  - 加密傳輸資料
    - 公開金鑰加密
    - 對稱密鑰加密
    - 綜合
  - 驗證通訊對象
    - 數位憑證確認伺服器為本尊，非冒牌
  - 資料完整性
    - 避免金鑰和憑證被竄改，所以在開始跟伺服器通訊前要先檢查有沒有被偷改 (shakehand)

- Mixed Content

  - passive
  - active

- iframe (inline frame) 內嵌假網站防止方法
  1. CSP (HTTP Header)
     - 禁止任何網站嵌入
       - `X-Frame-Options: DENY` (聽說有點過時
       - `Content-Security-Policy: frame-ancestors 'none';`
     - 禁止自己以外的網站嵌入
       - `X-Frame-Options: SAMEORIGIN`
       - `Content-Security-Policy: frame-ancestors 'self';`
     - 兩個一起用會比較好，避免一些瀏覽器端的問題
  2. Frame busting (javascript)
  ```js
  if (window !== window.top) {
    window.top.location = window.location.href;
  }
  ```
  推薦文章: [不識廬山真面目：Clickjacking 點擊劫持攻擊](https://blog.huli.tw/2021/09/26/what-is-clickjacking/)
