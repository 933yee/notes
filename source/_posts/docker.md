---
title: Docker 筆記
date: 2025-02-11 13:23:25
tags: [docker, web]
category:
---

[Docker 架構](https://raw.githubusercontent.com/collabnix/dockerlabs/master/beginners/images/comp_client_server.jpg)

- Docker Daemon
  Docker Daemon 負責管理和執行容器的所有操作。

- Docker Compose
  定義和管理多個 Docker 容器。讓你能夠用一個 YAML config file 來定義所有容器，然後一個命令就能啟動他們，像是你同時要有前端、後端、資料庫的時候

- 步驟
  1. 建立 `Dockerfile`
  2. 用這個 `Dockerfile` 來建立 `Image`
  3. 用這個 `Image` 來建立和運行 `Container`
