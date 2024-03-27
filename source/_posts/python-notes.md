---
title: Python 筆記
date: 2024-03-27 18:57:06
tags: Python
---

### 讀取 .env

#### .env
```shell
MY_TOKEN = "example token"
```

#### 讀取
```python
import os
from dotenv import load_dotenv

load_dotenv() # load .env 的東東

MY_TOKEN = os.getenv('MY_TOKEN')

```


