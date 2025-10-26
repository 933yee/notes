---
title: Algorithm
date: 2025-10-24 11:22:00
tags: [algorithm]
category: Algorithm
math: true
---

## Moore’s Voting Algorithm 摩爾投票法

找出一個序列中出現次數超過一半的元素。 (majority element)

```python
candidate = None
count = 0
for num in nums:
    if count == 0:
        candidate = num
    if num == candidate:
        count += 1
    else:
        count -= 1
```
