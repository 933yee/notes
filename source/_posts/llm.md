---
title: LLM
date: 2025-05-11 19:32:37
tags: [llm, research]
category: llm
---

- 詞向量
  像是如果要表示 **貓**，會用類似 [0.0074, 0.0023, -0.0012...] 的數字來表示貓的特徵，這些數字就是詞向量，共有 300 個數字。用這樣的向量可以表示詞空間的關係，像是 **狗**、**寵物** 等也用一樣的方法表示，這些詞在詞 World Space 中就會與 **貓** 比較接近。

  這種高維度的表示法還有一個優勢是，可以用向量運算推理單字，像是 **biggest** - **big** = **smallest** - **small**、**瑞士人** - **瑞士** = **德國人** - **德國** 等等。

  以 GPT-3 為例，詞向量的維度高達 12288，比 Google 的 word2vec 還要高出 20 倍。

- LLM
  會由多層 Transformer，前幾層的神經網路會專注理解句子的語法，解決字上的歧義，像是代詞 **his** 和多義詞 **bank** 等等。後面的層會專注於理解整個文本段落的理解。

- Memory

  - optimizer states, gradients, parameters, activations

- Batch size、 sequence length 與 activation memory 的關係
