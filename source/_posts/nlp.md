---
title: Natural Language Processing
date: 2025-09-24 10:03:22
tags: [nlp, ai, machine learning]
category: AI
math: true
---

以前一定要從語言學去分析，才能處理比較好。
現在都用統計的方式做，雖然人看不懂，但效果更好。

## Information Retrieval (IR): 資訊檢索

- Stemming: 詞幹還原
- Lemmatization: 詞形還原
- Stop Words: 常見但無意義的詞彙（如 "the", "is", "in"）
- Term Frequency (TF): 詞彙在文件中出現的頻率
  - $TF(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$
- Inverse Document Frequency (IDF): 詞彙在整個語料庫中出現的稀有程度
  - $IDF(t) = \log(\frac{N}{DF(t)})$
    - $N$: 文件總數
    - $DF(t)$: 包含詞彙 $t$ 的文件數
  - 同個 Feature 如果所有文本都有出現，代表他沒甚麼重要性，所以越稀有的詞彙，IDF 越高越重要
- Cosine Document Similarity: 衡量兩個向量之間的相似度
  - $Cosine\_Document\_Similarity(A, B) = \frac{A \cdot B}{||A|| \times ||B||}$
- BM25: 一種改進的 TF-IDF 權重計算方法，計算文件與查詢的相關性
  - $BM25(t, d) = IDF(t) \times \frac{f_{t,d} \times (k_1 + 1)}{f_{t,d} + k_1 \times (1 - b + b \times \frac{|d|}{avgdl})}$
    - $k_1$ 和 $b$ 是調整參數
    - $|d|$: 文件長度
    - $avgdl$: 平均文件長度

## Word Representation: 詞彙表示

- Bag of Words (BoW): 將文本表示為詞彙的無序集合，忽略語法和詞序
- One-hot Encoding: 將詞彙表示為高維稀疏向量，只有對應詞彙的位置為 1，其餘為 0
- Word Embeddings (dense space): 將詞彙映射到低維連續向量空間，捕捉詞彙之間的語義關係
- Latent Semantic Analysis (LSA): 使用奇異值分解 (SVD) 將詞彙和文件映射到潛在語義空間
- Continuous Bag of Words (CBOW): 預測中心詞彙，給定上下文詞彙
- Skip-gram: 預測上下文詞彙，給定中心詞
- part of speech (POS): 分析詞性
- word sense ambiguity (WSA): 詞義歧異
- N-grams: 連續 n 個字詞的組合
  - unigram: 1 個字詞
  - bigram: 2 個字詞
  - trigram: 3 個字詞
  - 越長越短都有各自的優缺點
- Linguistic Signature: 獨特且可辨識的語言模式
- Perplexity (PPL): 衡量語言模型預測下一個字詞的困難程度
  - $PPL(W) = P(w_1, w_2, ..., w_N)^{-1/N}$
  - PPL 越低，表示模型越好
  - 在 Fine-tuning 的時候，可以用來評估模型是不是保有原有的語言能力
- TF-IDF: 衡量詞彙在文件中的重要性
  - $TF-IDF(t, d) = TF(t, d) \times IDF(t)$
- PPMI (Positive Pointwise Mutual Information): 衡量詞彙與上下文之間的關聯強度
  - $PMI(t, c) = \log(\frac{P(t, c)}{P(t) \times P(c)})$
  - $PPMI(t, c) = \max(PMI(t, c), 0)$
    - $P(t, c)$: 詞彙 $t$ 和上下文 $c$ 同時出現的概率
    - $P(t)$: 詞彙 $t$ 出現的概率
    - $P(c)$: 上下文 $c$ 出現的概率
- Word2Vec: 基於 nn 的詞嵌入方法，包括 CBOW 和 Skip-gram 兩種模型
  - 要定義 `Positive Sample` 和 `Negative Sample`，且兩者比例要適當
- Contextualized Embeddings: 根據上下文動態生成詞嵌入，想是 BERT、GPT
  - 同一個詞彙在不同上下文中會有不同的表示，像是 `bank` 在 `river bank` 和 `financial bank` 中會有不同的向量
- Name Entity Recognition (NER): 識別文本中的專有名詞，如人名、地名、組織名等 s
