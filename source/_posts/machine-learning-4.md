---
title: Machine Learning - Self-Attention & Transformer
date: 2025-03-06 13:22:11
tags: ml
category:
math: true
---

> 惡補 ML https://www.youtube.com/watch?v=Ye018rCVvOo&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J

## Self-Attention

先前提到的 `Input` 都只是一個 `Vector`，然而很多時候，模型吃的是 **一組 Vector**，又稱 `Vector Set`、`Sequence`，又可以分成三類

- 每個 `Vector` 有一個 `Label`，輸入的數量等於輸出的數量，稱為 `Sequence Labeling`
  - ex: `Pos-Tagging`
- 整個 `Sequence` 只有一個 `Label`
  - ex: `Sentiment Analysis`
- 機器自己決定要有幾個 `Label`
  - ex: `Sequence-to-Sequence`、`Machine Translation`

對於 `Sequence Labeling`，如果像前面提到的 `CNN` 一樣，每個 `Vector` 都是獨立的，可能會忽略掉 `Vector` 之間的關係 (Context)。你也可以把整個 `Sequence` 丟到 `CNN` 裡面，但參數量、計算量都會超大，又久又容易 `Overfitting`，因此有了 `Self-Attention`

![https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html](https://sebastianraschka.com/images/blog/2023/self-attention-from-scratch/summary.png)

`Self-Attention` 的概念是，對於每個 `Vector`，都會有一個 `Query`、`Key`、`Value`，然後透過 `Query` 和 `Key` 的 `Dot Product` 來計算 `Attention Score`，再透過 `Softmax` 來計算 `Attention Weight`，最後再把 `Value` 乘上 `Attention Weight` 來得到 `Output`

- `Softmax` 是最常見的，不過也可以用別的 `Activation Function`
- `Attention Weight` 會讓 `Model` 知道哪些 `Vector` 是重要的，哪些是不重要的

### Multi-Head Self-Attention

相關這件事情可能有很多種形式，為了要找到資料中不同種類的相關性，可以用 `Multi-Head Self-Attention`

- Head 的數量也是 `Hyperparameter`

![https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff406d55e-990a-4e3b-be82-d966eb74a3e7_1766x1154.png)

### Positional Encoding

`Self-Attention` 並不會考慮到 `Position`，因此需要加上 `Positional Encoding`，讓 `Model` 知道 `Vector` 的位置

![Positional Encoding](https://i.imgur.com/QYpn2J3.png)

### Truncated Self-Attention

有時候 `Sequence` 會超長，造成 `Attention Matrix` 太大，計算量太大，甚至 Train 不起來，因此可以用 `Truncated Self-Attention`，只考慮某距離以內的 `Vector`，不考慮太遠的 `Vector`

![Truncated Self-Attention](https://lh5.googleusercontent.com/E-BGMJXwdWbYORmuf9OFvIDZ9ciriH88oWI8otaIjJDNuCyYvdFMtAeR7HqbbhK_WwHSLhMGyr77wBh7W1_kB1AQ9XAdouBsONFfqWltGXxlqtclXC7uRUU5NUxwFU80JHhIHDE)

### Self-Attention vs CNN

一般圖片都是用 `CNN` 來處理，但其實 `CNN` 是 `Self-Attention` 的一種，只是 `CNN` 會考慮到 `Local Pattern`，而 `Self-Attention` 會考慮到 `Global Pattern`

![Self-Attention vs CNN](https://miro.medium.com/v2/resize:fit:1400/1*IRvXQeATmX0JxJxz45_XUA.png)

`Self-Attention` 就是一種更彈性的 `CNN`，因此在訓練資料很大的時候，`Self-Attention` 可能比 `CNN` 更好

![Self-Attention vs CNN](https://miro.medium.com/v2/resize:fit:1400/1*2xKkjuDVe8zTUa6lVf3QGw.png)

### Self-Attention vs RNN

現在 `RNN` 幾乎被 `Self-Attention` 取代，因為 `RNN` 有兩大缺陷

- `Long-Term Dependency` 的問題，當 `Sequence` 很長時，很容易忘記越早輸入進來的資料
- 只能 `Sequential` 計算，無法平行運算

![Self-Attention vs RNN](https://miro.medium.com/v2/resize:fit:1400/1*qATp1B0W4BK0J4IL-sIsig.png)

## Transformer

是一個 `Sequence-to-Sequence` 的模型，由機器自己決定輸出的長度，常用在 `Machine Translation`、`Speech Recognition`、`Speech Translation`、`Chatbot`

大部分的 NLP 問題都可以直接看成 `QA` 問題，而 `QA` 問題都可以看成 `Sequence-to-Sequence` 的問題，只要把 `Question` 和 `Context` 組合在一起，丟進 `Sequence-to-Sequence` 的模型裡面，就可以得到答案。但是 NLP 的問題中，客製化模型的表現通常會更好。

有很多應用都可以硬用 `Sequence-to-Sequence` 的模型，像是 `Syntacic Parsing`，可以把這棵樹轉成 `Sequence`，直接塞進 `Sequence-to-Sequence` 的模型裡。其他還有 `Multi-label Classification`、`Object Detection` 等

### Sequence-to-Sequence Structure

Transformer 的 Encoder 就像一位記憶力超強的老師，他把一整本書（你的輸入句子）讀完，並且整理出一本精華筆記（Encoder 的輸出）。
然後，Decoder 是一個學生，他想要用自己的話來解釋這本書的內容（生成輸出句子）。

但這位學生不會一次就把整本書背出來，而是一步一步地問老師：「接下來我要怎麼說？」
每當學生說出一個單詞，他就會回頭看看老師的筆記（Cross-Attention），確認自己沒說錯，然後再繼續下一個單詞。

所以，Encoder 負責總結資訊，Decoder 負責一步步產生句子，並透過 Cross-Attention 確保自己說的話合理。

#### Encoder

![Encoder](./images/machine-learning/Encoder.png)
![Encoder](./images/machine-learning/Encoder-1.png)

#### Decoder - Autoregressive
