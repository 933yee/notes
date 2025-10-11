---
title: Deep Learning Lab
date: 2025-10-11 13:41:03
tags: [deep learning, ai, machine learning]
category: AI
math: true
---

## Lab 10: Word2Vec & Noise Contrastive Estimation using Subclassing

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable warning and info message
import tensorflow as tf
import numpy as np
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
```

Successfully downloaded the file data/text8.zip

### 下載並讀取資料集

```python
import os
import urllib

# Download the data.
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
DATA_FOLDER = "data"
FILE_NAME = "text8.zip"
EXPECTED_BYTES = 31344016

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def download(file_name, expected_bytes):
    """ Download the dataset text8 if it's not already downloaded """
    local_file_path = os.path.join(DATA_FOLDER, file_name)
    if os.path.exists(local_file_path):
        print("Dataset ready")
        return local_file_path
    file_name, _ = urllib.request.urlretrieve(os.path.join(DOWNLOAD_URL, file_name), local_file_path)
    file_stat = os.stat(local_file_path)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded the file', file_name)
    else:
        raise Exception(
              'File ' + file_name +
              ' might be corrupted. You should try downloading it with a browser.')
    return local_file_path

make_dir(DATA_FOLDER)
file_path = download(FILE_NAME, EXPECTED_BYTES)

import zipfile

# Read the data into a list of strings.
def read_data(file_path):
    """ Read data into a list of tokens """
    with zipfile.ZipFile(file_path) as f:
        # tf.compat.as_str() converts the input into string
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(file_path)
print('Data size', len(vocabulary))
vocabulary[:5]
```

Data size 17005207
['anarchism', 'originated', 'as', 'a', 'term']

### 建立字典

- 常見詞彙會被編號為較小的 ID
- 不常見的詞彙會被標記為 `UNK`，並被編號為 0

````python
import collections
# Build the dictionary and replace rare words with UNK token.
def build_dataset(words, n_words):
    """ Create two dictionaries and count of occuring words
        - word_to_id: map of words to their codes
        - id_to_word: maps codes to words (inverse word_to_id)
        - count: map of words to count of occurrences
    """
    # map unknown words to -1
    count = [['UNK', -1]]
    # count of occurences for words in vocabulary
    count.extend(collections.Counter(words).most_common(n_words - 1))
    word_to_id = dict() # (word, id)
    # record word id
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys())) # (id, word)
    return word_to_id, id_to_word, count

def convert_words_to_id(words, dictionary, count):
    """ Replace each word in the dataset with its index in the dictionary """
    data_w2id = []
    unk_count = 0
    for word in words:
        # return 0 if word is not in dictionary
        index = dictionary.get(word, 0) # 找不到就回傳 0 (UNK)
        if index == 0:
            unk_count += 1
        data_w2id.append(index)
    count[0][1] = unk_count
    return data_w2id, count
    ```

"""Filling 4 global variables:
# data_w2id - list of codes (integers from 0 to vocabulary_size-1).
              This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# word_to_id - map of words(strings) to their codes(integers)
# id_to_word - maps codes(integers) to words(strings)
"""

vocabulary_size = 50000
word_to_id, id_to_word, count = build_dataset(vocabulary, vocabulary_size)
data_w2id, count = convert_words_to_id(vocabulary, word_to_id, count)
del vocabulary  # reduce memory.

print('Most common words (+UNK)', count[:5])
print('Sample data: {}'.format(data_w2id[:10]))
print([id_to_word[i] for i in data_w2id[:10]])
````

Most common words (+UNK) [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)]
Sample data: [5234, 3081, 12, 6, 195, 2, 3134, 46, 59, 156]
['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']

```python
# utility function
# 給一串文字（已轉成 ID）和一個「視窗大小」，輸出許多 (中心詞, 上下文詞) 的配對。
def generate_sample(center_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    for idx, center in enumerate(center_words):
        context = random.randint(1, context_window_size)
        # get a random target before the center word
        # 取出中心詞左邊最多 context 個字
        for target in center_words[max(0, idx - context) : idx]:
            yield center, target
        # get a random target after the center word
        # 取出中心詞右邊最多 context 個字
        for target in center_words[idx + 1 : idx + context + 1]:
            yield center, target

# 把上面那個樣本產生器包成「批次」(batch)，每次吐出兩個 NumPy 陣列：center_batch 和 target_batch。
def batch_generator(data, skip_window, batch_size):
    """ Group a numeric stream into batches and yield them as Numpy arrays. """
    single_gen = generate_sample(data, skip_window)
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1], dtype=np.int32)
        for idx in range(batch_size):
            center_batch[idx], target_batch[idx] = next(single_gen)
        yield center_batch, target_batch
```

### 設定訓練參數，建立模型

```python
## some training settings
training_steps = 80000
skip_step = 2000

## some hyperparameters
batch_size = 512
embed_size = 512
num_sampled = 256
learning_rate = 1.0
```

```python
# from tensorflow.keras.layers import Layer
# 所有 TensorFlow 自訂層都要繼承它，並覆寫兩個方法：
# __init__(): 定義要訓練的權重
# call(): 前向傳播的計算邏輯
from tensorflow.python.keras.layers import Layer

# embedding matrix - hidden layer
# 這一層用來把「字的 ID」轉成對應的向量 (embedding)
class embedding_lookup(Layer):
    def __init__(self):
        super(embedding_lookup, self).__init__()
        # 設定初始值的方法（Glorot/Xavier 初始化），讓權重不會太大或太小
        embedding_init = tf.keras.initializers.GlorotUniform()
        # 建立一個可訓練矩陣
        self.embedding_matrix = self.add_weight(name="embedding_matrix",
                                                trainable=True,
                                                shape=[vocabulary_size, embed_size],
                                                initializer=embedding_init)

    def call(self, inputs):
        center_words = inputs
        # tf.nn.embedding_lookup(matrix, ids) 會幫你做查表，從 embedding_matrix 取出指定 ID 的那幾行。
        embedding = tf.nn.embedding_lookup(self.embedding_matrix,
                                           center_words,
                                           name='embedding')
        return embedding

# context matrix - prediction layer
# 用來學習「正樣本」比「隨機噪聲詞」更相關
class nce_loss(Layer):
    def __init__(self):
        super(nce_loss, self).__init__()
        # 截斷正態分佈，每個權重初始值會介於一個小範圍內
        nce_w_init = tf.keras.initializers.TruncatedNormal(stddev=1.0/(embed_size ** 0.5))
        self.nce_weight = self.add_weight(name='nce_weight',
                                          trainable=True,
                                          shape=[vocabulary_size, embed_size],
                                          initializer=nce_w_init)

        self.nce_bias = self.add_weight(name='nce_bias',
                                        trainable=True,
                                        shape=[vocabulary_size],
                                        initializer=tf.keras.initializers.Zeros)

    def call(self, inputs):
        # （中心詞向量, 目標詞 ID）
        embedding, target_words = inputs[0], inputs[1]
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight,
                                             biases=self.nce_bias,
                                             labels=target_words,
                                             inputs=embedding,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size),
                                             name='loss')
        return loss
```

### 建立整個 Word2Vec 模型

```python
# from tensorflow.keras import Model, Input
from tensorflow.python.keras import Model, Input

center_words = Input(shape=(), name='center_words', dtype='int32')
target_words = Input(shape=(1), name='target_words', dtype='int32')

embedding = embedding_lookup()(center_words)
loss = nce_loss()((embedding, target_words))

word2vec = Model(name='word2vec',
                 inputs=[center_words, target_words],
                 outputs=[loss])

word2vec.summary()
```

```
Model: "word2vec"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
center_words (InputLayer)       [(None,)]            0
__________________________________________________________________________________________________
embedding_lookup (embedding_loo (None, 512)          25600000    center_words[0][0]
__________________________________________________________________________________________________
target_words (InputLayer)       [(None, 1)]          0
__________________________________________________________________________________________________
nce_loss (nce_loss)             ()                   25650000    embedding_lookup[0][0]
                                                                 target_words[0][0]
==================================================================================================
Total params: 51,250,000
Trainable params: 51,250,000
Non-trainable params: 0
__________________________________________________________________________________________________
```

### 生成資料集 (中心詞, 上下文詞) batch

```python
## generator for `tf.data.Dataset`
def gen():
    """ Return a python generator that generates batches. """
    yield from batch_generator(data_w2id, 2, batch_size)



dataset = tf.data.Dataset.from_generator(
  gen,                                 # 資料來源：Python generator
  (tf.int32, tf.int32),                # 每次 yield 的兩個輸出類型
  (tf.TensorShape([batch_size]),       # 第 1 個輸出形狀
    tf.TensorShape([batch_size, 1])),  # 第 2 個輸出形狀
).repeat()
```

### 定義優化器和訓練指標

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.1,nesterov=True)
```

### 訓練模型

```python
@tf.function
def train_step(center_words, target_words):
    # tf.GradientTape() 是 TensorFlow 的「自動微分引擎 (autodiff engine)」。
    # 所有在這個 with 區塊裡發生的計算，
    # 都會被記錄在 tape 物件裡。
    with tf.GradientTape() as tape:
        # forward pass
        loss = word2vec([center_words, target_words])

    # backward pass
    gradients = tape.gradient(loss, word2vec.trainable_variables)
    # update weights
    optimizer.apply_gradients(zip(gradients, word2vec.trainable_variables))
    # 印出平均 loss
    train_loss(loss)
```

```python
x = []
y = []
for step, (center_words, target_words) in enumerate(dataset):
    if step == training_steps:
        break
    train_step(center_words, target_words)

    if ((step+1) % skip_step) == 0:
        template = 'Step {:0}, Loss: {:.2f}'
        x.append(step+1)
        y.append(train_loss.result())
        print(template.format(step+1, train_loss.result()))
        train_loss.reset_state()
```

Step 2000, Loss: 97.47
Step 4000, Loss: 20.06
Step 6000, Loss: 13.44
Step 8000, Loss: 10.15
Step 10000, Loss: 9.45
Step 12000, Loss: 8.64
Step 14000, Loss: 8.10
Step 16000, Loss: 7.40
Step 18000, Loss: 7.18
Step 20000, Loss: 6.91
Step 22000, Loss: 6.88
Step 24000, Loss: 6.70
Step 26000, Loss: 6.56
Step 28000, Loss: 6.40
Step 30000, Loss: 6.42
Step 32000, Loss: 6.38
Step 34000, Loss: 6.27
Step 36000, Loss: 6.26
Step 38000, Loss: 6.15
Step 40000, Loss: 6.13
Step 42000, Loss: 6.10
Step 44000, Loss: 5.95
Step 46000, Loss: 5.94
Step 48000, Loss: 5.96
Step 50000, Loss: 5.90
...
Step 74000, Loss: 5.77
Step 76000, Loss: 5.45
Step 78000, Loss: 5.59
Step 80000, Loss: 5.71
