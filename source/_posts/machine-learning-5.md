---
title: Machine Learning - PyTorch
date: 2025-03-06 13:22:11
tags: ml
category:
math: true
---

> 惡補 ML https://www.youtube.com/watch?v=Ye018rCVvOo&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J

- PyTorch 是一個開源的機器學習框架
  - 可以做 `Tensor Computation`，像是 `Numpy`，但可以用 `GPU` 加速
  - 可以幫你算 `Gradient`

#### Tensor

- Tensor: 高維度的矩陣或陣列
- 常用的 `Data Type` 有 `torch.float`、`torch.long`、`torch.FloatTensor`、`torch.LongTensor` 等
- PyTorch 的 `dim` 跟 `Numpy` 的 `axis` 一樣

##### Example

- Constructor

  ```python
  x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
  # tensor([1., 2., 3., 4.])

  x = torch.from_numpy(np.array([1, 2, 3, 4]))

  x = toch.zeros(2, 3)
  # tensor([[0., 0., 0.],
  #         [0., 0., 0.]])

  x = torch.ones(2, 3)
  # tensor([[1., 1., 1.],
  #         [1., 1., 1.]])
  ```

- Operators

  - `squeeze`
    ```python
    x = torch.zeros([1, 2, 3])
    print(x.shape)
    # torch.Size([1, 2, 3])
    x = x.squeeze(0) # 把第 0 維度的 1 拿掉
    print(x.shape)
    # torch.Size([2, 3])
    ```

  ````
  - `unsqueeze`
    ```python
    x = torch.zeros([2, 3])
    print(x.shape)
    # torch.Size([2, 3])
    x = x.unsqueeze(1) # 在第 1 維度插入 1
    print(x.shape)
    # torch.Size([2, 1, 3])
  ````

  - `cat`
    ```python
    x = torch.zeros([2, 1, 3])
    y = torch.zeros([2, 3, 3])
    z = torch.zeros([2, 2, 3])
    w = torch.cat([x, y, z], dim=1) # 合併第 1 維度
    print(w.shape)
    # torch.Size([2, 6, 3])
    ```
  - `pow`
    ```python
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    x = x.pow(2)
    # tensor([ 1.,  4.,  9., 16.])
    ```
  - `sum`
    ```python
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    x = x.sum()
    # tensor(10.)
    ```
  - `mean`
    ```python
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    x = x.mean()
    # tensor(2.5000)
    ```

##### vs Numpy

|        PyTorch         |        Numpy         |
| :--------------------: | :------------------: |
|        x.shape         |       x.shape        |
|        x.dtype         |       x.dtype        |
| x.reshape() / x.view() |     x.reshape()      |
|      x.squeeze()       |     x.squeeze()      |
|     x.unsqueeze(1)     | np.expand_dims(x, 1) |

##### Device

可以用 `toruch.cuda.is_available()` 檢查有沒有 `GPU`，然後用 `torch.cuda.device_count()` 來看有幾個 `GPU`

```python
x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
x = x.to('cuda') # 把 tensor 放到 GPU 上計算
```

## DNN 的架構

![PyTorch DNN](./images/machine-learning/PytorchDNN.png)

### Gradient

```python
x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True) # 設定 requires_grad=True 會記錄 Gradient
z = x.pow(2).sum()
z.backward() # 計算 Gradient
print(x.grad) # tensor([[ 2.,  0.],
              #         [-2.,  2.]])
```

### Dataset & DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    # Read data and preprocess
    def __init__(self):
        self.data = ...

    # Returns one sample at a time
    def __getitem__(self, idx):
        return self.data[idx]

    # Returns the size of the dataset
    def __len__(self):
        return len(self.data)

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

![Dataset & DataLoader](./images/machine-learning/DatasetDataLoader.png)

### Neural Network Layers

Linear Layer (Fully-connected Layer)

```python
layer = torch.nn.Linear(32, 64) # 32 -> 64
print(layer.weight.shape)
# torch.Size([64, 32])
print(layer.bias.shape)
# torch.Size([64])
```

### Activation Function

```python
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax()
```

### Loss Function

```python
nn.MSELoss()
nn.CrossEntropyLoss()
```

### Optimizer

```python
torch.optim.SGD(model.parameters(), lr, momentum) # Stoachastic Gradient Descent
torch.optim.Adam(model.parameters(), lr) # Adam
```

### Build Model

```python
import torch.nn as nn

class MyModel(nn.Module):
    # Initialize the model & define the layers
    def __init__(self):
        super(MyModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(10, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
        )

    # Compute the output of NN
    def forward(self, x):
        return self.nn(x)
```

### Training

```python
dataset = MyDataset(filename)
training_set = DataLoader(dataset, batch_size=16, shuffle=True)
model = MyModel().to('cuda')
critertion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train() # Set the model to training mode
    for x, y in training_set:
        x, y = x.to('cuda'), y.to('cuda') # Move the data to GPU
        optimizer.zero_grad() # Clear the gradient
        y_pred = model(x) # Forward pass
        loss = criterion(y_pred, y) # Compute the loss
        loss.backward() # Compute the gradient
        optimizer.step() # Update the parameters
```

### Evaluation (Validation)

```python
model.eval() # Set the model to evaluation mode
total_loss = 0
for x, y in validation_set:
    x, y = x.to('cuda'), y.to('cuda')
    with torch.no_grad(): # Disable gradient computation
        y_pred = model(x)
        loss = criterion(y_pred, y)

    total_loss += loss.cpu().item() * x.size(0) # Accumulate the loss
    avg_loss = total_loss / len(validation_set.dataset) # Compute the average loss
```

### Evaluation (Testing)

```python
model.eval() # Set the model to evaluation mode
predictions = []
for x in test_set:
    x = x.to('cuda')
    with torch.no_grad(): # Disable gradient computation
        y_pred = model(x)
        predictions.append(y_pred.cpu())
```

### Save & Load Model

```python
torch.save(model.state_dict(), path) # Save the model

checkpoint = torch.load(path) # Load the model
model.load_state_dict(checkpoint)
```

## 可重現性 (Reproducibility)

為了讓每次執行的結果都保持一樣，不然很難確定效果改進是因為調整 `hyperparameter`，還是來自隨機性的變異。

```python
myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
```

- `myseed = 42069`: 設定固定的 `seed`
- `torch.backends.cudnn.deterministic = True`: `cuDNN` 為了加速計算，會默認選擇最快的方法，可能導致 non-deterministic 的行為發生，改成 `True` 可以確保每次結果都一樣
- `torch.backends.cudnn.benchmark = False`: 禁止 `cuDNN` 自己選擇最佳的計算方法，使用固定的方法來保證每次計算方式相同

如果為了 Performance 而不在乎 Reproducibility，可以把上面設成相反的
