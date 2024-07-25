---
title: Javascript Note
date: 2024-07-23 09:49:27
tags: javascript
category: Javascript
math: true
---

## let、const、var 的差別

<iframe width = "100%" height = "400" src="//www.youtube.com/embed/Pychc22EG4Q" frameborder="0" allowfullscreen></iframe>

### var
var 是 function-scoped 的變數，作用域為整個 function

```js
function exampleVar(){
    var x = 1;
    if (true){
        var x = 2;
        console.log(x); // 2
    }
    console.log(x); // 2
}
```

因此上面這個例子，整個 function 的 x 是一樣的

### 比較

|            |       var       |     let      |    const     |
| :--------: | :-------------: | :----------: | :----------: |
|    範圍    | function-scoped | block-scpoed | block-scoped |
| 可重複定義 |        O        |      X       |      X       |
|   可修改   |        O        |      O       |      X       |
|  hoisting  |        O        |      X       |      X       |

### 進階範例
```js
for (var i = 0; i < 3; i++) {
    setTimeout(function() {
        console.log(i); // 3, 3, 3
    }, 100);
}
```

因為 var， i 只有一個，所以每次生成的 closure 捕捉到的 i 會是一樣的

```js
for (let i = 0; i < 3; i++) {
    setTimeout(function() {
        console.log(i); // 0, 1, 2
    }, 100);
}
```

