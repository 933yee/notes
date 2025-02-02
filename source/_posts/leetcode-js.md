---
title: leetcode javascript 30 days
date: 2025-02-01 23:48:41
tags: leetcode
category:
---

> 沒在 leetcode 寫過 js 耶，酷

### [2667. Create Hello World Function](https://leetcode.com/problems/create-hello-world-function/description/?envType=study-plan-v2&envId=30-days-of-javascript)

- 了解 closure 的概念就好

```js
/**
 * @return {Function}
 */
var createHelloWorld = function () {
  return function (...args) {
    return "Hello World";
  };
};

/**
 * const f = createHelloWorld();
 * f(); // "Hello World"
 */
```

### [2620. Counter](https://leetcode.com/problems/counter/description/?envType=study-plan-v2&envId=30-days-of-javascript)

- 一樣 closure，之後 call return 的 function 都可以修改最初傳進來的變數的值

```js
/**
 * @param {number} n
 * @return {Function} counter
 */
var createCounter = function (n) {
  return function () {
    return n++;
  };
};

/**
 * const counter = createCounter(10)
 * counter() // 10
 * counter() // 11
 * counter() // 12
 */
```
