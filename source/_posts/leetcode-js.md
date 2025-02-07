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

### [2704. To Be Or Not To Be](https://leetcode.com/studyplan/30-days-of-javascript/)

- 這題是寫測試 code，教我們怎麼 throw error

```js
/**
 * @param {string} val
 * @return {Object}
 */
var expect = function (val) {
  return {
    toBe: (v) => {
      if (v === val) return true;
      else throw new Error("Not Equal");
    },
    notToBe: (v) => {
      if (v !== val) return true;
      else throw new Error("Equal");
    },
  };
};

/**
 * expect(5).toBe(5); // true
 * expect(5).notToBe(5); // throws "Equal"
 */
```

### [2665. Counter II](https://leetcode.com/problems/counter-ii/description/?envType=study-plan-v2&envId=30-days-of-javascript)

```js
/**
 * @param {integer} init
 * @return { increment: Function, decrement: Function, reset: Function }
 */
var createCounter = function (init) {
  var cnt = init;
  return {
    increment: () => ++cnt,
    reset: () => (cnt = init),
    decrement: () => --cnt,
  };
};

/**
 * const counter = createCounter(5)
 * counter.increment(); // 6
 * counter.reset(); // 5
 * counter.decrement(); // 4
 */
```

### [2635. Apply Transform Over Each Element in Array](https://leetcode.com/problems/apply-transform-over-each-element-in-array/description/?envType=study-plan-v2&envId=30-days-of-javascript)

> 實作 map 來跑 input function，題目規定不能用 `Array.map` 讓我愣了一下
> 有各種不同的寫法

- `for loop`

  ```js
  /**
   * @param {number[]} arr
   * @param {Function} fn
   * @return {number[]}
   */
  var map = function (arr, fn) {
    const ret = [];
    for (let i = 0; i < arr.length; i++) ret.push(fn(arr[i], i));
    return ret;
  };
  ```

- js 可以不用 `push`，會自動填充

  ```js
  var map = function (arr, fn) {
    const ret = [];
    for (let i = 0; i < arr.length; i++) ret[i] = fn(arr[i], i);
    return ret;
  };
  ```

  - 補充

    - 如果中間有空的沒填，比方說

      ```js
      const arr = [];
      arr[0] = 0;
      arr[2] = 2;
      ```

      那麼 `arr[1]` 會自動填成 `undefined`，因此 `arr.length` 會是 `3`

    - `array` 可以用 `const` 是因為在 javascript 中，`const` 是 **保證變數本身的記憶體位置不變，但不保證內部內容不可變**，因此在陣列 arr reference 不變的情況下改內容是可以接受的

- `forEach`

  ```js
  var map = function (arr, fn) {
    const ret = [];
    arr.forEach((el, idx) => (ret[idx] = fn(el, idx)));
    return ret;
  };
  ```

  `forEach` 沒有回傳值，不能像 `map` 那樣直接 `return`

- `array.reduce`，但不太直觀

  - `array.reduce(callback, initialValue);`

  ```js
  var map = function (arr, fn) {
    return arr.reduce((ret, el, idx) => {
      ret[idx] = fn(el, idx);
      return ret;
    }, []);
  };
  ```

- 如果可以用 `Array.map`
  ```js
  var map = (arr, fn) => arr.map(fn);
  ```
  > 健康又快樂

### [2634. Filter Elements from Array](https://leetcode.com/problems/filter-elements-from-array/description/?envType=study-plan-v2&envId=30-days-of-javascript)

> 跟前一題類似

- `forEach`

  ```js
  /**
   * @param {number[]} arr
   * @param {Function} fn
   * @return {number[]}
   */
  var filter = function (arr, fn) {
    const ret = [];
    arr.forEach((el, idx) => fn(el, idx) && ret.push(el));
    return ret;
  };
  ```

- 如果可以用 `Array.filter`
  ```js
  var filter = (arr, fn) => arr.filter(fn);
  ```

### [2626. Array Reduce Transformation](https://leetcode.com/problems/array-reduce-transformation/description/?envType=study-plan-v2&envId=30-days-of-javascript)

> 跟前幾題類似

- `forEach`

  ```js
  /**
   * @param {number[]} nums
   * @param {Function} fn
   * @param {number} init
   * @return {number}
   */
  var reduce = function (nums, fn, init) {
    let acc = init;
    nums.forEach((el) => (acc = fn(acc, el)));
    return acc;
  };
  ```

- 如果可以用 `Array.reduce`
  ```js
  var reduce = (nums, fn, init) => nums.reduce(fn, init);
  ```

### [2629. Function Composition](https://leetcode.com/problems/function-composition/description/?envType=study-plan-v2&envId=30-days-of-javascript)

從最後面開始每個 function 跑一次

```js
/**
 * @param {Function[]} functions
 * @return {Function}
 */
var compose = function (functions) {
  return function (x) {
    functions.reverse().map((fn) => (x = fn(x)));
    return x;
  };
};

/**
 * const fn = compose([x => x + 1, x => 2 * x])
 * fn(4) // 9
 */
```

- `reduceRight`
  跟 `array.reduce()` 一樣，只是是從最右邊往左 reduce
  ```js
  var compose = function (functions) {
    return (x) => functions.reduceRight((acc, fn) => fn(acc), x);
  };
  ```

### [2703. Return Length of Arguments Passed](https://leetcode.com/problems/return-length-of-arguments-passed/description/?envType=study-plan-v2&envId=30-days-of-javascript)

好像只是在考 javascript 的 rest parameter?

```js
/**
 * @param {...(null|boolean|number|string|Array|Object)} args
 * @return {number}
 */
var argumentsLength = function (...args) {
  return args.length;
};

/**
 * argumentsLength(1, 2, 3); // 3
 */
```

### [2666. Allow One Function Call](https://leetcode.com/problems/allow-one-function-call/description/?envType=study-plan-v2&envId=30-days-of-javascript)

closure + rest parameter

```js
/**
 * @param {Function} fn
 * @return {Function}
 */
var once = function (fn) {
  let seen = false;
  return function (...args) {
    if (!seen) {
      seen = true;
      return fn(...args);
    }
  };
};

/**
 * let fn = (a,b,c) => (a + b + c)
 * let onceFn = once(fn)
 *
 * onceFn(1,2,3); // 6
 * onceFn(2,3,6); // returns undefined without calling fn
 */
```
