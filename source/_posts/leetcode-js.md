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

### [2623. Memoize](https://leetcode.com/problems/memoize/description/?envType=study-plan-v2&envId=30-days-of-javascript)

- Map

  - 在 javascript 中，`Map` 不能用中括號來存取，一定要 `Map.get()`、`Map.set()`、`Map.has()`。另外這題可以用 `JSON.stringif()` 把 `args` 變成 `key`

  ```js
  /**
   * @param {Function} fn
   * @return {Function}
   */
  function memoize(fn) {
    const hash = new Map();
    return function (...args) {
      const key = JSON.stringify(args);
      if (hash.has(key)) return hash.get(key);
      hash.set(key, fn(...args));
      return hash.get(key);
    };
  }

  /**
   * let callCount = 0;
   * const memoizedFn = memoize(function (a, b) {
   *	 callCount += 1;
   *   return a + b;
   * })
   * memoizedFn(2, 3) // 5
   * memoizedFn(2, 3) // 5
   * console.log(callCount) // 1
   */
  ```

- Object

  - `Object` 就比較靈活

  ```js
  /**
   * @param {Function} fn
   * @return {Function}
   */
  function memoize(fn) {
    const hash = {};
    return function (...args) {
      const key = JSON.stringify(args);
      if (hash[key] != undefined) return hash[key];
      hash[key] = fn(...args);
      return hash[key];
    };
  }
  ```

### [2723. Add Two Promises](https://leetcode.com/problems/add-two-promises/description/?envType=study-plan-v2&envId=30-days-of-javascript)

這邊要用 `Promise.all` 來等所有 `promise` 算好一起跑 `then`。

- 直接加

  - 因為這裡 arg 只有兩個數字，可以直接加起來就好

  ```js
  /**
   * @param {Promise} promise1
   * @param {Promise} promise2
   * @return {Promise}
   */
  var addTwoPromises = async function (promise1, promise2) {
    return Promise.all([promise1, promise2]).then(([a, b]) => a + b);
  };

  /**
   * addTwoPromises(Promise.resolve(2), Promise.resolve(2))
   *   .then(console.log); // 4
   */
  ```

- 當 arguments 不只兩個時，可以用 `Array.reduce()`
  ```js
  var addTwoPromises = async function (...args) {
    return Promise.all(args).then((ret) =>
      ret.reduce((acc, val) => acc + val, 0)
    );
  };
  ```

### [2621. Sleep](https://leetcode.com/problems/sleep/description/?envType=study-plan-v2&envId=30-days-of-javascript)

- `resolve()` 是 Promise 提供的 function，用來把 Promise 的狀態變成已完成。所以這邊在 `millis` 時間後執行 `resolve()` 就能完成 `return`。

```js
/**
 * @param {number} millis
 * @return {Promise}
 */
async function sleep(millis) {
  return new Promise((resolve) => setTimeout(resolve, millis));
}

/**
 * let t = Date.now()
 * sleep(100).then(() => console.log(Date.now() - t)) // 100
 */
```

### [2715. Timeout Cancellation](https://leetcode.com/problems/timeout-cancellation/description/?envType=study-plan-v2&envId=30-days-of-javascript)

> 滿酷的題目，很有趣

```js
/**
 * @param {Function} fn
 * @param {Array} args
 * @param {number} t
 * @return {Function}
 */
var cancellable = function (fn, args, t) {
  let timeout = setTimeout(() => fn(...args), t);
  return () => clearTimeout(timeout);
};

/**
 *  const result = [];
 *
 *  const fn = (x) => x * 5;
 *  const args = [2], t = 20, cancelTimeMs = 50;
 *
 *  const start = performance.now();
 *
 *  const log = (...argsArr) => {
 *      const diff = Math.floor(performance.now() - start);
 *      result.push({"time": diff, "returned": fn(...argsArr)});
 *  }
 *
 *  const cancel = cancellable(log, args, t);
 *
 *  const maxT = Math.max(t, cancelTimeMs);
 *
 *  setTimeout(cancel, cancelTimeMs);
 *
 *  setTimeout(() => {
 *      console.log(result); // [{"time":20,"returned":10}]
 *  }, maxT + 15)
 */
```
