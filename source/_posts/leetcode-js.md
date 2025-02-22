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

### [2725. Interval Cancellation](https://leetcode.com/problems/interval-cancellation/?envType=study-plan-v2&envId=30-days-of-javascript)

跟上一題差不多，但要先馬上 call fn 一次

```js
/**
 * @param {Function} fn
 * @param {Array} args
 * @param {number} t
 * @return {Function}
 */
var cancellable = function (fn, args, t) {
  fn(...args);
  const interval = setInterval(() => fn(...args), t);
  return () => clearInterval(interval);
};

/**
 *  const result = [];
 *
 *  const fn = (x) => x * 2;
 *  const args = [4], t = 35, cancelTimeMs = 190;
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
 *  setTimeout(cancel, cancelTimeMs);
 *
 *  setTimeout(() => {
 *      console.log(result); // [
 *                           //     {"time":0,"returned":8},
 *                           //     {"time":35,"returned":8},
 *                           //     {"time":70,"returned":8},
 *                           //     {"time":105,"returned":8},
 *                           //     {"time":140,"returned":8},
 *                           //     {"time":175,"returned":8}
 *                           // ]
 *  }, cancelTimeMs + t + 15)
 */
```

### [2637. Promise Time Limit](https://leetcode.com/problems/promise-time-limit/description/?envType=study-plan-v2&envId=30-days-of-javascript)

`Promise.race()` 會遍歷傳進來的一堆 `Promise`，然後回傳最先執行完的那個。

```js
/**
 * @param {Function} fn
 * @param {number} t
 * @return {Function}
 */
var timeLimit = function (fn, t) {
  return async function (...args) {
    const fn1 = fn(...args);
    const fn2 = new Promise((resolve, reject) =>
      setTimeout(() => reject("Time Limit Exceeded"), t)
    );
    return Promise.race([fn1, fn2]);
  };
};

/**
 * const limited = timeLimit((t) => new Promise(res => setTimeout(res, t)), 100);
 * limited(150).catch(console.log) // "Time Limit Exceeded" at t=100ms
 */
```

### [2622. Cache With Time Limit](https://leetcode.com/problems/cache-with-time-limit/description/?envType=study-plan-v2&envId=30-days-of-javascript)

在 js 中，用 `Function.prototype` 的方式可以宣告這個物件裡面包含的函式，像這題就宣告了 `get`、`set`、`count`

```js
var TimeLimitedCache = function () {
  this.cache = {};
};

/**
 * @param {number} key
 * @param {number} value
 * @param {number} duration time until expiration in ms
 * @return {boolean} if un-expired key already existed
 */
TimeLimitedCache.prototype.set = function (key, value, duration) {
  let ret = this.cache[key] != undefined;
  if (this.cache[key] != undefined) clearTimeout(this.cache[key].timer);
  else this.cache[key] = {};
  this.cache[key].val = value;
  this.cache[key].timer = setTimeout(() => {
    delete this.cache[key];
  }, duration);
  return ret;
};

/**
 * @param {number} key
 * @return {number} value associated with key
 */
TimeLimitedCache.prototype.get = function (key) {
  return this.cache[key] != undefined ? this.cache[key].val : -1;
};

/**
 * @return {number} count of non-expired keys
 */
TimeLimitedCache.prototype.count = function () {
  return Object.keys(this.cache).length;
};

/**
 * const timeLimitedCache = new TimeLimitedCache()
 * timeLimitedCache.set(1, 42, 1000); // false
 * timeLimitedCache.get(1) // 42
 * timeLimitedCache.count() // 1
 */
```

### [2627. Debounce](https://leetcode.com/problems/debounce/description/?envType=study-plan-v2&envId=30-days-of-javascript)

- 沒什麼特別的

```js
/**
 * @param {Function} fn
 * @param {number} t milliseconds
 * @return {Function}
 */
var debounce = function (fn, t) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), t);
  };
};

/**
 * const log = debounce(console.log, 100);
 * log('Hello'); // cancelled
 * log('Hello'); // cancelled
 * log('Hello'); // Logged at t=100ms
 */
```

### [2721. Execute Asynchronous Functions in Parallel](https://leetcode.com/problems/execute-asynchronous-functions-in-parallel/description/?envType=study-plan-v2&envId=30-days-of-javascript)

好難阿，題目規定不能用 `Promise.all`，要注意回傳的順序，還要確保所有 `Promise` 已經做完。

```js
/**
 * @param {Array<Function>} functions
 * @return {Promise<any>}
 */
var promiseAll = function (functions) {
  return new Promise((resolve, reject) => {
    let res = [];
    let count = 0;
    functions.map((fn, idx) => {
      fn()
        .then((ret) => {
          res[idx] = ret;
          count++;
          if (count == functions.length) resolve(res);
        })
        .catch((error) => reject(error));
    });
  });
};

/**
 * const promise = promiseAll([() => new Promise(res => res(42))])
 * promise.then(console.log); // [42]
 */
```

- 如果可以用 `Promise.all`
  ```js
  var promiseAll = (functions) => Promise.all(functions.map((fn) => fn()));
  ```

### [2727. Is Object Empty](https://leetcode.com/problems/is-object-empty/description/?envType=study-plan-v2&envId=30-days-of-javascript)

一開始最直覺的方法，直接轉成 `Array` 算 `length`，`Array` 也是 `Object` 的一種，所以不用管 `obj` 是不是 `Array`

```js
/**
 * @param {Object|Array} obj
 * @return {boolean}
 */
var isEmpty = function (obj) {
  return Object.keys(obj).length == 0;
};
```

更簡單的方法

```js
var isEmpty = function (obj) {
  for (const _ in obj) return false;
  return true;
};
```

### [2677. Chunk Array](https://leetcode.com/problems/chunk-array/description/?envType=study-plan-v2&envId=30-days-of-javascript)

> 我只會這種醜醜 code qq

```js
/**
 * @param {Array} arr
 * @param {number} size
 * @return {Array}
 */
var chunk = function (arr, size) {
  let ret = [];
  let cnt = -1;
  for (let i = 0; i < arr.length; i++) {
    if (i % size == 0) {
      ret.push([]);
      cnt++;
    }
    ret[cnt].push(arr[i]);
  }
  return ret;
};
```

- 更簡潔的寫法

  - `Array.slice()` 會回傳一個新的陣列，不會改變原本的陣列

  ```js
  let chunk = function (arr, size) {
    let ret = [];
    for (let i = 0; i < arr.length; i += size) ret.push(arr.slice(i, i + size));
    return ret;
  };
  ```

  `arr.slice` 會回傳 `[i, i+size)` 的陣列，不用擔心 `i+size` 超過 `arr.length` 的問題，因為 `slice` 會自動調整

### [2619. Array Prototype Last](https://leetcode.com/problems/array-prototype-last/description/?envType=study-plan-v2&envId=30-days-of-javascript)

`this[this.length - 1]` 也可以改成像 Python 那樣， `this.at(-1)`

```js
/**
 * @return {null|boolean|number|string|Array|Object}
 */
Array.prototype.last = function () {
  return this.length == 0 ? -1 : this[this.length - 1];
};

/**
 * const arr = [1, 2, 3];
 * arr.last(); // 3
 */
```

### [2631. Group By](https://leetcode.com/problems/group-by/description/?envType=study-plan-v2&envId=30-days-of-javascript)

不知道為什麼是 `medium`

```js
/**
 * @param {Function} fn
 * @return {Object}
 */
Array.prototype.groupBy = function (fn) {
  const ret = {};
  this.forEach((el) => {
    const key = fn(el);
    if (ret[key] == undefined) ret[key] = [];
    ret[key].push(el);
  });
  return ret;
};

/**
 * [1,2,3].groupBy(String) // {"1":[1],"2":[2],"3":[3]}
 */
```

### [2724. Sort By](https://leetcode.com/problems/sort-by/description/?envType=study-plan-v2&envId=30-days-of-javascript)

javascript 中，lambda function 可以直接寫在 `Array.sort()` 裡面

```js
/**
 * @param {Array} arr
 * @param {Function} fn
 * @return {Array}
 */
var sortBy = (arr, fn) => arr.sort((a, b) => fn(a) - fn(b));
```

不過 `arr.sort()` 會改變原本的 input array，實際上應該要複製一份才對

```js
var sortBy = (arr, fn) => [...arr].sort((a, b) => fn(a) - fn(b));
```

### [2722. Join Two Arrays by ID](https://leetcode.com/problems/join-two-arrays-by-id/description/?envType=study-plan-v2&envId=30-days-of-javascript)

有點麻煩的題目

```js
/**
 * @param {Array} arr1
 * @param {Array} arr2
 * @return {Array}
 */
var join = function (arr1, arr2) {
  const ret = [];
  const id_idx_map = {};
  arr1.forEach((el, idx) => {
    ret.push(el);
    id_idx_map[el.id] = idx;
  });
  arr2.forEach((el) => {
    if (id_idx_map[el.id] != undefined) {
      tar_arr = ret[id_idx_map[el.id]];
      Object.keys(el).forEach((key) => (tar_arr[key] = el[key]));
    } else ret.push(el);
  });
  return ret.sort((obj1, obj2) => obj1.id - obj2.id);
};
```

別人精簡的做法，結合我的 `ret` 和 `id_idx_map`，最後用 `Object.values()` 的時候會自動 sort，不過本質上差不多

```js
/**
 * @param {Array} arr1
 * @param {Array} arr2
 * @return {Array}
 */
var join = function (arr1, arr2) {
  const result = {};

  // 1. initialization
  arr1.forEach((item) => {
    result[item.id] = item;
  });
  // 2. joining
  arr2.forEach((item) => {
    if (result[item.id]) {
      Object.keys(item).forEach((key) => {
        result[item.id][key] = item[key];
      });
    } else {
      result[item.id] = item;
    }
  });

  return Object.values(result);
};
```
