---
title: 不知道這會持續幾天的 leetcode daily
date: 2025-01-31 21:22:11
tags: leetcode
category:
math: true
---

### 2025/02/01 [3151. Special Array I](https://leetcode.com/problems/special-array-i/description/?envType=daily-question&envId=2025-02-01)

- bitwise xor + and 即可
  - Time Complexity $O(n)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    bool isArraySpecial(vector<int>& nums) {
        for(int i=1; i<nums.size(); i++){
            if(((nums[i] ^ nums[i-1]) & 1) != 1)
                return false;
        }
        return true;
    }
};
```

### 2025/02/02 [1752. Check if Array Is Sorted and Rotated](https://leetcode.com/problems/check-if-array-is-sorted-and-rotated/description/?envType=daily-question&envId=2025-02-02)

- drop 只能一次，最後要檢查頭尾
  - Time Complexity $O(n)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    bool check(vector<int>& nums) {
        bool drop = false;
        int n = nums.size();
        for(int i=1; i<n; i++){
            if(nums[i] < nums[i-1]){
                if(drop) return false;
                drop = true;
            }
        }
        return (nums[n-1] > nums[0]) ? !drop : true;
    }
};
```

### 2025/02/03 [3105. Longest Strictly Increasing or Strictly Decreasing Subarray](https://leetcode.com/problems/longest-strictly-increasing-or-strictly-decreasing-subarray/description/?envType=daily-question&envId=2025-02-03)

- 寫個 cnt，遍歷陣列兩次，算出最長的值
  - Time Complexity $O(n)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    int longestMonotonicSubarray(vector<int>& nums) {
        int n = nums.size(), cnt = 1;
        int res = INT_MIN;
        for(int i=1; i<n; i++){
            if(nums[i] > nums[i-1]) cnt++;
            else cnt = 1;
            res = max(res, cnt);
        }
        cnt = 1;
        for(int i=1; i<n; i++){
            if(nums[i] < nums[i-1]) cnt++;
            else cnt = 1;
            res = max(res, cnt);
        }
        return max(res, cnt);
    }
};
```

### 2025/02/04 [1800. Maximum Ascending Subarray Sum](https://leetcode.com/problems/maximum-ascending-subarray-sum/description/?envType=daily-question&envId=2025-02-04)

- 遍歷一次，符合條件就累加
  - Time Complexity $O(n)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    int maxAscendingSum(vector<int>& nums) {
        int res = nums[0], n = nums.size(), cnt = nums[0];
        for(int i=1; i<n; i++){
            nums[i] > nums[i-1] ? cnt += nums[i] : cnt = nums[i];
            res = max(res, cnt);
        }
        return res;
    }
};
```

### 2025/02/05 [1790. Check if One String Swap Can Make Strings Equal](https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal/description/?envType=daily-question&envId=2025-02-05)

> 連五天 easy 耶

- 題目只允許交換一次，所以記住兩個 idx 就好
  - Time Complexity $O(n)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    bool areAlmostEqual(string s1, string s2) {
        int n = s1.size(), cnt = 0;
        int idx1, idx2;
        for(int i=0; i<n; i++){
            if(s1[i] != s2[i]){
                cnt++;
                if(cnt == 1) idx1 = i;
                else idx2 = i;
            }
        }
        if(cnt == 0) return true;
        if(cnt == 2 && s1[idx1] == s2[idx2] && s1[idx2] == s2[idx1]) return true;
        return false;
    }
};
```

### 2025/02/06 [1726. Tuple with Same Product](https://leetcode.com/problems/tuple-with-same-product/description/?envType=daily-question&envId=2025-02-06)

- 每個數字都不同，所以考慮在不同位置的情況時可以直接乘以 8。用 hash map 來儲存計算的結果的 counter。
  - Time Complexity $O(n)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    int tupleSameProduct(vector<int>& nums) {
        unordered_map<int, int> mp;
        int n = nums.size(), res = 0;
        for(int i=0; i<n; i++){
            for(int j=i+1; j<n; j++){
                int p = nums[i] * nums[j];
                mp[p]++;
            }
        }
        for(auto& i:mp)
            res += (i.second * (i.second - 1)) / 2 * 8;
        return res;
    }
};
```

### 2025/02/07 [3160. Find the Number of Distinct Colors Among the Balls](https://leetcode.com/problems/find-the-number-of-distinct-colors-among-the-balls/description/?envType=daily-question&envId=2025-02-07)

- 開兩個 hash map，記錄 **球對應的顏色** 和 **某顏色的數量**，當顏色數量變成 0 / 1 時，代表 少一種 / 多一種 顏色。
  - Time Complexity $O(n)$
  - Space Complexity $O(n)$

```cpp
class Solution {
public:
    vector<int> queryResults(int limit, vector<vector<int>>& queries) {
        vector<int> res;
        unordered_map<int, int> ball, color;
        int cnt = 0;
        for(auto& v:queries){
            int a = v[0], b = v[1];
            if(ball[a] != b){
                color[ball[a]]--;
                if(color[ball[a]] == 0) cnt--;
                ball[a] = b;
                color[ball[a]]++;
                if(color[ball[a]] == 1) cnt++;
            }
            res.push_back(cnt);
        }
        return res;
    }
};
```

### 2025/02/08 [2349. Design a Number Container System](https://leetcode.com/problems/design-a-number-container-system/description/?envType=daily-question&envId=2025-02-08)

- 因為 `1 <= index, number <= 109` ，需要開一個 hash map，負責記錄該 index 對應到的值，再開另一個 hash map，記錄 value 有哪些 index 指著。
  - Time Complexity $O(n \lg(n))$
  - Space Complexity $O(n)$

```cpp
class NumberContainers {
public:
    NumberContainers() {}

    void change(int index, int number) {
        if(index_mp.find(index) != index_mp.end()){
            value_map[index_mp[index]].erase(index);
            value_map[number].erase(index);
        }
        index_mp[index] = number;
        value_map[number].insert(index);
    }

    int find(int number) {
        if(value_map[number].empty()) return -1;
        return *value_map[number].begin();
    }
    unordered_map<int, int> index_mp;
    unordered_map<int, set<int>> value_map;
};

/**
 * Your NumberContainers object will be instantiated and called as such:
 * NumberContainers* obj = new NumberContainers();
 * obj->change(index,number);
 * int param_2 = obj->find(number);
 */
```

### 2025/02/09 [2364. Count Number of Bad Pairs](https://leetcode.com/problems/count-number-of-bad-pairs/description/?envType=daily-question&envId=2025-02-09)

- 用 hash map 記錄 index 和 value 的差值有幾個
  - Time Complexity $O(n)$
  - Space Complexity $O(n)$

```cpp
class Solution {
public:
    long long countBadPairs(vector<int>& nums) {
        long long ret = 0;
        unordered_map<int, int> mp;
        for(int i=0; i<nums.size(); i++){
            ret += i - mp[i - nums[i]];
            mp[i - nums[i]]++;
        }
        return ret;
    }
};
```

### 2025/02/10 [3174. Clear Digits](https://leetcode.com/problems/clear-digits/description/?envType=daily-question&envId=2025-02-10)

- 用 Stack 的方式記錄最後要回傳的值
  - Time Complexity $O(n)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    string clearDigits(string s) {
        string ret;
        for(char& c:s){
            if(isdigit(c) && ret.size()) ret.pop_back();
            else ret.push_back(c);
        }
        return ret;
    }
};
```

### 2025/02/11 [1910. Remove All Occurrences of a Substring](https://leetcode.com/problems/remove-all-occurrences-of-a-substring/description/?envType=daily-question&envId=2025-02-11)

> 快忘記 substr 和 find 怎麼用了

- 用 `string.substr()` 每次都往前檢查，找到就刪掉
  - Time Complexity $O(n^2)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    string removeOccurrences(string s, string part) {
        string ret;
        int n = part.size();
        for(char& c:s){
            ret.push_back(c);
            if(ret.size() >= n && ret.substr(ret.size() - n) == part)
                ret.erase(ret.size() - n, n);
        }
        return ret;
    }
};
```

- 用 `string.find()` 來檢查，可以更簡短
  - Time Complexity $O(n^2)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    string removeOccurrences(string s, string part) {
        int idx = s.find(part);
        while(idx != string::npos){
            s.erase(idx, part.size());
            idx = s.find(part);
        }
        return s;
    }
};
```

### 2025/02/12 [2342. Max Sum of a Pair With Equal Sum of Digits](https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/description/?envType=daily-question&envId=2025-02-12)

- 記錄曾經算出的最大值 `acc` 就好
  - Time Complexity $O(n)$
  - Space Complexity $O(n)$

```cpp
class Solution {
public:
    int maximumSum(vector<int>& nums) {
        unordered_map<int, int> mp;
        int res = -1;
        for(int& n:nums){
            int acc = 0, n_cpy = n;
            while(n_cpy){
                acc += n_cpy%10;
                n_cpy /= 10;
            }
            if(mp.find(acc) != mp.end())
                res = max(res, mp[acc] + n);
            mp[acc] = max(mp[acc], n);
        }
        return res;
    }
};
```
