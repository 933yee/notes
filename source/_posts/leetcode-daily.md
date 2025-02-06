---
title: 不知道這會持續幾天的 leetcode daily
date: 2025-01-31 21:22:11
tags: leetcode
category:
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
