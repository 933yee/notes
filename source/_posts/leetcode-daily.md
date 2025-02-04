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

> 連四天 easy 耶

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
