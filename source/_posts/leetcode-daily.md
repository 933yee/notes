---
title: 不知道這會持續幾天的 leetcode daily
date: 2025-01-31 21:22:11
tags: leetcode
category:
---

### 2025/02/01 [3151. Special Array I](https://leetcode.com/problems/special-array-i/description/?envType=daily-question&envId=2025-02-01)

1. bitwise xor + and 即可
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

1. drop 只能一次，最後要檢查頭尾
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
