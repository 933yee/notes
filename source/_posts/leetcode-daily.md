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
