---
title: LeetCode Contest
date: 2025-11-02 22:05:11
categories: [Leetcode]
tags: [leetcode]
math: true
---

## Weekly Contest 474

### 3731. Find Missing Elements - Easy

給定 nums 陣列，找出裡面最小和最大的值之間缺少的所有整數。

```c++
class Solution {
public:
    std::vector<int> findMissingElements(std::vector<int>& nums) {
        int min_val = *min_element(nums.begin(), nums.end());
        int max_val = *max_element(nums.begin(), nums.end());

        unordered_set<int> nums_set(nums.begin(), nums.end());

        vector<int> missing;

        for (int i = min_val + 1; i < max_val; ++i) {
            if (nums_set.find(i) == nums_set.end())
                missing.push_back(i);
        }

        return missing;
    }
};
```

### 3732. Maximum Product of Three Elements After One Replacement - Medium

給定 nums 陣列，可以用 [$-10^5, 10^5$] 範圍內的任意整數替換陣列中的一個元素，求替換後三個元素乘積的最大值。

```c++
class Solution {
public:
    long long maxProduct(std::vector<int>& nums) {
        using ll = long long;
        int n = nums.size();
        sort(nums.begin(), nums.end());

        ll s1 = nums[0];
        ll s2 = nums[1];
        ll l1 = nums[n - 1];
        ll l2 = nums[n - 2];

        const ll R = 100000LL;

        ll prod1 = max(s1 * s2, l1 * l2) * R;
        ll prod2 = (s1 * l1) * (-R);

        return max({prod1, prod2});
    }
};
```

### 3733. Minimum Time to Complete All Deliveries - Medium

給定兩個無人機， 分別需要完成 d[0] 和 d[1] 個任務，兩台分別每 r[0] 和 r[1] 小時需要充電一次，且兩者不能同時工作，求完成所有任務的最少時間。

```cpp
class Solution {
    // 輾轉相除法求最大公因數
    long long gcd(long long a, long long b) {
        while (b) {
            a %= b;
            swap(a, b);
        }
        return a;
    }

    // 求最小公倍數
    long long lcm(long long a, long long b) {
        if (a == 0 || b == 0) return 0;
        return (a / gcd(a, b)) * b;
    }

     bool check(long long T, long long d0, long long d1, long long r0, long long r1, long long r_lcm) {
        long long recharge_0 = T / r0;
        long long recharge_1 = T / r1;
        long long recharge_both = T / r_lcm;

        long long slots_0 = recharge_1 - recharge_both;
        long long slots_1 = recharge_0 - recharge_both;
        long long slots_both = T - recharge_0 - recharge_1 + recharge_both;

        long long remaining_0 = (d0 > slots_0) ? (d0 - slots_0) : 0;

        long long remaining_1 = (d1 > slots_1) ? (d1 - slots_1) : 0;

        return (remaining_0 + remaining_1) <= slots_both;
    }

public:

    long long minimumTime(vector<int>& d, vector<int>& r) {
        long long d0_ll = d[0];
        long long d1_ll = d[1];
        long long r0_ll = r[0];
        long long r1_ll = r[1];

        long long r_lcm = lcm(r0_ll, r1_ll);

        long long low = d0_ll + d1_ll;
        long long high = 2LL * (d0_ll + d1_ll);

        long long ans = high;

        // 用二分搜尋找出最小可行時間
        while (low <= high) {
            long long mid = low + (high - low) / 2;

            long long f = mid;

            if (check(f, d0_ll, d1_ll, r0_ll, r1_ll, r_lcm)) {
                ans = f;
                high = f - 1;
            } else {
                low = f + 1;
            }
        }

        return ans;
    }
};
```
