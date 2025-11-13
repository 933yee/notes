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
public:
    long long gcd(long long a, long long b){
        while(b){
            a %= b;
            swap(a, b);
        }
        return a;
    }

    long long lcm(long long a, long long b){
        if(a == 0 || b == 0) return 0;
        return a / gcd(a, b) * b;
    }

    bool valid(long long t, vector<long long>& lld, vector<long long>& llr, long long rlcm){
        long long r0 = t / llr[0];
        long long r1 = t / llr[1];
        long long r_both = t / rlcm;

        long long slot0 = r1 - r_both;
        long long slot1 = r0 - r_both;
        long long slot_both = t - r0 - r1 + r_both;

        long long remaining0 = max(lld[0] - slot0, (long long) 0);
        long long remaining1 = max(lld[1] - slot1, (long long) 0);
        return (remaining0 + remaining1) <= slot_both;
    }

    long long minimumTime(vector<int>& d, vector<int>& r) {
        vector<long long> lld(d.begin(), d.end());
        vector<long long> llr(r.begin(), r.end());

        long long left = d[0] + d[1];
        long long right = 2 * left;

        long long ans = right;
        long long rlcm = lcm(llr[0], llr[1]);
        while(left <= right){
            long long m = left + (right - left) / 2;
            if(valid(m, lld, llr, rlcm)){
                right = m - 1;
                ans = m;
            }else
                left = m + 1;
        }
        return ans;
    }
};
```

### 3734. Lexicographically Smallest Palindromic Permutation Greater Than Target

給定一個字串 s 和 target，找出 s 的所有回文排列中，比 target 字典序更大的最小回文排列，若無則回傳空字串。

```c++
class Solution {
public:
    string lexPalindromicPermutation(string s, string target) {
        int left[26]{};
        for (char b : s) {
            left[b - 'a']++;
        }
        auto valid = [&]() -> bool {
            for (int c : left) {
                if (c < 0) {
                    return false;
                }
            }
            return true;
        };

        string mid_ch;
        // 先檢查 s 能不能組成回文，順便把中間字元找出來
        for (int i = 0; i < 26; i++) {
            int c = left[i];
            if (c % 2 == 0) {
                continue;
            }
            if (!mid_ch.empty()) {
                return "";
            }
            mid_ch = 'a' + i;
            left[i]--;
        }

        int n = s.size();
        // 先假設能夠組成和 target 左半邊相同的字串 (不含中間字元)
        for (int i = 0; i < n / 2; i++) {
            left[target[i] - 'a'] -= 2;
        }

        if (valid()) {
            // 特殊情況，如果合理且 s 右半邊 (包含中間字元) 比 target 大，直接回傳
            string right_s = target.substr(0, n / 2);
            ranges::reverse(right_s);
            right_s = mid_ch + right_s;
            if (right_s > target.substr(n / 2)) {
                return target.substr(0, n / 2) + right_s;
            }
        }

        // 不能的話就從中間開始往前找，嘗試把字元換成更大的字元
        for (int i = n / 2 - 1; i >= 0; i--) {
            int b = target[i] - 'a';
            left[b] += 2;
            if (!valid()) {
                continue;
            }

            for (int j = b + 1; j < 26; j++) {
                if (left[j] == 0) {
                    continue;
                }

                left[j] -= 2;
                target.resize(i + 1);
                target[i] = 'a' + j;

                for (int k = 0; k < 26; k++) {
                    target += string(left[k] / 2, 'a' + k);
                }

                string right_s = target;
                ranges::reverse(right_s);
                target += mid_ch;
                target += right_s;

                return target;
            }
        }
        return "";
    }
};
```
