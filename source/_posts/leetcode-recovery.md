---
title: Leetcode 復健 - Neetcode 150
date: 2024-09-07 15:53:12
tags: 
category: 
math: true
---

# Arrays & Hashing

## Contains Duplicate - Easy
```cpp
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_map<int, int> mp;
        for(int& i:nums){
            if(mp.count(i) != 0){
                return true;
            }
            mp[i]++;
        }
        return false;
    }
};
```

## Valid Anagram - Easy
```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        int s_count[26] = {};
        int t_count[26] = {};
        for(char& c:s) s_count[c - 'a']++;
        for(char& c:t) t_count[c - 'a']++;
        for(int i=0; i<26; i++){
            if(s_count[i] != t_count[i]) 
                return false;
        }
        return true;
    }
};
```

## Two Sum - Easy
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> mp;
        int size = nums.size();
        for(int i=0; i<size; i++){
            int complement = target - nums[i];
            if(mp.count(complement) != 0){
                return {mp[complement], i};
            }
            mp[nums[i]] = i;
        }
        return {};
    }
};
```

## Group Anagrams - Medium

對於每個 string 都去計算一次 a ~ z 的數量，算好後看之前有沒有出現過

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
         vector<pair<vector<int>, int>> rec;
         vector<vector<string>> ans;
         for(string& s:strs){
            vector<int> count(26);
            for(char& c:s) count[c-'a']++;
            bool found = false;
            for(auto& i:rec){
                if(i.first == count){
                    found = true;
                    ans[i.second].push_back(s);
                    break;
                }
            }
            if(!found){
                rec.push_back({count, ans.size()});
                ans.push_back(vector<string>{s});
            }
         }
        return ans;
    }
};
```

雖然用 a ~ z 的 array 存會比直接 sort 較快，但往前找也會花很多時間，還不如直接 sort 用 hashmap 找就好

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> ans;
        unordered_map<string, vector<string>> mp;
        for (string& s : strs) {
            string s_copy = s;
            sort(s_copy.begin(), s_copy.end());
            mp[s_copy].push_back(s);
        }
        for(auto& i:mp) ans.push_back(i.second);
        return ans;
    }
};
```

## Top K Frequent Elements - Medium

先用 hashmap 存起來，再倒到 priority queue 裡面，最後 pop k 個
```cpp
class Solution {
public:

    struct cmp {
        bool operator()(pair<int, int> a, pair<int, int> b) {
            return a.second < b.second;
        }
    };


    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        vector<int> ans;
        for(int& i:nums) mp[i]++;
        priority_queue<pair<int, int>, vector<pair<int, int>>, cmp> pq;
        for(auto& i:mp){
            pq.push({i.first, i.second});
        }
        while(k--){
            ans.push_back(pq.top().first);
            pq.pop();
        }
        return ans;
    }
};
```

因為是要計算出現的頻率，且頻率絕對不會超過 nums 的 size，因此第二步可以改成用 vector 來存，反正空間一定夠。最後再從後往前找 k 個即可。

```cpp
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        vector<int> ans;
        for(int& i:nums) mp[i]++;
        vector<vector<int>> count(nums.size() + 1);
        for(auto& i:mp) count[i.second].push_back(i.first);
        for(int i=count.size()-1; i>=0; i--){
            if(k == 0) break;
            for(int j=0; j<count[i].size(); j++){
                ans.push_back(count[i][j]);
                k--;
            }
        }
        return ans;
    }
};
```

## Product of Array Except Self - Medium
```cpp
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> L{1}, R{1}, ans;
        int n = nums.size();

        for(int i = 0; i < n; i++)
            L.push_back(L.back() * nums[i]);

        for(int i = n - 1; i >= 0; i--)
            R.push_back(R.back() * nums[i]);

        reverse(R.begin(), R.end());

        for(int i = 0; i < n; i++)
            ans.push_back(L[i] * R[i+1]);

        return ans;
    }
};
```