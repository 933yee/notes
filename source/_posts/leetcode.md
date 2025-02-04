---
title: 不知道這會持續幾天的 leetcode 100
date: 2025-01-31 21:22:11
tags: leetcode
category:
---

## Arrays & Hashing

### [217. Contains Duplicate](https://leetcode.com/problems/contains-duplicate/description/)

1. Hash map
   - Time Complexity $O(n)$
   - Space Complexity $O(n)$

```cpp
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_map<int, int> mp;
        for(int& i:nums){
            if(mp.count(i) != 0) return true;
            mp[i]++;
        }
        return false;
    }
};
```

2. 更簡短的寫法

```cpp
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_set<int> st(nums.begin(), nums.end());
        return st.size() < nums.size();
    }
};
```

### [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/description/)

- 算出兩者 a ~ z 的數量存到不同 vector，最後直接比較兩個 vector 是否一樣 (其實就 Hash map)
  - Time Complexity $O(n + m)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        vector<int> s_cnt(26), t_cnt(26);
        for(char& c:s) s_cnt[c - 'a']++;
        for(char& c:t) t_cnt[c - 'a']++;
        return s_cnt == t_cnt;
    }
};
```

### [1. Two Sum](https://leetcode.com/problems/two-sum/description/)

1. Hash map
   - Time Complexity $O(n)$
   - Space Complexity $O(n)$

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> mp;
        for(int i=0; i<nums.size(); i++){
            int k = target - nums[i];
            if(mp.count(k))
                return vector<int>{i, mp[k]};
            mp[nums[i]] = i;
        }
        return vector<int>(0);
    }
};
```

2. 更簡短的寫法
   - `vector<int>` 可以省略耶 (wow

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> mp;
        for(int i=0; i<nums.size(); i++){
            int k = target - nums[i];
            if(mp.count(k))
                return {i, mp[k]};
            mp[nums[i]] = i;
        }
        return {};
    }
};
```

### [49. Group Anagrams](https://leetcode.com/problems/group-anagrams/description/)

- 計算每個 string a ~ z 的數量，比較先前存的結果，沒有匹配的話就塞到新的 vector (超慢
  - Time Complexity $O(n * m)$
  - Space Complexity $O(n)$

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<int>> str_cnt;
        vector<vector<string>> ret;
        for(string& s:strs){
            vector<int> tmp_cnt(26);
            for(char& c:s) tmp_cnt[c - 'a']++;
            bool v_same = false;
            for(int i=0; i<str_cnt.size(); i++){
                if(str_cnt[i] == tmp_cnt){
                    ret[i].push_back(s);
                    v_same = true;
                    break;
                }
            }
            if(!v_same){
                str_cnt.push_back(tmp_cnt);
                ret.push_back({s});
            }
        }
        return ret;
    }
};
```

- 用 hash map，把 sort 過的 string 當作 key
  - Time Complexity $O(n * \mlg(m))$
  - Space Complexity $O(n * m)$

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> ret;
        unordered_map<string, int> mp;
        for(string s:strs){
            string sorted_s = s;
            sort(sorted_s.begin(), sorted_s.end());
            if(mp.count(sorted_s))
                ret[mp[sorted_s]].push_back(s);
            else{
                mp[sorted_s] = ret.size();
                ret.push_back({s});
            }
        }
        return ret;
    }
};
```

### [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/)

1. 先存進 map ，再遍歷 map 的值，丟進 priority queue 裡面取出前 k 個 (快忘光 cmp 的語法了 qq，也可以直接用 greater
   - Time Complexity $O(n\lg{n})$
   - Space Complexity $O(n)$

```cpp
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        for(int& i:nums) mp[i]++;

        auto cmp = [](pair<int, int>& a, pair<int, int>& b) {
            return a.second < b.second;
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pq;
        for(auto& i:mp)
            pq.push(i);

        vector<int> ret;
        while(k--){
            ret.push_back(pq.top().first);
            pq.pop();
        }
        return ret;
    }
};
```

2. heap 那邊可以做一些優化，可以從小排到大，然後確保每次操作 heap 裡面都只有 k 個
   - Time Complexity $O(n\lg{k})$
   - Space Complexity $O(n)$

```cpp
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        for(int& i:nums) mp[i]++;

        auto cmp = [](pair<int, int>& a, pair<int, int>& b) {
            return a.second > b.second;
        };
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(cmp)> pq;
        for(auto& i:mp){
            pq.push(i);
            if(pq.size() > k)
                pq.pop();
        }

        vector<int> ret;
        while(k--){
            ret.push_back(pq.top().first);
            pq.pop();
        }
        return ret;
    }
};
```

### [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)

- 開兩個 vector 記錄左右兩邊一路乘過去的值，最後兩者相乘
  - Time Complexity $O(n)$
  - Space Complexity $O(n)$

```cpp
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> L(n+2), R(n+2), ret(n);
        L[0] = L[1] = R[n] = R[n+1] = 1;
        for(int i=0; i<n; i++){
            L[i+2] = L[i+1] * nums[i];
            R[n-1-i] = R[n-i] * nums[n-1-i];
        }
        for(int i=0; i<n; i++)
            ret[i] = L[i+1] * R[i+1];
        return ret;
    }
};
```

### [36. Valid Sudoku](https://leetcode.com/problems/valid-sudoku/description/)

- 檢查每個 row、column、九宮格的情況
  - Time Complexity $O(1)$
  - Space Complexity $O(1)$

```cpp
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        for(int i=0; i<9; i++){
            vector<int> seen(9);
            for(int j=0; j<9; j++){
                char ch = board[i][j];
                if(ch == '.') continue;
                if(seen[ch-'1']) return false;
                ++seen[ch-'1'];
            }
        }
        for(int i=0; i<9; i++){
            vector<int> seen(9);
            for(int j=0; j<9; j++){
                char ch = board[j][i];
                if(ch == '.') continue;
                if(seen[ch-'1']) return false;
                ++seen[ch-'1'];
            }
        }
        for(int i=0; i<9; i++){
            vector<int> seen(9);
            for(int j=0; j<9; j++){
                char ch = board[i/3*3+j/3][i%3*3+j%3];
                if(ch == '.') continue;
                if(seen[ch-'1']) return false;
                ++seen[ch-'1'];
            }
        }
        return true;
    }
};
```

### [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/description/)

- 檢查每個數字左右兩邊連續數字的數量，檢查過的要刪掉避免重複檢查
  - Time Complexity $O(n)$
  - Space Complexity $O(n)$

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> st(nums.begin(), nums.end());
        int ret = 0;
        for(int i:nums){
            if(st.count(i) == 0) continue;
            int cnt = 1; int cur = i;
            st.erase(cur);
            int temp = cur + 1;
            while(st.count(temp)){
                ++cnt;
                st.erase(temp++);
            }
            temp = cur - 1;
            while(st.count(temp)){
                ++cnt;
                st.erase(temp--);
            }
            ret = max(ret, cnt);
        }
        return ret;
    }
};
```

## Two Pointers

### [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/description/)

- 用左右兩個指針收縮，檢查是不是 Palindrome

```cpp
class Solution {
public:
    bool isPalindrome(string s) {
        int l = 0, r = s.size() - 1;
        while(l < r){
            while(!isalnum(s[l]) && l < r) l++;
            while(!isalnum(s[r]) && l < r) r--;
            if(l < r && tolower(s[l++]) != tolower(s[r--]))
                return false;
        }
        return true;
    }
};
```
