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

### 2025/02/13 [3066. Minimum Operations to Exceed Threshold Value II](https://leetcode.com/problems/minimum-operations-to-exceed-threshold-value-ii/description/?envType=daily-question&envId=2025-02-13)

- 用 `heap` 來記錄最小的值
  - Time Complexity $O(n\lg{n})$
  - Space Complexity $O(n)$

```cpp
class Solution {
public:
    int minOperations(vector<int>& nums, int k) {
        priority_queue<long long, vector<long long>, greater<long long>> pq(nums.begin(), nums.end());
        int ret = 0;
        while(!pq.empty()){
            long long a = pq.top();
            pq.pop();
            if(a >= k) break;
            long long b = pq.top();
            pq.pop();
            pq.push(min(a, b) * 2 + max(a, b));
            ret++;
        }
        return ret;
    }
};
```

### 2025/02/14 [1352. Product of the Last K Numbers](https://leetcode.com/problems/product-of-the-last-k-numbers/description/?envType=daily-question&envId=2025-02-14)

- 儲存最後是 0 的 idx，在 `getProduct` 的時候才不會除以 `0`。另外我原本是把 ` v.size()` 直接寫在 `if` 裡面，後來發現 `v.size()` 是 `unsigned` 的型態，算出負數會發生 `runtime error`。
  - Time Complexity $O(n)$
  - Space Complexity $O(n)$

```cpp
class ProductOfNumbers {
public:
    ProductOfNumbers() { }
    void add(int num) {
        if(num == 0){
            v.clear();
            v.push_back(1);
            lastIdx = v.size() - 1;
        }else{
            v.push_back(v.back() * num);
        }
    }
    int getProduct(int k) {
        int sz = v.size();
        if(sz - 1 - k < lastIdx) return 0;
        return v.back() / v[sz - 1 - k];
    }
    vector<int> v{1};
    int lastIdx = 0;
};

/**
 * Your ProductOfNumbers object will be instantiated and called as such:
 * ProductOfNumbers* obj = new ProductOfNumbers();
 * obj->add(num);
 * int param_2 = obj->getProduct(k);
 */
```

### 2025/02/15 [2698. Find the Punishment Number of an Integer](https://leetcode.com/problems/find-the-punishment-number-of-an-integer/description/?envType=daily-question&envId=2025-02-15)

暴力 DFS + 剪枝

```cpp
class Solution {
public:
    bool dfs(int cur, int n, int sum){
        if(cur == 0) return sum == n;
        for(int i = 10; i <= 1000000; i *= 10){
            if(dfs(cur / i, n, sum + cur % i))
                return true;
        }
        return false;
    }

    int punishmentNumber(int n) {
        int res = 0;
        for(int i = 1; i <= n; i++){
            if(dfs(i * i, i, 0))
                res += i * i;
        }
        return res;
    }
};
```

### 2025/02/16 [1718. Construct the Lexicographically Largest Valid Sequence](https://leetcode.com/problems/construct-the-lexicographically-largest-valid-sequence/description/?envType=daily-question&envId=2025-02-16)

暴力 DFS + 剪枝，每次放 2~n 的數字進 `vector` 時，可以一起更新 `idx + i` 的值，因為也只能放那邊

```cpp
class Solution {
public:
    vector<int> v, used;
    bool dfs(int n, int idx){
        if(v.size() == idx)
            return true;
        if(v[idx] != 0)
            return dfs(n, idx+1);
        for(int i=n; i>=1; i--){
            if(!used[i]){
                used[i] = true;
                if(i == 1){
                    v[idx] = 1;
                    if(dfs(n, idx+1))
                        return true;
                    v[idx] = 0;
                }else if(idx + i < v.size() && v[idx + i] == 0){
                    v[idx] = v[idx + i] = i;
                    if(dfs(n, idx+1))
                        return true;
                    v[idx] = v[idx + i] = 0;
                }
                used[i] = false;
            }
        }
        return false;
    }

    vector<int> constructDistancedSequence(int n) {
        used.resize(n+1, false);
        v.resize(n*2-1, 0);
        dfs(n, 0);
        return v;
    }
};
```

### 2025/02/17 [1079. Letter Tile Possibilities](https://leetcode.com/problems/letter-tile-possibilities/description/?envType=daily-question&envId=2025-02-17)

DFS，要加個 `seen` 確保不要在同個地方看同個字母，還有每次準備跑下個 `dfs()` 之前答案都要 `+1`

```cpp
class Solution {
public:
    int used[8] = {};
    int dfs(string& s, int cnt){
        if(cnt == s.size()) return 0;
        int seen[26] = {};
        int res = 0;
        for(int i=0; i<s.size(); i++){
            if(!used[i] && !seen[s[i] - 'A']){
                used[i] = 1;
                seen[s[i] - 'A'] = 1;
                res += 1 + dfs(s, cnt + 1);
                used[i] = 0;
            }
        }
        return res;
    }
    int numTilePossibilities(string tiles) {
        return dfs(tiles, 0);
    }
};
```

### 2025/02/18 [2375. Construct Smallest Number From DI String](https://leetcode.com/problems/construct-smallest-number-from-di-string/description/?envType=daily-question&envId=2025-02-18)

DFS，跟前一天差不多

```cpp
class Solution {
public:
    string res;
    int used[10] = {};
    bool dfs(string& pattern, int cnt){
        if(cnt == pattern.size() + 1) return true;
        for(int i=1; i<=pattern.size() + 1; i++){
            if(!used[i]){
                if(cnt >= 1 && pattern[cnt-1] == 'I' && (res.back() - '0') >= i) continue;
                if(cnt >= 1 && pattern[cnt-1] == 'D' && (res.back() - '0') <= i) continue;
                used[i] = 1;
                res.push_back(i + '0');
                if(dfs(pattern, cnt + 1))
                    return true;
                res.pop_back();
                used[i] = 0;
            }
        }
        return false;
    }

    string smallestNumber(string pattern) {
        dfs(pattern, 0);
        return res;
    }
};
```

### 2025/02/19 [1415. The k-th Lexicographical String of All Happy Strings of Length n](https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/description/?envType=daily-question&envId=2025-02-19)

DFS，確保不要重複，且在找到第 k 個時就可以直接 return

```cpp
class Solution {
public:
    string str = "abc";
    bool dfs(int& n, int& k, int& k_cnt, string& ret){
        if(ret.size() == n){
            k_cnt++;
            return k_cnt == k;
        }

        for(char& c:str){
            if(!ret.empty() && ret.back() == c) continue;
            ret.push_back(c);
            if(dfs(n, k, k_cnt, ret))
                return true;
            ret.pop_back();
        }
        return false;
    }

    string getHappyString(int n, int k) {
        string ret;
        int k_cnt = 0;
        dfs(n, k, k_cnt, ret);
        return ret;
    }
};
```

### 2025/02/20 [1980. Find Unique Binary String](https://leetcode.com/problems/find-unique-binary-string/description/?envType=daily-question&envId=2025-02-20)

`n` 最大到 16，應該也可以窮舉。這裡是直接讓每個 `num` 在不同位置至少有一個不同的位元，就能確保不會有重複的字串。

```cpp
class Solution {
public:
    string findDifferentBinaryString(vector<string>& nums) {
        string res;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i][i] == '0')
                res.push_back('1');
            else
                res.push_back('0');
        }
        return res;
    }
};

```

### 2025/02/21 [1261. Find Elements in a Contaminated Binary Tree](https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree/description/?envType=daily-question&envId=2025-02-21)

好像有沒有 recover 都沒差，只要把所有的值都存起來就好(X

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class FindElements {
public:
    unordered_set<int> st;
    FindElements(TreeNode* root) {
        recover(root, 0);
    }

    void recover(TreeNode* node, int new_val){
        if(node == NULL) return;
        st.insert(new_val);
        recover(node->left, new_val * 2 + 1);
        recover(node->right, new_val * 2 + 2);
    }

    bool find(int target) {
        return st.find(target) != st.end();
    }
};

/**
 * Your FindElements object will be instantiated and called as such:
 * FindElements* obj = new FindElements(root);
 * bool param_1 = obj->find(target);
 */
```

### 2025/02/22 [1028. Recover a Tree From Preorder Traversal](https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/description/?envType=daily-question&envId=2025-02-22)

題目給的 input 是 `DFS preorder` 的結果，所以可以用 stack 還原回去。每次都要檢查 `depth` 的值 (就是 dash 的數量) 直到深度差一層為止，那就是 parent。

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* recoverFromPreorder(string traversal) {
        stack<TreeNode*> stk;
        stack<int> depth_stk;
        TreeNode* head = new TreeNode(0);
        stk.push(head);
        depth_stk.push(-1);
        int i = 0;

        while(i < traversal.size()){
            int num_dashes = 0, j = i, cur_num = 0;
            while(j < traversal.size() && traversal[j] == '-'){
                num_dashes++;
                j++;
            }

            while(j < traversal.size() && traversal[j] != '-'){
                cur_num = (cur_num * 10) + (traversal[j] - '0');
                j++;
            }

            while(depth_stk.top() + 1 != num_dashes){
                depth_stk.pop();
                stk.pop();
            }

            TreeNode* parent = stk.top();
            TreeNode* newNode = new TreeNode(cur_num);
            if(parent->left == nullptr)
                parent->left = newNode;
            else
                parent->right = newNode;

            stk.push(newNode);
            depth_stk.push(num_dashes);
            i = j;
        }
        return head->left;
    }
};
```

### 2025/02/23 [889. Construct Binary Tree from Preorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/description/?envType=daily-question&envId=2025-02-23)

從 `preorder` 開始遍歷，每個值都去 `postorder` 的位置往右找，第一個已經用過的就是 parent node

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* constructFromPrePost(vector<int>& preorder, vector<int>& postorder) {
        int n = preorder.size();
        vector<TreeNode*> used(n, nullptr);
        unordered_map<int, int> idx_mp;
        for(int i=0; i<n; i++) idx_mp[postorder[i]] = i;

        TreeNode* head = new TreeNode(preorder[0]);
        used.back() = head;

        for(int i=1; i<n; i++){
            int cur_val = preorder[i];
            int cur_idx = idx_mp[cur_val];
            TreeNode* newVal = new TreeNode(cur_val);
            used[cur_idx] = newVal;
            for(int j=cur_idx+1; j<n; j++){
                if(used[j] != nullptr){
                    if(used[j]->left == nullptr)
                        used[j]->left = newVal;
                    else
                        used[j]->right = newVal;
                    break;
                }
            }
        }
        return head;
    }
};
```

### 2025/02/25 [1524. Number of Sub-arrays With Odd Sum](https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum/description/?envType=daily-question&envId=2025-02-25)

記錄當前全部的總和、prefix sum 的奇數和偶數個數。如果當前的總和是偶數，代表它扣掉前面 prefix sum 的奇數個數就是 `包含當前數字的 subarray sum 為奇數的個數`，反之，如果當前的總和是奇數，代表它扣掉前面 prefix sum 的偶數個數就是 `包含當前數字的 subarray sum 為奇數的個數`，但是要加上 `1`，因為自己本身也是一個 subarray。

```cpp
class Solution {
public:
    int numOfSubarrays(vector<int>& arr) {
        long long cur_sum = 0, odd = 0, even = 0, res = 0;
        for(int& i:arr){
            cur_sum += i;
            if(cur_sum & 1){
                res += 1 + even;
                odd++;
            } else{
                res += odd;
                even++;
            }
        }
        return res % 1000000007;
    }
};
```

### 2025/03/01 [2460. Apply Operations to an Array](https://leetcode.com/problems/apply-operations-to-an-array/description/?envType=daily-question&envId=2025-03-01)

可以在初始化的時候就先塞好一堆 0，`vector<int> ret(n, 0);`

```cpp
class Solution {
public:
    vector<int> applyOperations(vector<int>& nums) {
        int n = nums.size(), cnt = 0;
        vector<int> ret(n, 0);
        for(int i=0; i<n-1; i++){
            if(nums[i] == nums[i+1]){
                nums[i] *= 2;
                nums[i+1] = 0;
            }
        }

        for(int i=0, j=0; i<n; i++)
            if(nums[i] != 0)
                ret[j++] = nums[i];

        return ret;
    }
};
```

### 2025/03/02 [2570. Merge Two 2D Arrays by Summing Values](https://leetcode.com/problems/merge-two-2d-arrays-by-summing-values/description/?envType=daily-question&envId=2025-03-02)

直接用 hash 存起來

```cpp
class Solution {
public:
    vector<vector<int>> mergeArrays(vector<vector<int>>& nums1, vector<vector<int>>& nums2) {
        map <int, int> mp;
        for(auto& i:nums1) mp[i[0]] = i[1];
        for(auto& i:nums2){
            if(mp.find(i[0]) != mp.end())
                mp[i[0]] += i[1];
            else
                mp[i[0]] = i[1];
        }
        vector<vector<int>> ret;
        for(auto& i:mp)
            ret.push_back({i.first, i.second});
        return ret;
    }
};
```

因為兩個 vector 都是根據 id sorted，所以可以直接用 `two pointer` 的方式來做

```cpp
class Solution {
public:
    vector<vector<int>> mergeArrays(vector<vector<int>>& nums1, vector<vector<int>>& nums2) {
        int n = nums1.size(), m = nums2.size();
        int idx1 = 0, idx2 = 0;
        vector<vector<int>> ret;
        while(idx1 < n || idx2 < m){
            if(idx1 < n && idx2 < m){
                if(nums1[idx1][0] < nums2[idx2][0]){
                    ret.push_back(nums1[idx1]);
                    idx1++;
                } else if(nums1[idx1][0] > nums2[idx2][0]){
                    ret.push_back(nums2[idx2]);
                    idx2++;
                } else{
                    ret.push_back({nums1[idx1][0], nums1[idx1][1] + nums2[idx2][1]});
                    idx1++; idx2++;
                }
            } else if(idx1 < n){
                ret.push_back(nums1[idx1]);
                idx1++;
            } else{
                ret.push_back(nums2[idx2]);
                idx2++;
            }
        }
        return ret;
    }
};
```

### 2025/03/03 [2161. Partition Array According to Given Pivot](https://leetcode.com/problems/partition-array-according-to-given-pivot/description/?envType=daily-question&envId=2025-03-03)

用兩個 index 來記錄小於 pivot 和大於 pivot 的位置，最後再把剩下的補上

```cpp
class Solution {
public:
    vector<int> pivotArray(vector<int>& nums, int pivot) {
        int n = nums.size();
        vector<int> ret(n);
        int cnt1 = 0, cnt2 = 0;
        for(int i=0; i<n; i++){
            if(nums[i] < pivot) cnt1++;
            else if(nums[i] > pivot) cnt2++;
        }
        int idx1 = 0, idx2 = n - cnt2;
        for(int i=cnt1; i<idx2; i++) ret[i] = pivot;
        for(int i=0; i<n; i++){
            if(nums[i] < pivot) ret[idx1++] = nums[i];
            else if(nums[i] > pivot) ret[idx2++] = nums[i];
        }
        return ret;
    }
};
```

### 2025/03/04 [1780. Check if Number is a Sum of Powers of Three](https://leetcode.com/problems/check-if-number-is-a-sum-of-powers-of-three/description/?envType=daily-question&envId=2025-03-04)

最直覺的方法是直接 DFS 爆開，因為 `n` 最大只到 `10^7`，最多只會到 $3^15$

```cpp
class Solution {
public:
    vector<int> tmp{1};
    bool dfs(int idx, int cur){
        if(cur < 0 || idx > 15) return false;
        if(cur == 0) return true;
        return dfs(idx+1, cur-tmp[idx]) || dfs(idx+1, cur);
    }

    bool checkPowersOfThree(int n) {
        for(int i=0; i<15; i++)
            tmp.push_back(tmp.back() * 3);
        return dfs(0, n);
    }
};
```

討論區看到一個很猛的方法，直接把 `n` 看成 3 進位，每次檢查 LSB，如果有出現 `2` 就代表不是由 `3^x` 組成的

```cpp
class Solution {
public:
    bool checkPowersOfThree(int n) {
        while(n > 0){
            if(n % 3 == 2) return false;
            n /= 3;
        }
        return true;
    }
};
```

### 2025/03/05 [2579. Count Total Number of Colored Cells](https://leetcode.com/problems/count-total-number-of-colored-cells/description/?envType=daily-question&envId=2025-03-05)

直接算兩個三角形去掉重疊的部分

```cpp
class Solution {
public:
    long long coloredCells(int n) {
        return (long long)2*n*n-(2*n-1);
    }
};
```

也可以用遞迴

```cpp
class Solution {
public:
    long long coloredCells(int n) {
        if(n == 1) return 1;
        return (n-2) * 4 + 4 + coloredCells(n-1);
    }
};
```

### 2025/03/06 [2965. Find Missing and Repeated Values](https://leetcode.com/problems/find-missing-and-repeated-values/description/?envType=daily-question&envId=2025-03-06)

略

```cpp
class Solution {
public:
    vector<int> findMissingAndRepeatedValues(vector<vector<int>>& grid) {
        int n = grid.size();
        vector<int> rec(n * n + 1), ans(2);
        for(auto& v:grid){
            for(auto& i:v)
                rec[i]++;
        }
        for(int i=1; i<=n*n; i++){
            if(rec[i] == 0) ans[1] = i;
            else if(rec[i] == 2) ans[0] = i;
        }
        return ans;
    }
};
```
