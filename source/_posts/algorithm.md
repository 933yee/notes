---
title: Algorithm
date: 2025-10-24 11:22:00
tags: [algorithm]
category: Algorithm
math: true
---

## Moore’s Voting Algorithm 摩爾投票法

找出一個序列中出現次數超過一半的元素。 (majority element)

```python
candidate = None
count = 0
for num in nums:
    if count == 0:
        candidate = num
    if num == candidate:
        count += 1
    else:
        count -= 1
```

## 龜兔賽跑演算法 / 快慢指標法 (Floyd's Tortoise and Hare)

| 類型                    | 解釋                                                                          | 範例                      |
| ----------------------- | ----------------------------------------------------------------------------- | ------------------------- |
| **Cycle Detection**     | 判斷 linked list 是否有環。若有，`slow` 和 `fast` 最終會相遇。                | `Floyd’s Cycle Detection` |
| **Middle Node Finding** | 當 `fast` 走兩步、`slow` 走一步時，當 `fast` 到終點，`slow` 在中間。          | 找中間節點                |
| **Nth Node from End**   | `fast` 先走 N 步，再讓 `slow` 一起走，`fast` 到尾時，`slow` 就在倒數第 N 個。 | 找倒數第 k 個節點         |
| **List Intersection**   | 判斷兩條 linked list 是否交會。                                               | 判斷是否有共同節點        |

### Linked List Cycle

[Leetcode 141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

```c++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast && fast->next){
            slow = slow->next;
            fast = fast->next->next;
            if(slow == fast) return true;
        }
        return false;
    }
};
```

快指針每次走兩步，慢指針每次走一步。如果有環，兩者在環中的相對距離會每次減少 1，最終相遇。

### Middle of the Linked List

[Leetcode 143. Reorder List](https://leetcode.com/problems/reorder-list/)

```c++
class Solution {
public:
    void reorderList(ListNode* head) {
        // find the middle node
        ListNode* slow = head, * fast = head;
        while(fast && fast->next){
            fast = fast->next->next;
            slow = slow->next;
        }

        // reverse
        ListNode* cur = slow->next, * prev = nullptr;
        slow->next = nullptr;
        while(cur){
            ListNode* tmp = cur->next;
            cur->next = prev;
            prev = cur;
            cur = tmp;
        }
        // combine two linked list
        ListNode* l1 = head;
        ListNode* l2 = prev;
        while(l1 && l2){
            ListNode* n1 = l1->next;
            ListNode* n2 = l2->next;
            l1->next = l2;
            if(n1 == nullptr) break;
            l2->next = n1;
            l1 = n1;
            l2 = n2;
        }
    }
};
```

藉由快慢指標快速找到中間節點後，將後半段 linked list 反轉，最後將前後兩段交錯合併即可。

### Remove Nth Node From End of List

[Leetcode 19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(0, head);
        ListNode* slow = dummy, * fast = dummy;
        for(int i = 0; i <= n; i++)
            fast = fast->next;
        while(fast){
            fast = fast->next;
            slow = slow->next;
        }
        slow->next = slow->next->next;
        return dummy->next;
    }
};
```

可以使用一個虛擬節點 `dummy` 指向 head，讓 `fast` 先走 n+1 步，然後 `slow` 和 `fast` 一起走，當 `fast` 到尾時，`slow` 就在倒數第 n 個節點的前一個節點，將其刪除即可。
