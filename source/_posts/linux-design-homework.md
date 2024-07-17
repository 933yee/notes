---
title: Linux Design Homework
date: 2024-07-17 15:03:27
tags: linux
category: Linux
math: true
---

> https://wiki.csie.ncku.edu.tw/linux/schedule


## [2018q1 Homework (quiz4)](https://hackmd.io/@sysprog/linked-list-quiz)

### Quiz 1

```c
#include <stdlib.h>
#include <stdio.h>
struct node { int data; struct node *next, *prev; };

void FuncA(struct node **start, int value) {
    if (!*start) {
        struct node *new_node = malloc(sizeof(struct node));
        new_node->data = value;
        new_node->next = new_node->prev = new_node;
        *start = new_node;
        return;
    }
    struct node *last = (*start)->prev;
    struct node *new_node = malloc(sizeof(struct node));
    new_node->data = value;
    new_node->next = *start;
    (*start)->prev = new_node;
    new_node->prev = last;
    last->next = new_node;
}

void FuncB(struct node **start, int value) {
    struct node *last = (*start)->prev;
    struct node *new_node = malloc(sizeof(struct node));
    new_node->data = value;
    new_node->next = *start;
    new_node->prev = last;
    last->next = (*start)->prev = new_node;
    *start = new_node;
}

void FuncC(struct node **start, int value1, int value2) {
    struct node *new_node = malloc(sizeof(struct node));
    new_node->data = value1;
    struct node *temp = *start;
    while (temp->data != value2)
        temp = temp->next;
    struct node *next = temp->next;
    temp->next = new_node;
    new_node->prev = temp;
    new_node->next = next;
    next->prev = new_node;
}

void display(struct node *start) {
    struct node *temp = start;
    printf("Traversal in forward direction \n");
    for (; temp->next != start; temp = temp->next)
	    printf("%d ", temp->data);
    printf("%d ", temp->data);
    printf("\nTraversal in reverse direction \n");
    struct node *last = start->prev;
    for (temp = last; temp->prev != last; temp = temp->prev)
	printf("%d ", temp->data);
    printf("%d ", temp->data);
    printf("\n");
}

int main() {
    struct node *start = NULL;
    FuncA(&start, 51); FuncB(&start, 48);
    FuncA(&start, 72); FuncA(&start, 86);
    FuncC(&start, 63, 51);
    display(start);
    return 0;
}
```


#### FuncA 的作用是
```c
void FuncA(struct node **start, int value) {
    // 這是 circular 的 link list
    if (!*start) {
        // 初始 node，prev、next 都指向自己
        struct node *new_node = malloc(sizeof(struct node));
        new_node->data = value;
        new_node->next = new_node->prev = new_node;
        *start = new_node;
        return;
    }
    struct node *last = (*start)->prev; // 因為是 circular，start 的 prev 就是最後一個 node
    struct node *new_node = malloc(sizeof(struct node));
    new_node->data = value;
    new_node->next = *start; // 這裡可以發現新 node 的 next 是 start，所以是新加在 link list 的最後面
    (*start)->prev = new_node; 
    new_node->prev = last;
    last->next = new_node;
}
```

##### Ans: 
    (e) 建立新節點，內容是 value，並安插在結尾

#### FuncB 的作用是
``` c
void FuncB(struct node **start, int value) {
    struct node *last = (*start)->prev;
    struct node *new_node = malloc(sizeof(struct node));
    new_node->data = value;
    new_node->next = *start;
    new_node->prev = last; 
    last->next = (*start)->prev = new_node; // 到這邊是新加一個 node start 的 prev，last 的 next
    *start = new_node; // 更新 start，可以得知新 node 是新的 start，所以變成是加在 list 最前面
}
```

##### Ans: 
    (d) 建立新節點，內容是 value，並安插在開頭


#### FuncC 的作用是
``` c
void FuncC(struct node **start, int value1, int value2) {
    struct node *new_node = malloc(sizeof(struct node));
    new_node->data = value1;
    struct node *temp = *start;
    while (temp->data != value2) // 找到 value 為 value2 的 node
        temp = temp->next;
    // 把新加的 value1 的 node 加在 value2 的 node 後面
    struct node *next = temp->next;
    temp->next = new_node;
    new_node->prev = temp;
    new_node->next = next;
    next->prev = new_node;
}
```

##### Ans: 
    (e) 找到節點內容為 value2 的節點，並在之後插入新節點，內容為 value1

#### 在程式輸出中，訊息 Traversal in forward direction 後依序印出哪幾個數字呢？
``` c
int main() {
    struct node *start = NULL;
    FuncA(&start, 51); // 51
    FuncB(&start, 48); // 48 -> 51 
    FuncA(&start, 72); // 48 -> 51 -> 72
    FuncA(&start, 86); // 48 -> 51 -> 72 -> 86
    FuncC(&start, 63, 51); // 48 -> 51 -> 63 -> 72 -> 86
    display(start);
    return 0;
}
```

##### Ans: 
    48 -> 51 -> 63 -> 72 -> 86

#### 在程式輸出中，訊息 Traversal in reverse direction 後依序印出哪幾個數字呢？

##### Ans:
    86 -> 72 -> 63 -> 51 -> 48

### Quiz 2

``` c
#include <stdio.h>
#include <stdlib.h>

/* Link list node */
struct node { int data; struct node *next; };

int FuncX(struct node *head, int *data) {
    struct node *node;
    for (node = head->next; node && node != head; node = node->next)
        data++;
    return node - head;
}

struct node *node_new(int data) {
    struct node *temp = malloc(sizeof(struct node));
    temp->data = data; temp->next = NULL;
    return temp;
}

int main() {
    int count = 0;
    struct node *head = node_new(0);
    head->next = node_new(1);
    head->next->next = node_new(2);
    head->next->next->next = node_new(3);
    head->next->next->next->next = node_new(4);
    printf("K1 >> %s\n", FuncX(head, &count) ? "Yes" : "No");
    head->next->next->next->next = head;
    printf("K2 >> %s\n", FuncX(head, &count) ? "Yes" : "No");
    head->next->next->next->next->next = head->next;
    printf("K3 >> %s\n", FuncX(head, &count) ? "Yes" : "No");
    head->next = head->next->next->next->next->next->next->next->next;
    printf("K4 >> %s\n", FuncX(head, &count) ? "Yes" : "No");
    printf("K5 >> %d\n", head->next->data);
    printf("count >> %d\n", count);
    return 0;
}
```
#### FuncX 的作用是 (涵蓋程式執行行為的正確描述最多者)

##### Ans:
    (f) 判斷是否為 circular linked list，若為 circular 則回傳 0，其他非零值，過程中計算走訪的節點總數

``` c
int main() {
    int count = 0;
    struct node *head = node_new(0); // 0
    head->next = node_new(1); // 0 -> 1
    head->next->next = node_new(2); // 0 -> 1 -> 2
    head->next->next->next = node_new(3); // 0 -> 1 -> 2 -> 3
    head->next->next->next->next = node_new(4); // 0 -> 1 -> 2 -> 3 -> 4
    printf("K1 >> %s\n", FuncX(head, &count) ? "Yes" : "No"); // Yes
    head->next->next->next->next = head; // 0 -> 1 -> 2 -> 3 -> 0 -> ...
    printf("K2 >> %s\n", FuncX(head, &count) ? "Yes" : "No"); // No
    head->next->next->next->next->next = head->next; // 0 -> 1 -> 2 -> 3 -> 0 -> ...
    printf("K3 >> %s\n", FuncX(head, &count) ? "Yes" : "No"); // No
    head->next = head->next->next->next->next->next->next->next->next; // 0 -> 0 -> ...
    printf("K4 >> %s\n", FuncX(head, &count) ? "Yes" : "No"); // No
    printf("K5 >> %d\n", head->next->data); // 0
    printf("count >> %d\n", count); // 10
    return 0;
}
```


#### K1 >> 後面接的輸出為何

##### Ans:
    Yes

#### K2 >> 後面接的輸出為何

##### Ans:
    No

#### K3 >> 後面接的輸出為何

##### Ans:
    No

#### K4 >> 後面接的輸出為何

##### Ans:
    No

#### K5 >> 後面接的輸出為何

##### Ans:
    0

#### count >> 後面接的輸出為何

##### Ans:
    10

### 訂正
FuncX 的作用應為 `(e) 判斷是否為 circular linked list，若為 circular 則回傳 0，其他非零值`，沒辦法計算結點總數，因為它傳入的是 count 的 pointer，且在 FuncX 裡面是用 `data++` 而非 `(*data)++`，所以 main() 裡面的 count 應為 0


## [2020q1 第 1 週測驗題](https://hackmd.io/@sysprog/linux2020-quiz1)

### Quiz 1
``` c
typedef struct __list {
    int data;
    struct __list *next;
} list;

// 在不存在環狀結構的狀況下，以下函式能夠對 linked list 元素從小到大排序:
list *sort(list *start) {
    if (!start || !start->next)
        return start;
    list *left = start;
    list *right = left->next;
    LL0;

    left = sort(left);
    right = sort(right);

    for (list *merge = NULL; left || right; ) {
        if (!right || (left && left->data < right->data)) {
            if (!merge) {
                LL1;
            } else {
                LL2;
                merge = merge->next;
            }
            LL3;
        } else {
            if (!merge) {
                LL4;
            } else {
                LL5;
                merge = merge->next;
            }
            LL6;
        }
    }
    return start;
}
```


#### LL0 = ?

##### Ans:
    (a) left->next = NULL


#### LL1 = ?
    start = merge = left


#### LL2 = ?
    merge->next = left


#### LL3 = ?
    left = left->next


#### LL4 = ?
    start = merge = right


#### LL5 = ?
    merge->next = right

#### LL6 = ?
    right = right->next

- ### 解釋上述程式運作原理

    顯然這份程式碼是在做遞迴版的 Link list 的 `Insertion Sort`。

    當它分割成 left 和 right 兩份時，在運行最下方的 for loop 時 left 那份不能碰到 right 那份，兩者不應該重疊，因此可以合理推測 LL0 是 `left->next = NULL`。

    至於 for loop 做的事情只是單純一個一個 node 比大小而已，因此像是 `right 為空`或 `left 值 < right 值`時，就更新 merge->next 的值為 left，然後由於最後是 return start node，因此當 merge 為空時 (第一個比較的出來的 node)，也要設定 start node，所以 LL1 是 `start = merge = left`。

- ### 指出程式改進空間，特別是考慮到 [Optimizing merge sort](https://en.wikipedia.org/wiki/Merge_sort#Optimizing_merge_sort)

    由於是在這份程式碼中， left 永遠只會有一個 node，並將其插入 right 中的 list。但是 Insertion Sort 的時間複雜度為 $\Theta(n^2)$。

    最好優化的方式自然會聯想到 `Merge Sort`，其時間複雜度為 $\Theta(n log(n))$，只要在這邊去尋找 list 的中點，將其設為 right，也就是讓 left 和 right 的長度相近，一人一半。

- ### 將上述 singly-linked list 擴充為 circular doubly-linked list 並重新實作對應的 sort
- ### 依循 Linux 核心 include/linux/list.h 程式碼的方式，改寫上述排序程式
- ### 嘗試將原本遞迴的程式改寫為 iterative 版本