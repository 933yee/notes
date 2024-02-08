---
title: Time Complexity
date: 2023-01-17 16:17:00
categories: [Algorithm, Time Complexity]
tags: [Time Complexity, Algorithm] 
math: true
---

![Big-O Complexity Chart](https://miro.medium.com/v2/resize:fit:1400/1*5ZLci3SuR0zM_QlZOADv8Q.jpeg)

## **Big-O ($O$)**

### **Definition**
- f(n) = $O$(g(n)) iff $\exists$ <span style="color:yellow">  c, n<sub>0</sub> > 0 </span> such that<span style="color:yellow"> f(n)$\le$ c $\cdot$ g(n) </span> $\forall$ <span style="color:yellow"> n $\ge$ n<sub>0</sub> </span>

### **Examples**
  - 3n+2 = $O$(n)
    : When c=4, n<sub>0</sub> = 2, 3n+2 $\le$ 4n for all n $\ge$ 2.

  - 100n+6 = $O$(n)
    : When c=101, n<sub>0</sub> = 6, 100n+6 $\le$ 101n for all n $\ge$ 6. 

  - 10n<sup>2</sup>+4n+2 = $O$(n<sup>2</sup>)
    : When c=11, n<sub>0</sub> = 5, 10n<sup>2</sup>+4n+2 $\le$ 11n<sup>2</sup> for all n $\ge$ 5. 

- ### **Properties**
    - f(n) = $O$(g(n)) states that $O$(g(n)) is an <span style="color:yellow"> upper bound </span> of f(n), so n = $O$(n) = $O$(n<sup>2.5</sup>) = $O$(n<sup>3</sup>) = $O$(n<sup>n</sup>). However, we want g(n) <span style="color:yellow"> as small as possible </span>.
    - Big-O refers to <span style="color:yellow"> worst-case running time </span> of a program.


## **Big-Omega($\Omega$)**

### **Definition**
  - f(n) = $\Omega$(g(n)) iff $\exists$ <span style="color:yellow">  c, n<sub>0</sub> > 0 </span> such that <span style="color:yellow">f(n)$\ge$ c $\cdot$ g(n) </span> $\forall$ <span style="color:yellow"> n  $\ge$ n<sub>0</sub> </span>. 

### **Examples**
  - 3n+2 = $\Omega$(n)
    : When c=3, n<sub>0</sub> = 1, 3n+2 $\ge$ 3n $\forall$ n $\ge$ 1.

  - 100n+6 = $\Omega$(n)
    : When c=100, n<sub>0</sub> = 1, 100n+6 $\ge$ 100n $\forall$ n $\ge$ 1.  

  - 10n<sup>2</sup>+4n+2 = $\Omega$(n<sup>2</sup>)
    : When c=1, n<sub>0</sub> = 1, 10n<sup>2</sup>+4n+2 $\ge$ n<sup>2</sup> $\forall$ n $\ge$ 1.

### **Properties**
  - f(n) = $\Omega$(g(n)) states that $\Omega$(g(n)) is a <span style="color:yellow"> lower bound </span> of f(n).
  - $\Omega$ refers to <span style="color:yellow"> best-case running time </span> of a program.

## **Big-Theta($\theta$)**

### **Definition**
  - f(n) = $\theta$(g(n)) iff <span style="color:yellow"> f(n) = $O$(g(n)) </span> and <span style="color:yellow"> f(n) = $\Omega$(g(n))</span>.

### **Examples**
  - 3n+2 = $\theta$(n)
  - 100n+6 = $\theta$(n)
  - 10n<sup>2</sup>+4n+2 = $\theta$(n<sup>2</sup>)

### **Properties**
  - f(n) = $\theta$(g(n)) states that $\theta$(g(n)) is a <span style="color:yellow"> tight bound </span> of f(n).
  - $\theta$ refers to <span style="color:yellow"> average-case running time </span >of a program.


## **Cheat Sheets**

![Data Structure Operations](https://pic4.zhimg.com/80/v2-bea9f0ddbc2d810e9feba3f3cc8b2b7f_720w.webp)
![Array Sorting](https://pic4.zhimg.com/80/v2-c9074ce39abbdebd1120451bf657e67f_720w.webp)


## 參考資料
- [Big-O Algorithm Complexity Cheat Sheet](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjr647gjoiEAxX2hq8BHQilDfMQFnoECBEQAQ&url=https%3A%2F%2Fwww.bigocheatsheet.com%2F&usg=AOvVaw0j8XV1sZ0vh9PgRFBYyAHO&opi=89978449)