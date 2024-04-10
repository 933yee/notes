---
title: Unity 新手教學 - 介面介紹和基本操作
date: 2024-04-10 14:58:04
tags: Unity
---

在這次教學中，我們將學習了解 Unity 的開發者介面，以及如何新增物件還有編輯的快捷鍵。

## Unity 介面介紹
在 Unity Editor 中，開發者的介面分割得非常明確，大致可以將畫面分成幾個區塊，每個區塊都有特定的功能

![Unity Editor](./images/unity-tutorial-2/UnityEditor.png)

### Hierarchy View
預設位於介面的左上方，這裡顯示了場景中的所有物體的層次結構。你可以在這裡看到物體之間的父子關係，還有它們的名稱。

在預設場景中，Unity 會幫你新增兩個物件，Main Camera 和 Direction Light。Camera 可以想像成有人扛著攝影機負責轉播現在遊戲的情況，所以玩家看到的遊戲畫面都至少有一個 Camera 負責。Direction Light 則負責場景中的光照。

#### 補充
- 請記得把所有物件都以英文命名，不然在程式碼中尋找中文名稱的物件時，極有可能發生找不到的問題
- 請把所有物件的名稱都命名的淺顯易懂，最好是一看到名字就知道這個東西在幹嘛，不然日後會被自己害死... (親身經驗


### Scene View
預設位於介面的中間，這是遊戲的可視化界面，可以在這裡編輯和調整場景中的物體

### Game View
預設位於介面的中間，這裡顯示遊戲的運行結果，也就是你在 Scene View 更改的結果，可以在這邊做簡單的 Demo，看一下目前遊戲長什麼樣子

### Inspector View
預設位於介面的右邊，這裡顯示的是點選的物體或資源的屬性，對於物體而言，可能會有位置、旋轉、碰撞器或是腳本等屬性，都在這邊可以做修改它們的設定

### Project View
預設位於介面的下方，這邊包含遊戲會用到的所有資源。在 Project View 裡面可以看到有兩個資料夾，Assets 和 Packages，Packages 存放的是許多 Unity 內建既有的東西，我們可以不用理它。我們所新加入的任何資源都會在 Assets 裡面，像是 Material、Script、Shader、Prefab、音效、圖片以及導入的 Package 等等

現在我們就我們開始在場景中新增物件吧～

## 基本操作

### 新增物件

![Add Game Object](./images/unity-tutorial-2/AddGameObject.png)
在左邊的 Hierarchy View 中點擊滑鼠右鍵，選擇 **3D OBject** 可以看到許多選擇，像是正方體、球體、膠囊體(?、圓柱體等等，這邊先選擇正方體

![Add Game Object](./images/unity-tutorial-2/AddGameObject2.png)

新增好後 Hierarchy View 就多了一個名為 Cube 的 Cube，Scene View 和 Game View 也可以看到這個正方體。點選這個正方體，右邊的 Inspector View 就會顯示這個正方體的詳細資訊。

###　快捷鍵
想要快速編輯遊戲場景，勢必要熟悉 Unity 的一些操作，這邊簡單介紹一下我經常使用的快捷鍵

#### 滑鼠滾輪
放大和縮小當前的畫面
![Mouse Scroll](./images/unity-tutorial-2/MouseScroll.gif)

#### ALT
能夠改變當前的視角角度
![ALT](./images/unity-tutorial-2/ALT.gif)

#### Q
快速切換成 View Tool，可以在 Scene View 中自由移動
![Q](./images/unity-tutorial-2/Q.gif)

#### W
快速切換成 Move Tool，可以移動物體的位置
![W](./images/unity-tutorial-2/W.gif)

#### E
快速切換成 Rotate Tool，可以旋轉物體
![E](./images/unity-tutorial-2/E.gif)

#### R
快速切換成 Scale Tool，可以改變物體的大小
![R](./images/unity-tutorial-2/R.gif)

事實上，上面提到的 QWER 也可以直接用滑鼠點 Scene View 左上角的 Tools 來使用，不過直接用快捷鍵會在日後方便許多，建議可以多熟悉使用快捷鍵來編輯。

除此之外，在移動物體、旋轉物體和縮放物體的時候，可以看到右上角 Inspector 中的 Transform 也會隨之改變，前面有提到 Inspector View 可以檢視當前物體的屬性，且物體的位置、旋轉角度和大小都存在於 **Transform 屬性** 裡面，所以其實也可以直接在這邊輸入數字去做修改，和在 Scene View 去編輯是沒有差別的。

![Inspector View](./images/unity-tutorial-2/InspectorView.gif)

#### F
快速在場景中找到某物件。當遊戲場景很大的時候，這個功能會變得很好用，不用慢慢自己找
![R](./images/unity-tutorial-2/F.gif)

#### Ctrl + Z / Ctrl + Y
Undo 和 Redo，這應該不用多說什麼

#### Ctrl + S
儲存檔案。請**務必**養成做完任何改動都點一下 Ctrl + S 的習慣，Unity 並不會自動幫你儲存。有時候做完大幅度的改動後 Unity 突然崩潰，所有努力直接化為泡影，當下真的欲哭無淚 QAQ

![Unity Crash](./images/unity-tutorial-2/UnityCrash.png)
每次看到這畫面心都會抖一下... 

---

到目前為止，你應該已經知道怎麼在遊戲中新增物件，還有怎麼改變物體在場景的位置、大小等，記得多多使用快捷鍵來提高效率！還有記住有事沒事就點一下 Ctrl + S，日後一定會感謝自己。

下一篇會介紹怎麼使用 Unity Asset Store、Sketchfab 導入 3D 場景和物件，或是導入自己的素材，下次再見～[Unity 新手教學 - 導入物件](https://933yee.github.io/notes/2024/04/10/unity-tutorial-3/)
