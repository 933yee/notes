---
title: Unity 新手教學 - WHY and HOW
date: 2024-04-10 13:40:34
tags: Unity
---
> 3D 虛擬影像即時互動 期中報告

在這次教學中，我們將學習 Unity 相較於 Unreal Engine 有哪些優勢，以及如何下載和建立一個新的 Unity 專案
## 為什麼要用 Unity？
在遊戲開發的世界裡，Unity 是一個非常受歡迎的遊戲引擎，擁有許多獨特的優勢使它成為許多開發者的首選。儘管 Epic Games 開發的 Unreal Engine 因 Fortnite 的成功而聲名大噪，使他成為當今最熱門的遊戲引擎之一，許多 3A 大作也都是出自於 Unreal

說了那麼多，為什麼我們還要考慮使用 Unity 呢？讓我們來看看 Unity 有哪些 Unreal 沒有的優勢：

1. **簡單好上手**
對於新手來說，Unity 提供了一個相對輕鬆的入門門檻。它的介面設計和操作方式相對更直觀，學習曲線也比 Unreal 穩定許多。此外，個人覺得 C# 會比 C++ 更好上手，不用處理一堆指標的問題，會更適合新手去學習，不過 C++ 能夠壓榨出更多硬體效能就是了～

2. **適合獨立遊戲製作**
Unreal Engine 通常被用於開發大型、高品質的遊戲，開發時間、成本會比較高，Unity 則更適用於快速開發、跨平台部署和較小規模的專案。如果你是一個獨立開發者或是想要快速將遊戲推向市場的團隊，Unity 可能會更適合你。（這也是為什麼 Unity 收費事件會被炎上，新的收費方案將會扼殺許多獨立遊戲工作室，我最期待的 Silksong 可能要再多等五年，好險後來 CEO 下台了（X

3. **龐大的開發者社群**
當你在用 Unity 的時候遇到問題，唯一需要做的就是去 Google 一下。不管是 Bug 訊息還是你想做的事情，幾乎都能找到別人分享的程式碼或影片教學。而且，Unity 的[官方文件](https://docs.unity3d.com/Manual/index.html)也極其有好，裡面涵蓋了各種用法的詳細使用說明。相比之下，我個人覺得 Unreal Engine 的教學文件就...好像沒什麼用，常常看了還是搞不清楚在說啥，這也是被人詬病的地方。

其他還有像是豐富的資源庫、跨平台性等優勢，這裡就先不贅述了。

## 下載 Unity
在使用 Unity 之前，我們會需要去下載 [Unity Hub](https://unity.com/download)。 

### Unity Hub
Unity Hub 是 Unity 的一個管理工具，能夠讓你輕鬆管理和更新 Unity 的各種版本，也能夠在需要時隨時切換。

![Unity Download](./images/unity-tutorial/UnityDownload.png)

### Unity Editor
Unity Editor 版本就相當於你的 Unity 版本，不同版本之間使用上會有差異，不同 Package 能夠支援的版本也不一樣，所以在建立專案前，我們會需要決定這個專案要使用的 Unity 版本是什麼。

下載好 Unity Hub 後，我們可以看到這個畫面，下面的那兩個專案是我之前建立的

![Unity Hub](./images/unity-tutorial/UnityHub.png)

這邊點選左邊的**安裝**，下面的那兩個 Unity Editor 版本也是我之前下載的

![Unity Hub](./images/unity-tutorial/UnityHub2.png)

在這邊可以新增你想要的 Unity Editor 版本，也可以管理你已經擁有的版本

### 管理授權
在新增專案之前，可以看到上面會跟你說 *`沒有可用授權 要建立並開啟專案，您需要有一個有效的授權`* 

![Unity Hub](./images/unity-tutorial/UnityHub3.png)

這時候可以點選右上角的**管理授權**

![Unity Hub](./images/unity-tutorial/UnityHub4.png)

然後點選**新增授權**

![Unity Hub](./images/unity-tutorial/UnityHub5.png)

這邊選擇**取得免費的個人版授權**就好

![Unity Hub](./images/unity-tutorial/UnityHub6.png)

重開後會發現現在已經有授權了～不過這授權是有時效的，過期後再做一遍剛剛做的事就好，現在就可以開始建立專案了。

## 建立專案 
授權好後，現在我們來建立一個自己的專案

![Unity Hub](./images/unity-tutorial/UnityHub7.png)

這邊點選右上角的 **新專案**

![Unity Hub](./images/unity-tutorial/UnityHub8.png)

可以看到一個專案也有各種不同的範本，像是 2D、3D、2D Mobile、3D Mobile...，這邊我們先選擇 3D (Built-in)，最上方可以選擇 Unity 的版本，右邊可以命名專案的名稱和專案存放的位置

![Unity Hub](./images/unity-tutorial/UnityHub9.png)

建立好後，能夠看到這個預設畫面，現在就可以開始製作自己的遊戲了～ 

![Unity Editor](./images/unity-tutorial/UnityEditor.png)

---

到目前為止，你應該大致理解 Unity 有哪些好處，還有 Unity 下載的流程是什麼。

下一篇教學會介紹如何在 3D 場景中執行一些操作，下次再見～[Unity 新手教學 - 介面介紹和基本操作](https://933yee.github.io/notes/2024/04/10/unity-tutorial-2/)
