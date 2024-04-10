---
title: Unity 新手教學 - 導入物件
date: 2024-04-10 16:43:15
tags: Unity
---

在這次教學中，我們將學習如何導入自己的圖片、音樂，或是將來自 Sketchfab 和 Unity Asset Store 的酷炫 3D 物件導入到你的 Unity 專案中

## 導入自己的物件
導入自己的物件非常簡單，可以在 Project View 空白處點擊右鍵，Import New Asset，然後選擇要導入的物件即可

![Import New Asset](./images/unity-tutorial-3/ImportNewAsset.png)

還有另一個方法，可以直接把那個東西拉進來就好（我都這樣，因為我很懶 XD

![Import Demo](./images/unity-tutorial-3/ImportDemo.gif)

### 導入圖片
值得一提的是，導入的圖片會自動變成 Texture，所以可以直接用在 3D 模型、UI 元素、粒子效果上面
![Texture](./images/unity-tutorial-3/Texture.png)

## Unity Asset Store

### 什麼是 Unity Asset Store？
[Unity Asset Store](https://assetstore.unity.com/zh-CN) 是 Unity 官方提供的一個平台，類似於 APP store，但是專注於 Unity 開發的資源。這裡有各種各樣的資源，包括 2D pixel art、3D 模型、材質、音效、程式碼，甚至是完整的專案模板、各種輔助型的工具。無論你是需要一個小工具還是一個完整的遊戲框架，Unity Asset Store 都能滿足你的需求。

### 如何使用？
我這邊隨便選擇一個 [Mars Landscape 3D](https://assetstore.unity.com/packages/3d/environments/landscapes/mars-landscape-3d-175814) 的 Package，點擊右邊的 **Add to My Assets**，它應該會叫你先登入帳號

![Add New Asset](./images/unity-tutorial-3/AddNewAsset.png)

登入並 **Add to My Assets** 後，就成功加入了到你帳號的 Assets 裡面了

![Add New Asset](./images/unity-tutorial-3/AddNewAsset2.png)

接著點擊 **Open In Unity**，它會自動幫你打開 Unity 裡面的 Package Manager

![Package Manager](./images/unity-tutorial-3/PackageManager.png)

Package Manager 會有你所有的 Assets，之後你想要檢視 Package Manager 的話，也可以從上方的 Windows 裡面打開

![Package Manager](./images/unity-tutorial-3/PackageManager2.png)

點擊右上方的 **Download**，下載完後再點 **Import**，然後繼續點 **Import**

![Package Manager](./images/unity-tutorial-3/PackageManager3.png)

現在這個 Unity Asset Store 的 Package 就成功導入到你的專案中了！

![Project View](./images/unity-tutorial-3/ProjectView.png)

![Demo Scene](./images/unity-tutorial-3/DemoScene.png)

#### 注意
每次 Import Package 後，可以先去找這個 Package 所提供的 Demo Scene，執行看看有沒有問題，確保這個 Package 是和你現在的 Unity 專案相容的，避免未來出現問題卻找不到問題點在哪

## Sketchfab 

### 什麼是 Sketchfab？
[Sketchfab](https://sketchfab.com/) 是一個線上的平台，提供大量的 3D 模型資源，從建築到角色、動物等各種類型的物件都可以在這裡取得。這些模型通常由社群或專業的模型師製作，你可以在 Sketchfab 上瀏覽、分享，甚至購買這些模型來使用在你的專案中。

### 如何使用？
這邊隨便選擇一個 Pop cat 的 3D 模型，右上角兩個符號代表這模型有動畫，且是可以下載的

![Sketchfab Model](./images/unity-tutorial-3/SketchfabModel.png)

![Sketchfab Model](./images/unity-tutorial-3/SketchfabModel2.png)

![Sketchfab Model](./images/unity-tutorial-3/SketchfabModel3.png)

可以看到這邊沒有 Unity 支援的 .fbx、.dae (Collada)、.dxf 和 .obj. 等格式，不過有提供 .blend，所以我們可以自己開 blender 轉換成 .fbx

![Blender](./images/unity-tutorial-3/Blender.png)

導出成 .fbx 的時候，記得把右上角的 Path Mode 改成 Copy，並點擊右邊的 Embed Textures，之後導入進 Unity 中才能生成對應的 Texture

![Blender](./images/unity-tutorial-3/Blender2.png)

按照先前說的方式導入模型進 Unity 後，你會發現它沒有 Texture

![Cat Model](./images/unity-tutorial-3/CatModel.png)

這時候只要點選剛剛導入進 Assets 的模型，在右邊 Inspector 中的 Materials 裡面點擊 Extract Textures...，就可以順利加上 Texture 了

![Cat Model](./images/unity-tutorial-3/CatInspector.png)

這樣就模型就有 Texture 了！

![Cat Model Texture](./images/unity-tutorial-3/CatModelTexture.png)

## Prefab 是什麼？
你可能有注意到，當你從 Unity Asset Store 或 Sketchfab 導入模型後，在 Hierarchy View 中看到一些藍色的物件。這些藍色物件被稱為 Prefab（預製配件）。Prefab 是 Unity 中的一種特殊、可以重複使用的物件。

![Prefab](./images/unity-tutorial-3/Prefab.png)

舉個例子，假如你在場景中有大量一樣的石頭，這時候你就可以把石頭做成一個 Prefab，這就有點像是這顆石頭的 **藍圖**，你可以根據這個藍圖去建立一大堆石頭。當你想要一次修改時，就可以改這個藍圖就好，不用每一顆慢慢去修改。

![Cat Army](./images/unity-tutorial-3/CatArmy.png)

![Modify Prefab](./images/unity-tutorial-3/ModifyPrefab.gif)

這邊我想要更改 Cat 模型的 Scale，只需要更改藍圖就好，這樣場景中所有根據這個 Prefab 建立的物件就會一起更新。

### 如何建立 Prefab？
非常簡單，只要把你 Hierarchy View 中的物件拉進你的 Project View 裡面就好。像是我這邊建立一個 Cube 到場景中，把它拉進下方就成功囉～

![Create Prefab](./images/unity-tutorial-3/CreatePrefab.gif)

到目前為止，你應該已經知道怎麼導入其它人作好的素材，還有為什麼會需要 Prefab 這個東西。

---

下次再見～