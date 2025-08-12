---
title: Paper
date: 2025-05-11 19:04:37
tags: [paper, research]
category: paper
---

# Quantized Side Tuning: Fast and Memory-Efficient Tuning of Quantized Large Language Models

## 摘要

本文提出一種名為 Quantized Side Tuning (QST) 的訓練框架，旨在快速且記憶體效率高地微調大型語言模型（LLMs）。該方法透過兩階段處理：

1. 將模型權重量化為 4-bit，減少儲存所需的記憶體。
2. 設計一個與主模型分離的「側邊網路（side network）」，該網路僅根據主模型的隱藏層輸出進行任務預測，從而避免在主模型上進行反向傳播，進一步減少中介激活值的記憶體消耗。

此外，QST 運用低秩適配器（如 LoRA、Adapter）與無梯度下取樣模組來降低可訓練參數數量。實驗顯示，QST 能夠將總記憶體占用降低至最多 1/7，訓練速度提升 3 倍，並在準確率上保持與現有最先進方法相當的水準。

## 前言

隨著大型語言模型（LLMs）不斷增大（達數百億參數），其在微調過程中所需的記憶體與計算資源急遽上升。現有的 PEFT（參數高效率微調）雖能部分減少訓練參數，但仍需保留大量中介激活值，因此無法從根本上解決記憶體瓶頸。本文提出 QST，目的在於同時解決模型權重、優化器狀態與中介激活值三大記憶體來源問題。

## 研究目的本研究主要目的為：

- 設計一個針對量化 LLMs 的記憶體節省且快速微調方法。
- 減少三個主要記憶體負擔來源：模型權重、優化器狀態與中介激活值。
- 維持微調後模型的高效性能，並提升在多種下游任務上的表現。

## 文獻探討

- 2.1 Parameter-Efficient Finetuning：
  回顧 PEFT 方法如 LoRA、Adapters、Prompt Tuning 等，其特點在於僅微調少量參數；但這些方法仍需保留大部分中介激活值，導致記憶體需求仍偏高。

- 2.2 Memory-Efficient Training and Finetuning：
  探討如 Gradient Checkpointing、可逆網路、網路剪枝與蒸餾等降低記憶體需求技術，指出這些技術或需額外運算、或不適用於大模型；因此提出 QST 為一種可針對全模型大小皆適用的解法。

## 研究方法

- 研究設計：
  提出 QST 架構，包含 4-bit 量化與 side tuning 雙階段設計。

- 研究對象：
  使用多種主流 LLM 架構（如 OPT、LLaMA-2）進行實驗，涵蓋模型參數從 1.3B 至 70B。

- 研究工具：
  實作於 PyTorch 與 HuggingFace 平台，使用 NF4 為 4-bit 資料型態、bfloat16 為計算型態。

- 資料處理與分析：
  評估模型在 GLUE、MMLU 等標準資料集上的準確率與記憶體消耗，並以 FLOPS/token 量測訓練速度。

## 研究結果

- 記憶體消耗： QST 可比 QLoRA 降低記憶體使用量高達 2.3 倍。
- 訓練速度： 相較 QLoRA 提升 3 倍以上。
- 準確率： 在 GLUE 與 MMLU 數據集上與 QLoRA 等方法表現相當，誤差小於 1%。
- 模型擴展性： 可有效應用於 70B 參數的 LLaMA-2 模型。
- 下取樣模組比較： Adapter 為最佳方案，在效能與記憶體效率間取得平衡。
- 應用測試： 在 MT-Bench 聊天測試中，QST 甚至超越原始 LLaMA-2-70B 模型。

## 結論與建議

- 結論：
  - QST 為一項具突破性的微調方法，能在不犧牲效能的情況下，顯著降低訓練資源消耗，特別適合應用於記憶體受限的現實場景。
- 建議：
  - 未來可擴展 QST 至多模態模型。
  - 探索更多無參數下取樣方法，以進一步降低資源需求。
  - 深入研究 QST 於推論時的優化策略。

### 需要查的

- gradient checkpointing
- PEFT
- quantization
- QLoRA
- finetuning
- side network
- LLM head

# DISP-LLM: Dimension-Independent Structural Pruning for Large Language Models

## 摘要

本研究提出一種新穎的維度無關結構剪枝方法，DISP-LLM，針對大型語言模型（LLMs）進行壓縮。相較於既有的結構剪枝方法（如 LLM-Pruner 和 SliceGPT）在彈性或額外參數負擔上的侷限，本方法打破了剪枝過程中嵌入維度的結構依賴性，允許每個轉換層使用不同的特徵子集。本法具備以下優勢：每層可使用不同輸入/輸出寬度，提升剪枝彈性且不引入額外參數。實驗驗證指出，本方法於 OPT、LLaMA、LLaMA-2、Phi-1.5 與 Phi-2 等模型上優於現有技術，首次證明結構剪枝可達與半結構剪枝相當的準確性。

## 前言

大型語言模型在自然語言處理任務上表現卓越，但其龐大的參數量與高昂的運算成本限制了其在資源受限設備上的應用。過往解法如稀疏化、量化與剪枝方法已獲關注，尤其是結構剪枝能在不改變模型權重的情況下減少成本。然現有方法如 LLM-Pruner 受限於結構依賴性，SliceGPT 則需額外投影參數，影響效能與結構簡潔性。故本研究旨在提供一種不引入額外參數且具高彈性的結構剪枝策略。

## 研究目的

- 目標：設計一種維度無關（dimension-independent）結構剪枝方法，避免層與層間的特徵依賴，達成靈活剪枝與高效推論。
- 問題：如何在不引入額外參數的前提下，讓每層自由選擇輸入/輸出特徵子集，並同時維持模型效能？

## 文獻探討

- Magnitude-based Pruning： 透過 L1/L2 範數移除小權重，但對 LLMs 效能傷害大（Han et al., 2015）；
- Optimal Brain Damage / Surgeon： 依據 Hessian 資訊進行剪枝，計算成本高（LeCun et al., 1990；Hassibi & Stork, 1993）；
- LLM-Pruner： 遵循結構依賴進行剪枝，彈性不足（Ma et al., 2023）；
- SliceGPT： 使用主成分分析進行空間投影，具彈性但需新增轉換參數（Ashkboos et al., 2024）；
- SparseGPT/Wanda： 使用 Hessian 近似進行半結構剪枝，難以達成實際加速（Frantar et al., 2023；Sun et al., 2024）；

## 研究方法

- 研究設計： 提出 DISP-LLM 方法，透過**索引選擇與索引加總（Index Select & Add）**打破層間特徵依賴；
- 研究對象： 多種 LLMs 模型（OPT、LLaMA、Phi 系列）；
- 研究工具： 使用 Pytorch 與 Hugging Face 進行訓練與剪枝，搭配 GRU 型超網路（Hypernetwork）與 ReinMax 方法決定每層剪枝寬度；
- 資料處理與分析： 透過語言建模損失與正規化條項進行結構優化，並使用 WikiText-2、Alpaca 等資料進行訓練與評估。

## 研究結果

- DISP-LLM 在各剪枝率（10%~50%）下均優於 LLM-Pruner、SliceGPT 與 LLM Surgeon；
- 相對於 SliceGPT，效能更佳且無新增參數負擔；
- 在多種任務如 WinoGrande、HellaSwag、ARC 等零樣本評估中具備領先表現；
- 可靈活決定每層寬度，各層剪枝比率不一，有助於效率與效能的雙重提升。

## 結論與建議

- 結論： DISP-LLM 透過打破結構依賴、使用超網路與指標操作成功提升剪枝彈性，為大型語言模型剪枝提供新路徑；
- 建議：
  - 後續可探討不同任務導向下的子網路學習；
  - 評估結合 LoRA 等方法進行微調與壓縮的混合應用；
  - 延伸至多模態模型壓縮場景。

# Hot or Cold? Adaptive Temperature Sampling for Code Generation with Large Language Models

## 摘要

該研究指出現有大型語言模型（LLMs）使用的解碼策略多為自然語言生成所設計，忽略了自然語言與程式語言間的差異。本文首度系統性探討專為程式碼生成設計的解碼策略，分析發現程式碼中存在「困難標記（challenging tokens）」與「穩定標記（confident tokens）」，且困難標記通常出現在程式碼區塊的開頭。基於此，作者提出「自適應溫度採樣法（Adaptive Temperature Sampling, AdapT）」，針對不同類型的標記動態調整溫度參數，結果顯示此法顯著優於現有狀態最佳解碼方法。

## 前言

- 背景：大型語言模型（LLMs）在自動程式碼生成方面展現強大潛力，例如 AlphaCode、Codex 等模型。
- 動機：現有解碼策略未專為程式碼設計，沿用自然語言生成的方法，如溫度採樣（temperature sampling）在語法嚴格的程式碼生成中成效有限，需開發新策略提升準確性與多樣性。

## 研究目的

- 分析程式碼生成中損失函數（loss distribution）的特性，找出難預測與易預測的標記類型。
- 根據上述分析，設計自適應溫度採樣法（AdapT）。
- 評估該方法於不同大型語言模型與資料集上的實際效能。

## 文獻探討

- 解碼策略分類：搜尋式（如 Greedy、Beam Search）與採樣式（如 Temperature、Top-k、Top-p Sampling）。
- 限制：搜尋式缺乏多樣性，採樣式易受尾部亂數影響，對語法嚴格的程式碼生成成效有限。
- 現有方法如 Codex 雖使用溫度調節技術，但其本質設計仍偏向自然語言處理。

## 研究方法

- 研究設計：
  - 比較自然語言與程式碼的損失分布特性。
  - 定義「挑戰性標記」與「穩定標記」並量化其預測困難度。
- 研究對象：
  - 開源 LLM 模型：CodeGen-2B、InCoder-6B、CodeGeeX-13B。
  - 資料集：HumanEval（164 題）、MBPP（500 題）、APPS（訓練集 5000 題）。
- 研究工具與實驗設置：
  - 使用 PyTorch 與 GPU (NVIDIA V100) 執行模型。
  - 評估指標：Pass@k（k=1,5,10,15）。
- 資料分析：
  - 分析預測困難度（Prediction Difficulty, PD）。
  - 將溫度係數動態設定為兩值：a（困難標記）、b（穩定標記），依條件分配。

## 研究結果

- 挑戰性標記集中出現在程式區塊起始位置。
- AdapT 採樣法顯著提升程式碼正確率：
  - CodeGeeX 在 HumanEval 中 Pass@15 提升 13.6%（36.0% → 40.9%）。
- AdapT 提高生成程式碼品質：
  - 降低 NameError、SyntaxError、TypeError 等錯誤發生率。
  - 比 SP（標準溫度採樣）產出更多通過測試的程式碼片段。

## 結論與建議

- 結論：
  - LLM 在程式碼生成中面臨的挑戰與自然語言不同。
  - 自適應溫度採樣有效區分挑戰性與穩定標記，提升準確率與可讀性。
  - AdapT 展現高魯棒性與跨模型、跨資料集適應性。
- 建議：
  - 可採學習型方法動態決定溫度。
  - 將領域知識納入解碼過程，以應對真實開發需求。
  - 建立多階段解碼策略（如先產出自然語言規劃，再生成程式碼）。

# Memory-Efficient Vision Transformers: An Activation-Aware Mixed-Rank Compression Strategy

## 摘要

本研究針對 Vision Transformers (ViTs) 模型推論時所面臨的高記憶體需求問題，提出一種基於輸入啟動（activation-aware）的混合秩（mixed-rank）壓縮策略，藉由對各層權重張量進行低秩分解，並搭配層別誤差補償方法，以達成大幅減少模型參數量而不明顯損失預測準確度的目標。方法上，將原始權重張量拆解為兩個參數高效的子張量之和，並透過啟動輸入與誤差梯度導向的最小化策略進行細緻優化。實驗顯示，該方法能將 DeiT-B 模型參數減少 60%，準確率僅下降不到 1%，並可壓縮大型 ViT 至與較小模型同級的參數量，同時提升預測精度，證明其為記憶體受限環境中可行之解決方案。

## 前言

Vision Transformers 自提出後在多項視覺任務中展現卓越性能，包括圖像分類、物件偵測與語意分割。然而，其高參數與記憶體使用成本限制了其在實務中之部署潛力。傳統壓縮方法如剪枝、量化、知識蒸餾、token pruning 雖有效，但在 transformer 架構中效果有限，尤其是直接應用低秩分解時會導致性能大幅下降。本研究即在此背景下，探索能兼顧效能與記憶體需求的新式壓縮方法。

## 研究目的

本研究旨在提出一種新型「啟動感知混合秩壓縮方法（Activation-Aware Mixed-Rank Compression）」，目標包括：

1. 利用啟動輸入資訊導引權重張量之低秩分解。
2. 對 ViT 各層進行差異化秩配置（mixed-rank）。
3. 發展層級誤差補償機制（layer-wise error compensation）。
4. 在大幅壓縮參數量的同時維持高準確率，避免過去低秩方法常見之性能損失。

## 文獻探討

- Eckart & Young (1936)：理論基礎，證明 SVD 在固定秩下為最小誤差解。
- Jaderberg et al. (2014)、Hsu et al. (2022)：探討低秩在 CNN/Transformer 壓縮中之應用與挑戰。
- Yu & Wu (2023)：證明輸出特徵比權重更適合低秩近似，並導入激活感知策略。
- Li et al. (2023)：採用低秩與稀疏矩陣聯合表示，但仍會因能量流失導致準確率下降。
- Kumar (2022)：提出 attention block 採低秩，feed-forward 採剪枝的混合法。

## 研究方法

- 研究設計：
  - 分三階段：Activation-Aware SVD、Mixed-Rank Optimization、Layer-Wise Error Compensation
- 研究對象：
  - 多個 ViT 架構（包括 DeiT、ViT-B、Swin）
- 研究工具與技術：
  - Singular Value Decomposition (SVD)
  - 啟動輸入加權的張量分解
  - 啟發式逐層秩指派（greedy local search）
  - 層別梯度導向補償（gradient-based optimization）
- 資料處理與分析：
  - 以 proxy dataset 模擬啟動輸入
  - 使用 Frobenius norm 評估重建誤差
  - 進行 ImageNet 準確率與參數壓縮比分析

## 研究結果

- 壓縮率與精度：
  - DeiT-B 壓縮 60% 僅下降 0.7%（81.80% → 81.10%）
  - DeiT-S 壓縮後比原模型準確度更高（79.80% → 80.30%）
- 相較其他方法：
  - 優於 SPViT、S2ViTE、WDPruning 等現有壓縮技術
- 極端壓縮案例：
  - ViT-L 壓縮為 ViT-B 尺寸後仍優於原 ViT-B 模型
- 與量化兼容性強：
  - 搭配 8-bit quantization 仍可維持準確率

## 結論與建議

- 結論：
  - 本研究提出的啟動感知混合秩壓縮方法，能有效降低 ViT 模型參數量，並在保留原始準確率的同時，顯著減少記憶體需求。所提出的三階段流程具有高度可擴展性與實作效能，亦兼容後訓量化策略。
- 建議：
  - 此法未來可延伸應用至語言模型（如 LLMs）。
  - 建議探討動態秩指派與多任務情境下之壓縮策略。
