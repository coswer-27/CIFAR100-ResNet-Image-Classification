# CIFAR-100 影像分類器：深層 ResNet 網路優化與實作

這個專案基於 TensorFlow/Keras 框架，在 Google Colab (A100 GPU) 環境下，針對 CIFAR-100 資料集建構並優化了一個深層殘差網路 (ResNet)，用以解決多類別影像分類任務。

在模型受到嚴格的架構限制與未進行大幅度資料增強 (Data Augmentation) 的前提下，透過**精準的超參數調校與過擬合抑制策略**，實現了高效能與快速收斂。

## 🚀 技術亮點與架構設計

為了在低解析度 (32x32) 且高類別數 (100 類) 的 CIFAR-100 資料集上取得良好泛化能力，本專案針對經典 ResNet 進行了以下優化：

- **深度殘差區塊 (Deep Residual Blocks)**：
  - 設計 4 個階段 (Stages) 的殘差架構，每個 Stage 包含 4 個 Residual Blocks，以深度網路擷取高階語意特徵。
  - 使用 Skip Connection（跳躍連接）傳遞梯度，並在通道數（Filters）倍增時導入 1x1 卷積層進行特徵對齊，有效解決深層網路的梯度消失問題。
- **嚴格的過擬合 (Overfitting) 控制機制**：
  - **Batch Normalization (批次正規化)**：於所有卷積層後、啟用函數前強制加入，穩定內部協變量轉移 (Internal Covariate Shift)。
  - **L2 Regularization (權重正規化)**：將 L2 懲罰係數提高至 `5e-4`，強力約束權重大小，提升模型泛化能力。
  - **Global Average Pooling (全域平均池化)**：在輸出層前取代傳統的 Flatten 層，大幅減少全連接層的參數量，降低過擬合風險並提高運算效率。
- **優化器與學習率調度策略 (Learning Rate Scheduling)**：
  - 採用具備動量的 **SGD 優化器** (`Momentum = 0.9`, `Nesterov = True`)，確保訓練後期的穩定收斂。
  - 導入 **餘弦退火排程 (Cosine Annealing Decay)**：前期給予較學習率 (`0.1`) 加速探索，後期隨 Epoch 曲線平滑遞減，精準鎖定全域最佳解。

## 📂 專案檔案結構

- `mid_112370102_楊子力.ipynb`：完整的模型建構、訓練流程與評估程式碼 (Jupyter Notebook)。
- `mid_112370102_楊子力.pdf`：專案分析報告，包含參數調校邏輯、訓練曲線圖表與實驗心得。
- `mid_112370102_楊子力_cifar100_predictions.csv`：測試集 10,000 張影像的預測結果、真實標籤與信心度 (Confidence Score) 輸出。

## 📊 實驗結果與效能

- **硬體環境**：Google Colab (NVIDIA A100 Tensor Core GPU)
- **訓練輪數 (Epochs)**：250 輪 (總耗時約 45 分鐘)
- **批次大小 (Batch Size)**：128
- **測試集最高準確率 (Test Accuracy)**：`68.7%`

模型在訓練過程中展現了優異的收斂軌跡，驗證集損失函數 (Validation Loss) 趨勢平滑，證明 L2 正規化與餘弦退火策略成功抑制了深層網路易產生的過擬合現象。

## 🛠️ 開發工具與套件

- **深度學習框架**：TensorFlow 2.x / Keras
- **數據處理與視覺化**：NumPy, Pandas, Matplotlib
