# PointInterp - 輕量級LiDAR點雲超解析度框架

PointInterp 是一個輕量且高效的 LiDAR 點雲超解析度（Super-Resolution）非2D投影的解決方案，透過插值法和基於 PointNet 的神經網路模型，實現點雲的稀疏掃描線增強，例如從16線提升到32線、或32線提升到64線，顯著提高點雲的解析度與細節。

## 專案概述
本專案主要提供以下功能：

* **數據轉換管線**：將 KITTI `.bin` 原始點雲轉換為結構化的 HDF5 格式。
* **傳統插值法**：提供穩定的插值基準作為比較依據。
* **PointNet-SR-Mini 神經網路**：專為點雲超解析度設計的輕量級網路。
* **Chamfer Distance 計算工具**：用於評估點雲預測的精確程度。
* **量化分析與可視化腳本**：協助分析點雲處理結果。

## 主要功能
* **多掃描線HDF5轉換器** (`dataset_to_h5.py`)：將64線原始點雲拆分為32線及16線，並存成HDF5檔案。
* **Chamfer Distance實現**：提供 NumPy 和 KD-Tree 兩種高效的計算方式 (`metrics.py`)。
* **插值基準方法**：提供角度與方位角(azimuth)的雙向插值 (`interpolation.py`)。
* **PointNet-SR-Mini 模型** (`pointnet_sr_mini.py`)：

  * 支援16→32及32→64掃描線增強。
  * 可從checkpoint繼續訓練，內建EarlyStopping與學習率調整機制。
* **分析腳本** (`analyze.py`, `main.py`)：

  * 對插值與神經網路預測的效果進行量化評估。
  * 提供單禎點雲的即時可視化比較。

## 主要演算法邏輯

本專案針對 LiDAR 稀疏掃描線提出「差集‑補點」式超解析演算法，流程分為插值基線與深度增補兩層。

**1. 插值基線**  
先以仰角 θ=arcsin(z/r) 將 N 線點雲分箱；對相鄰兩環以 KD‑Tree 配對並插入中點

$$
p_{\text{mid}}=\tfrac12(p_i+p_j)
$$

再於方位角 φ (0–2π) 以 Δφ=0.5° 均衡抽補，確保最終環數精確為目標 R\_t。該步可單獨作為傳統基準。

**2. 差集擷取**  
對原始低環點雲 P\_orig 與高環點雲 P\_tgt，取

$$
\Delta P=\{\,t\in P_{\text{tgt}}\mid\min_{o\in P_{\text{orig}}}\|t-o\|>\tau\}
$$

(τ≈0.05 m) 作為待補點集合。

**3. PointNet‑SR‑Mini**  
對 ΔP( B×N×3 ) 先以 1×1 卷積提特徵 f\_i∈ℝ²⁵⁶，再經 max‑pool 得全域向量 g∈ℝ²⁵⁶，拼接後輸出偏移量 o，最終

$$
P_{\text{pred}}=\Delta P+o
$$

**4. 損失函數**  
採平方 Chamfer 距離

$$
\mathrm{CD}(P,G)=\frac{1}{|G|} \sum_{g \in G} \min_{p \in P} \|g-p\|^2 + \frac{1}{|P|} \sum_{p \in P} \min_{g \in G} \|p-g\|^2
$$

其中 G 為真實高環點雲。優化器用 Adam；ReduceLROnPlateau 動態調降學習率並結合 Early‑Stopping。

**5. 深度學習模型**  

```scss
Input (B, N, 3)
      ↓
MLP1 (Conv1D 3→64→128→256)
      ↓
MaxPool (Global Feature 256)
      ↓
FC Layers (256→512→1024)
      ↓
Expand & Concatenate (256+1024=1280)
      ↓
Decoder (Conv1D 1280→512→256→128→3)
      ↓
Output: Predicted Offsets + Input
```

![images](./figures/architecture.png)

**6. 效果**
在 KITTI 資料集，16→32 環平均 CD 較插值再降約 10%，32→64 環降幅逾 60%；RTX 3080 上可即時推論。演算法輕量、易訓練，亦可替換為 PointNet++、PU‑Net 等更深模型以進一步提升品質。

下圖為KITTI Dataset隨機某禎誤差可視化結果，青色為Ground Truth，紅色為推理結果與GT之差集
![alt text](./figures/pcd_residual.png)

## 專案結構
```plaintext
.
├── dataset_to_h5.py        # 將KITTI原始.bin檔案轉換為HDF5格式 (16/32/64掃描線)
├── io.py                   # HDF5點雲資料載入工具
├── metrics.py              # Chamfer Distance及差集計算工具
├── interpolation.py        # 點雲插值工具
├── pointnet_sr_mini.py     # PointNet-SR-Mini模型定義與訓練
├── analyze.py              # 插值基準法分析腳本
├── main.py                 # 單禎點雲推論與Open3D可視化腳本
├── dataset/                # 放置訓練與測試用的HDF5資料
├── ckpt_srmini/            # 模型checkpoint儲存資料夾
└── figures/                # Chamfer Distance分析曲線圖
```

## 資料集結構
``` plaintext
/train.h5
 ├── 16/    （16線點雲）
 │    ├── 2011_09_26_0001_0000000000  → (N, 4)  （包含 x, y, z, intensity）
 │    ├── 2011_09_26_0001_0000000001  → (M, 4)
 │    └── ...（更多 frame）
 │
 ├── 32/    （32線點雲）
 │    ├── 2011_09_26_0001_0000000000  → (N, 4)
 │    ├── 2011_09_26_0001_0000000001  → (M, 4)
 │    └── ...（更多 frame）
 │
 └── 64/    （64線點雲，Ground Truth）
      ├── 2011_09_26_0001_0000000000  → (N, 4)
      ├── 2011_09_26_0001_0000000001  → (M, 4)
      └── ...（更多 frame）

```

## 軟體依賴
本專案適用於 Python 3.8+，並需安裝以下第三方套件：

* [PyTorch](https://pytorch.org/)
* [Open3D](http://www.open3d.org/)
* [SciPy](https://www.scipy.org/)
* [h5py](https://www.h5py.org/)
* [tqdm](https://github.com/tqdm/tqdm)
* [matplotlib](https://matplotlib.org/)
* [numpy](https://numpy.org/)

安裝指令：

```bash
pip install torch torchvision open3d scipy h5py tqdm matplotlib numpy
```

## 資料準備流程

1. 修改 `dataset_to_h5.py` 中的 `bin_folder_list` 變數，指定你的KITTI原始`.bin`檔案所在目錄。

2. 執行以下指令進行資料轉換：

```bash
python dataset_to_h5.py
```

執行後會產生HDF5檔案（預設名稱為`verify_data.h5`），包含16、32、64線點雲。

3. 將生成的檔案移動至`dataset/`資料夾：

```bash
mv verify_data.h5 dataset/train_sample148.h5
```

## 模型訓練方式

範例：

```bash
python pointnet_sr_mini.py \
  --train_h5 dataset/train_sample148.h5 \
  --val_h5 dataset/verify340.h5 \
  --orig 16 \
  --tgt 32 \
  --epochs 100 \
  --bs 8 \
  --out ckpt_srmini
```

參數說明：

* `--train_h5`: 訓練用的HDF5檔案路徑。
* `--val_h5`: 驗證用HDF5檔案路徑（非必須）。
* `--orig`/`--tgt`: 原始與目標掃描線數。
* `--epochs`: 訓練回合數。
* `--bs`: 批次大小。
* `--out`: checkpoint與記錄儲存路徑。

訓練完成後的資料夾內容：

```plaintext
ckpt_srmini/
├── best_pointnet_sr_16to32.pth
├── loss_history.json
└── loss_curve.png
```

## 推論與分析

### 模型推論評估

使用神經網路模型進行推論並計算Chamfer Distance：

```bash
python your_script.py --filepath dataset/test_sample148.h5 --interpolation dl
```

### 傳統插值分析基準

```bash
python your_script.py --filepath dataset/test_sample148.h5 --interpolation linear
```

### 單禎點雲即時視覺化

隨機選擇一禎進行即時可視化比較效果：

```bash
python main.py
```

## 分析結果
下列資訊使用 `test_sample148.h5` 測試集
* **16線→32線**：Chamfer Distance 受限於原始資料稀疏性，CD平均約為0.34，效果較插值法改善10%左右。
```plaintext
=== 16->32 PointNet-SR-mini 分析報告 ===
總Frame數量: 148
平均 Chamfer Distance: 0.345550
標準差: 0.262420
最大 Chamfer Distance: 2.427180 (Frame 141)
最小 Chamfer Distance: 0.112630 (Frame 36)
```
* **32線→64線**：Chamfer Distance 平均約為0.054，明顯較插值法優化達60%以上。
```plaintext
=== 32->64 PointNet-SR-mini 分析報告 ===
總Frame數量: 148
平均 Chamfer Distance: 0.054512
標準差: 0.029365
最大 Chamfer Distance: 0.191678 (Frame 25)
最小 Chamfer Distance: 0.009076 (Frame 36)
```
* 所有分析曲線圖會儲存於`figures/`資料夾。

## 建議

* 嘗試採用更強大的模型，例如PointNet++、PU-Net、EdgeConv等。
* 加入 Earth Mover's Distance (EMD)、排斥損失 (Repulsion Loss)、多尺度特徵融合與後處理技巧（如Voxel濾波、kNN平滑）。
* 可擴充至更大規模資料集如NuScenes、應用混合精度訓練、One-Cycle LR等技巧。

## 授權方式

本專案採用 MIT License 授權，可自由修改與應用於研究或產品開發。

---

歡迎提供意見、提交問題或貢獻程式碼。
