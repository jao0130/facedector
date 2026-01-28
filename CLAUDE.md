# Face Detector - 輕量級臉部偵測深度學習專案

## 專案概述

基於 MobileNetV2 + FPN 的輕量級臉部偵測模型，支援邊界框偵測和 5 點臉部關鍵點定位。

## 目錄結構

```
facedector/
├── configs/
│   └── config.yaml          # 訓練配置
├── data/
│   ├── augmentation.py      # 數據增強
│   ├── dataset.py           # PURE 資料集載入器
│   └── generate_labels.py   # MediaPipe 標籤生成
├── models/
│   ├── backbone.py          # MobileNetV2 骨幹網絡
│   ├── detector.py          # 檢測頭和關鍵點頭
│   ├── face_detector.py     # 完整模型定義
│   └── face_landmarker.task # MediaPipe 預訓練模型
├── utils/
│   ├── losses.py            # 損失函數 (GIoU, Wing, Focal)
│   ├── metrics.py           # 評估指標 (IoU, NME, mAP)
│   └── visualization.py     # 可視化工具
├── export/
│   ├── export_tflite.py     # TFLite 導出
│   └── export_tfjs.py       # TensorFlow.js 導出
├── train.py                 # 訓練腳本
├── inference.py             # 推理腳本
└── requirements.txt         # 依賴包
```

## 模型架構

```
Input [B, 256, 256, 3]
    ↓
MobileNetV2 Backbone (α=0.5)
    ↓
Feature Pyramid Network (FPN)
    ↓
Multi-Head Output:
├── Bbox Head → [B, 4]
├── Landmark Head → [B, 5, 2]
└── Confidence Head → [B, 1]
```

## 常用指令

### 環境設置
```bash
pip install -r requirements.txt
```

### 標籤生成
```bash
python data/generate_labels.py --pure_dir D:/PURE --output_dir D:/PURE_labels --model_path models/face_landmarker.task
```

### 訓練
```bash
# 基本訓練
python train.py --config configs/config.yaml --epochs 100

# 恢復訓練
python train.py --config configs/config.yaml --resume checkpoints/best_model.weights.h5
```

### 推理測試
```bash
# 單張影像
python inference.py --model checkpoints/best_model.weights.h5 --config configs/config.yaml --input test.jpg --output result.jpg

# 視頻
python inference.py --model checkpoints/best_model.weights.h5 --config configs/config.yaml --input video.mp4 --output result.mp4

# 即時攝影機
python inference.py --model checkpoints/best_model.weights.h5 --config configs/config.yaml --input 0
```

### 模型導出
```bash
# TFLite (移動端)
python export/export_tflite.py --model checkpoints/best_model.weights.h5 --config configs/config.yaml --output export/face_detector.tflite --verify

# TensorFlow.js (Web)
python export/export_tfjs.py --model checkpoints/best_model.weights.h5 --config configs/config.yaml --output export/tfjs_model
```

## 訓練配置 (config.yaml)

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `model.input_size` | 256 | 輸入影像尺寸 |
| `model.backbone_alpha` | 0.5 | MobileNetV2 寬度因子 |
| `training.batch_size` | 32 | 批次大小 |
| `training.epochs` | 100 | 訓練輪數 |
| `training.initial_learning_rate` | 0.001 | 初始學習率 |
| `training.warmup_epochs` | 5 | 學習率預熱期 |
| `loss.landmark_weight` | 0.5 | 關鍵點損失權重 |

## 輸出檔案

| 路徑 | 說明 |
|------|------|
| `checkpoints/best_model.weights.h5` | 最佳驗證模型 |
| `checkpoints/final_model.weights.h5` | 最終模型 |
| `logs/training_history.json` | 訓練歷史數據 |
| `logs/training_curves.png` | 訓練曲線圖 |
| `export/face_detector.tflite` | TFLite 模型 |
| `export/tfjs_model/` | TFJS 模型目錄 |

## 資料集

- **來源**: PURE 資料集
- **路徑**: `D:/PURE`
- **標籤**: `D:/PURE_labels` (JSON 格式)
- **分割**: 80% 訓練 / 20% 驗證

## 5 點關鍵點定義

| 索引 | 名稱 | MediaPipe 索引 |
|------|------|----------------|
| 0 | 左眼 | 468 (左虹膜中心) |
| 1 | 右眼 | 473 (右虹膜中心) |
| 2 | 鼻尖 | 4 |
| 3 | 左嘴角 | 61 |
| 4 | 右嘴角 | 291 |

## 依賴版本

- TensorFlow >= 2.13
- OpenCV >= 4.8.0
- MediaPipe >= 0.10.0
- NumPy >= 1.24.0
