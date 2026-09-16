# DINOv2 ViT-S/14 六角度分類實驗

## 實驗設定

- frozen feature extractor：`facebook/dinov2-small`（ViT-S/14，384 維 pooled feature）
- classifier：OpenCV Linear SVM
- 樣本：396 張人工確認合格照片，112 個訂單
- 類別：左前、右前、左後、右後、車前座、車後座
- 驗證：依車輛與訂單分組的 5 折驗證；同一訂單不跨訓練與測試
- 比較基準：相同資料與相同 seed 的 MobileNetV2 1,280 維特徵＋Linear SVM

## 結果

| 指標 | MobileNetV2 | DINOv2 ViT-S/14 | 差異 |
|---|---:|---:|---:|
| 整體準確率 | 78.28% | 89.39% | +11.11 pp |
| 平衡準確率 | 81.26% | 90.13% | +8.87 pp |
| 左前 recall | 76.25% | 83.75% | +7.50 pp |
| 右前 recall | 64.94% | 85.71% | +20.77 pp |
| 左後 recall | 74.03% | 90.91% | +16.88 pp |
| 右後 recall | 80.00% | 92.94% | +12.94 pp |
| 車前座 recall | 95.12% | 90.24% | -4.88 pp |
| 車後座 recall | 97.22% | 97.22% | 0.00 pp |

DINOv2 將左前／右前互相誤判由 44 張降至 24 張，左後／右後互相誤判由 36 張降至 11 張。
四個車外類別平均 recall 從 73.81% 提升至 88.33%。

## 測試部署狀態與限制

- 五折準確率為 89.87%、88.61%、95.00%、96.20%、77.22%，最後一折顯示部分未見車輛或訂單仍有 domain gap。
- 車前座 recall 比既有模型下降 4.88 個百分點。
- 目前已將 DINOv2 設為測試環境的主要六角度模型，搭配 Qwen 進行照片品質檢查；此切換不代表正式上線驗收完成。
- 可用 `CARMA_ANGLE_CLASSIFIER_BACKEND=mobilenetv2` 切回舊模型進行 A/B 對照。
- 下一步應分析第 5 折錯誤、補充左右困難樣本與前座照片，再進行水平翻轉標籤交換實驗。
- 達標後先以 shadow mode 同時記錄 MobileNetV2、DINOv2 與 Qwen 結果，再決定是否切換。

原始指標位於：

- `models/angle_classifier/metrics.json`
- `models/dinov2_angle_classifier/metrics.json`
