# Qwen3-VL iRent 角度與照片品質 zero-shot 基準

本工具只讀取 `downloaded_images/`，抽樣時只建立 CSV manifest，不會複製、移動或修改官方圖片。模型固定預設為 `Qwen/Qwen3-VL-8B-Instruct`，使用 BF16、`device_map="auto"`、SDPA 與 deterministic decoding。

## 執行順序

請從專案根目錄、啟用既有 `.venv` 後執行：

```bash
source .venv/bin/activate
python scripts/check_vlm_environment.py
python scripts/build_angle_eval_set.py --seed 42
python scripts/test_single_image.py \
  --image "$(sed -n '2p' data/eval/angle_eval_manifest.csv | cut -d, -f1)" \
  --model Qwen/Qwen3-VL-8B-Instruct
python scripts/evaluate_angle_vlm.py \
  --manifest data/eval/angle_eval_manifest.csv \
  --output outputs/qwen3_vl_angle_baseline \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --limit 1
python scripts/evaluate_angle_vlm.py \
  --manifest data/eval/angle_eval_manifest.csv \
  --output outputs/qwen3_vl_angle_baseline \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --limit 5 --resume
```

若檔案路徑含逗號，請直接從 manifest 複製 `image_path`，不要使用上面的簡便 `cut` 指令。批次推論每張完成後會立刻 append 到 `predictions.jsonl`；`--resume` 依 `image_path` 跳過已完成項目。

完整 120 張評估須在前 5 張結果經人工確認後才執行：

```bash
python scripts/evaluate_angle_vlm.py \
  --manifest data/eval/angle_eval_manifest.csv \
  --output outputs/qwen3_vl_angle_baseline \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --resume
```

`failure_cases.csv` 是「模型輸出與 Excel/ImageType 弱標籤不一致或推論失敗、需人工檢查」的案例清單，不應直接視為模型判斷錯誤。
