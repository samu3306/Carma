# Carma — iRent 智能車況管家

Carma 是取還車影像檢查 MVP。使用者依序拍攝指定角度，系統即時檢查照片品質與拍攝位置，
並在還車時比較取車基準照；營運端可檢視預警、人工複核及派工。

目前角度分類以 CPU 執行的 DINOv2 ViT-S/14 INT8 ONNX 特徵搭配 Linear SVM 為主。
AI 結果只用於拍攝引導與疑似風險提示，不應直接作為索賠或責任判定。

## 快速啟動

### Docker（建議）

需求：Docker Desktop，或 Docker Engine＋Compose。

```bash
docker compose up --build
```

開啟：

- 完整取還車 MVP：http://localhost:8080/
- 營運後台：http://localhost:8080/admin
- 單張模型測試：http://localhost:8080/test
- 健康檢查：http://localhost:8080/health

本機資料會保存在 Docker volume `carma_data`，不會連接正式 Cloud SQL 或正式照片 bucket。

## 本機開發

Python 套件一律安裝在專案根目錄的 `.venv`。優先使用 Python 3.11，不使用 Conda，
也不要安裝到系統 Python。若 `.venv` 已存在，請先檢查版本與可用性，不要任意刪除重建。

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip --version
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m backend.app.seed
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

`python` 與 `pip` 的路徑都應位於 `.venv`。若 PowerShell 阻擋啟用腳本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Linux／macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -c 'import sys; print(sys.executable)'
python -m pip --version
python -m pip install -r requirements.txt
cp .env.example .env
python -m backend.app.seed
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

`python` 與 `pip` 的路徑都應位於 `.venv`。

另開終端啟動前端：

```bash
cd frontend
npm ci
npm run dev
```

開啟 http://localhost:5173/。完成後執行 `deactivate` 離開虛擬環境。

## 測試

後端：

```bash
python -m pytest
```

前端：

```bash
cd frontend
npm ci
npm run build
```

健康檢查的角度模型應顯示：

```json
{
  "angle_classifier_enabled": true,
  "angle_classifier": "human_trained_dinov2_vits14_onnx_svm"
}
```

## 目前拍攝流程

1. 輸入測試車牌。
2. 完成取車的四張外觀基準照。
3. 返回租程畫面。
4. 點擊開始還車，拍攝四張外觀與前、後車室照片。
5. 每張照片立即檢查模糊、曝光、車輛入鏡、重複照片與指定角度。
6. 角度或車內外位置不符時要求重拍。
7. 完成後產生車況結果及後台預警。

目前 Qwen 品質與車損服務預設關閉；DINOv2 角度模型與固定 CV 規則可在 CPU 執行。

## 專案結構

```text
backend/      FastAPI API、資料庫與影像服務
frontend/     React／TypeScript 使用者端與營運後台
src/          VLM 共用介面與資料結構
models/       Git 允許的輕量執行模型
scripts/      資料整理、模型訓練與評估工具
tests/        後端與模型自動化測試
deploy/       Cloud Run 部署文件與腳本
docs/         架構及模型實驗紀錄
```

## 不可提交的資料

`.gitignore` 已排除：

- `.venv/`、`frontend/node_modules/`
- `.env`、服務帳號 JSON、API key 與密碼
- `uploads/`、`downloaded_images/`
- SQLite 資料庫與暫存檔
- 原始照片、Excel、標註資料與訓練輸出
- Qwen、Hugging Face cache、完整 FP32 DINOv2 與其他大型權重

Git 只保留執行 MVP 必要的 DINOv2 INT8、SVM 與 YOLO 模型。提交前仍應執行
`git status --short`，確認沒有私人或敏感資料。

## 延伸文件

- [目前系統架構](docs/current-architecture.md)
- [DINOv2 角度模型實驗](docs/dinov2-angle-experiment.md)
- [Cloud Run 部署](deploy/cloudrun/README.md)

