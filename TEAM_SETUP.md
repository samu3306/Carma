# Carma 隊友測試指南

這個 GitHub 版本只包含完整 MVP 執行所需的程式與輕量模型，不包含原始照片、標註資料、
上傳內容、SQLite、Qwen 權重或本機虛擬環境。

## Repository 應包含

- `backend/`、`frontend/`、`src/`、`scripts/`、`tests/`
- `deploy/`、`docs/`
- 根目錄 `Dockerfile`、`docker-compose.yml`、`requirements.txt`
- `.env.example`、`.gitignore`、`.gcloudignore`、`.dockerignore`
- `frontend/public/guides/` 六張拍攝範例
- `models/yolov4-tiny.cfg`、`models/yolov4-tiny.weights`
- `models/dinov2_angle_classifier/angle_metadata.npz`
- `models/dinov2_angle_classifier/angle_svm.xml`
- `models/dinov2_angle_classifier/dinov2_vits14_matmul_int8.onnx`

直接執行 `git add .` 時，專案的 `.gitignore` 只會放行上述執行期模型；
FP32 DINO、Hugging Face cache 與訓練產物仍會被排除。

## 最快方式：Docker

需求：Docker Desktop 或 Docker Engine＋Compose。

```bash
docker compose up --build
```

開啟：

- 完整取還車 MVP：http://localhost:8080/
- 營運後台：http://localhost:8080/admin
- 健康檢查：http://localhost:8080/health

第一次建置會下載 Python 與 npm 套件。Docker volume `carma_data` 會保留隊友本機的
SQLite 與測試上傳；它不會連到正式 Cloud SQL 或正式照片 bucket。

## 不用 Docker：專案虛擬環境

優先使用 Python 3.11。

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m backend.app.seed
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

另開 PowerShell：

```powershell
cd frontend
npm ci
npm run dev
```

開啟 http://localhost:5173/。完成後執行 `deactivate` 退出虛擬環境。

### Linux／macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m backend.app.seed
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

另開終端：

```bash
cd frontend
npm ci
npm run dev
```

開啟 http://localhost:5173/。完成後執行 `deactivate`。

## 驗證模型

`/health` 應包含：

```json
{
  "angle_classifier_enabled": true,
  "angle_classifier": "human_trained_dinov2_vits14_onnx_svm"
}
```

正確角度應允許進到下一張；把左前照片放進右前欄位時，應顯示需要重拍。

## 不應提交

- `.env`、服務帳號 JSON、API key 或密碼
- `.venv/`、`frontend/node_modules/`
- `uploads/`、`downloaded_images/`、SQLite
- 原始訓練照片、Excel、標註資料與 `outputs/`
- Qwen、Hugging Face cache、完整 FP32 DINOv2 或其他實驗模型
