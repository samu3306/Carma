# Carma Cloud Run smoke deployment

最初的 smoke 部署只驗證公開網址、前端、FastAPI 與上傳流程。目前容器已加入
DINOv2 ViT-S/14 MatMul-INT8 ONNX＋Linear SVM 的 CPU 六角度推論；Qwen 仍未載入。
若尚未執行第 5 節，SQLite 與上傳照片仍會隨 Cloud Run instance 被替換而消失。

## 1. Cloud Shell 設定

```bash
export CARMA_PROJECT_ID=project-77771f20-96a6-4523-a0f
export CARMA_REGION=asia-southeast1

gcloud config set project "$CARMA_PROJECT_ID"
gcloud config set run/region "$CARMA_REGION"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## 2. 將原始碼放進 Cloud Shell

把 Carma 專案上傳或 clone 到 Cloud Shell，切換到包含根目錄 `Dockerfile` 的資料夾。
部署來源會依 `.gcloudignore` 排除虛擬環境、資料集、照片與非必要模型。

## 3. 建置並部署 CPU smoke service

```bash
gcloud run deploy carma-web \
  --source . \
  --region "$CARMA_REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 2 \
  --memory 4Gi \
  --concurrency 4 \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 300
```

完成後，指令會輸出 `*.run.app` 網址。先檢查：

```bash
export CARMA_URL="Cloud Run 輸出的網址"
curl "$CARMA_URL/health"
```

瀏覽器開啟 `$CARMA_URL`，輸入測試車牌並走一次取車流程。

新版健康檢查應顯示：

```json
{
  "angle_classifier_enabled": true,
  "angle_classifier": "human_trained_dinov2_vits14_onnx_svm"
}
```

## 4. 停止計算資源

smoke service 設為 `min-instances=0`，閒置時會自動縮到零。若不再使用，可刪除服務：

```bash
gcloud run services delete carma-web --region "$CARMA_REGION"
```

## 5. 啟用永久儲存

正式展示版使用 Cloud SQL PostgreSQL 保存訂單、檢查結果與後台預警，並把原始照片和分析圖
保存到 Cloud Storage。資料庫連線字串放在 Secret Manager，不寫進映像檔或原始碼。

在 Cloud Shell 的 Carma 專案根目錄執行：

```bash
chmod +x deploy/cloudrun/setup-persistence.sh
export CARMA_PROJECT_ID=project-77771f20-96a6-4523-a0f
export CARMA_REGION=asia-southeast1
./deploy/cloudrun/setup-persistence.sh
```

腳本會建立 `carma-postgres`、`${CARMA_PROJECT_ID}-carma-media`、資料庫使用者與 Secret，
並重新部署 `carma-web`。重跑時會沿用既有 Secret，不會刪除資料或任意輪替資料庫密碼。

部署後確認：

```bash
curl -s "https://carma-web-418753561086.asia-southeast1.run.app/health"
```

回應必須包含：

```json
{
  "database_backend": "postgresql",
  "storage_backend": "cloud_storage_mount",
  "persistent_storage": true
}
```

Cloud SQL 是計費資源。活動結束後若確定不再使用，可先匯出備份再停止或刪除；刪除資料庫屬不可逆操作。

## 正式版尚需完成

- 私有 `carma-vlm` GPU service 與 IAM service-to-service 驗證。
- 輕量車牌 OCR 與低信心才呼叫 VLM 的分層流程。
- 使用者與營運後台的登入及 API 授權。
