# Carma

隊友若要快速啟動完整 MVP，請先看 [TEAM_SETUP.md](TEAM_SETUP.md)。

本專案的 Python 相依套件一律安裝在專案根目錄的 `.venv`，不可安裝到系統 Python，也不使用 Conda。優先使用 Python 3.11。

## Windows（PowerShell）

在專案根目錄執行：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

安裝套件前，先確認 Python 與 pip 均來自 `.venv`：

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

輸出路徑應包含 `Carma\.venv\Scripts\python.exe` 與 `Carma\.venv\Lib\site-packages`。確認後安裝固定版本的相依套件：

```powershell
python -m pip install -r requirements.txt
```

離開虛擬環境：

```powershell
deactivate
```

若 PowerShell 阻擋啟用腳本，可只針對目前程序調整原則後再啟用：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Linux／macOS

在專案根目錄執行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

安裝套件前，先確認 Python 與 pip 均來自 `.venv`：

```bash
python -c 'import sys; print(sys.executable)'
python -m pip --version
```

輸出路徑應包含 `Carma/.venv/bin/python` 與 `Carma/.venv/lib/.../site-packages`。確認後安裝固定版本的相依套件：

```bash
python -m pip install -r requirements.txt
```

離開虛擬環境：

```bash
deactivate
```

## 既有 `.venv` 的處理方式

不要直接刪除或重建既有環境。先檢查版本、執行檔位置及 pip：

```bash
.venv/bin/python --version
.venv/bin/python -c 'import sys; print(sys.executable)'
.venv/bin/python -m pip --version
```

Windows PowerShell 對應指令如下：

```powershell
.venv\Scripts\python.exe --version
.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
.venv\Scripts\python.exe -m pip --version
```

只有在確認 `.venv` 不存在時才建立；若環境存在但不可用，應先找出原因再決定是否重建。

## 相依套件維護

主要套件必須在 `requirements.txt` 使用 `套件名稱==版本` 固定版本。安裝或執行 pip 時，建議一律使用 `python -m pip`，以避免誤用全域 pip。

## 系統架構

- Backend：FastAPI、Pydantic、SQLAlchemy、SQLite。
- 影像處理：OpenCV、NumPy、scikit-image、Pillow。
- Frontend：React、Vite、TypeScript。
- 部署：Docker Compose；API 文件由 FastAPI Swagger 提供。
- 模型層：`QualityAnalyzer`、`DamageDetector`、`CleanlinessClassifier` 介面與 API 分離。

目前車輛是否入鏡與車身框使用預訓練 YOLOv4-tiny，其餘品質檢查仍包含 Demo 規則。六角度分類、車牌文字辨識、車損與整潔度尚非正式模型，不可用於正式索賠或責任判定。

## 本機啟動

確認虛擬環境已啟用且 Python、pip 路徑都位於 `.venv` 後：

```bash
python -m pip install -r requirements.txt
python -m backend.app.seed
python -m uvicorn backend.app.main:app --reload
```

另開終端啟動前端：

```bash
cd frontend
npm install
npm run dev
```

各介面使用獨立入口：

- 使用者拍攝端：`http://localhost:5173/?order_id=<ORDER_UUID>`（由 iRent 本次訂單頁帶入）
- 營運管理端：http://localhost:5173/admin
- 資料標註端：http://localhost:5173/admin/labeling
- Swagger：http://localhost:8000/docs；健康檢查：http://localhost:8000/health

## Docker Compose

執行 `docker compose up --build`，再開啟 http://localhost:8080。SQLite 與上傳圖片存放在 `carma_data` volume。

## 測試與正式建置

```bash
.venv/bin/python -m pytest
cd frontend
npm run build
npm audit
```

## Demo 操作流程

seed 會建立四個合成展示情境，不會修改官方原始照片：

1. `DEMO-BLUR`：左前照片模糊，顯示具體重拍提示。
2. `DEMO-DAMAGE`：還車左前有合成刮痕，顯示差異熱區與維修任務。
3. `DEMO-DIRTY`：車內合成雜物紋理超過門檻，建立清潔任務。
4. `DEMO-CLEAN`：六個位置完整且無異常，自動完成。

使用者端不建立或選擇訂單，而是由 iRent 本次訂單頁以 `order_id` 開啟，自動顯示唯讀訂單編號、車牌與車款。系統會先要求完成取車 4 張外觀基準照；只有同一訂單已完成基準照，才可開始還車 2 張車內＋4 張外觀拍攝及新車損比對。每張照片都會立即檢查。營運管理端是獨立入口，可篩選案件、查看取車／還車／差異熱區三圖、保存人工判定、建立派工並更新狀態。

車外照片會以 Qwen3-VL 的目標車第二階段檢查讀取車牌字號，移除連字號與空白後和訂單車牌比對；
車牌不可讀或字號不符時要求重拍。模型測試頁可選填預期車牌。此功能仍屬 VLM OCR，正式上線前應以
獨立車牌偵測器與 OCR 模型做交叉驗證，並持續收集夜間、反光及斜角車牌案例。

### 即時拍攝防呆

每個位置都有俯視站位圖或車內座位範圍圖。手機拍照／選檔後，後端會立即回傳逐項結果：

- 硬性重拍：解析度不足、模糊、過暗／過曝、鏡頭疑似霧化、重複照片、車外未偵測到車輛、距離太遠／太近、車外照放入車內欄位、車內只拍到局部區域。
- 提醒確認：車身貼近單一邊界、車牌形狀候選不明、左右／前後精確角度。
- 車牌目前由 CV 與 Qwen 目標車二次檢核判斷是否清楚入鏡，不代表已讀出或驗證車牌號碼。
- 六角度採嚴格比對：DINOv2＋Linear SVM 的最終判定位置必須與官方要求完全相同；左右、前後或車內外任一不符都會要求重拍，不提供使用者自行放行。

車輛偵測器會讀取以下不納入 Git 的模型檔：

```text
models/yolov4-tiny.cfg
models/yolov4-tiny.weights
```

若模型不存在，服務仍可啟動並降級使用 OpenCV 輪廓規則；也可設定 `CARMA_VEHICLE_DETECTOR_ENABLED=false` 明確停用。正式環境應由部署流程從受控模型庫提供權重並驗證 checksum。

## 官方資料格式與風險

Excel 欄位為 `order_number`、`CarNo`、`ImageType`、`Image`、`下載網址`、`查看圖片`、`上傳分類(常有誤)`、`時間`。ImageType 1～4 是四個車外角度，10、11 是前後車內，5～9 是其他或異常補拍。

盤點共 27,367 筆、4,820 個訂單、3,121 輛車；只有 2,526 個訂單恰有六張。上傳分類不可當可靠 ground truth，索賠資料夾也只提供訂單層級弱標籤。本 MVP 不下載外部資料、不改動原始 Excel、流程圖或照片。
### 依 ImageType 建立弱標籤資料夾

以下指令不搬移或修改原始圖片；預設以 symbolic link 建立七類資料夾，並輸出 UTF-8 CSV 清單：

```bash
.venv/bin/python scripts/split_by_image_type.py
```

人工瀏覽建議使用扁平模式，每個類別直接放圖片，並將車牌與訂單編入唯一檔名：

```bash
.venv/bin/python scripts/split_by_image_type.py --flat --destination datasets/by_image_type_flat
```

七個類別為 `left_front`、`right_front`、`left_rear`、`right_rear`、`interior_front`、`interior_rear`、`other`。一般模式保留 `CarNo/order_number/filename` 結構；扁平模式使用 `CarNo__order_number__filename`。兩者的 `manifest.csv` 都會記錄原始來源、ImageType、分類與連結狀態。

這些分類仍是依 ImageType 建立的弱標籤，不可直接視為人工確認的 ground truth。腳本可安全重複執行；使用 `--dry-run` 只預覽，`--mode copy` 才會實際複製圖片。

### 人工標註頁面

啟動後直接開啟 http://localhost:5173/admin/labeling。第一次進入會從
`datasets/by_image_type_flat/manifest.csv` 同步資料，目前的照片會依序出現在待檢查佇列。

每張照片可執行：

- 按 `1`～`7` 選擇左前、右前、左後、右後、車前座、車後座或其他。
- 按 `Enter` 接受目前分類；若選擇的類別不同，會保存為已改分類。
- 選擇重拍原因後按 `R`，標記照片不符合拍攝規定。
- 按 `X` 排除不適合訓練的資料，按 `S` 或右方向鍵稍後再看。
- 按 `Z` 復原上一筆，按 `Q`／`E` 只旋轉畫面預覽。
- 「匯出 CSV」會輸出目前全部人工標註，供後續訓練或抽樣複核。

標註結果保存於 SQLite 的 `dataset_annotations` 資料表。「需要重拍」及「排除」都只更新狀態，
不會刪除、搬移或修改 `downloaded_images` 中的原始照片；重新同步也不會覆寫既有人工標註。

因原始影像含車牌等資料，標註 API 僅接受本機連線。若服務在遠端主機，請透過 SSH 轉發：

```bash
ssh -L 5173:127.0.0.1:5173 -L 8000:127.0.0.1:8000 使用者@伺服器
```

再由自己的電腦開啟 http://localhost:5173/admin/labeling。不要將標註 API 或原始圖片目錄直接公開到網際網路。
`/admin` 路由只完成介面分離，並不等於身分驗證；正式公開部署前仍必須加入管理員登入、角色權限與後端 API 授權。

### 第一輪輕量重拍基準

先產出人工標註資料健檢報告：

```bash
.venv/bin/python scripts/analyze_annotations.py
```

再使用專案既有的 MobileNetV2 ONNX 倒數第二層（1,280 維）作為固定特徵，
並以 OpenCV 線性 SVM 建立「可接受／需要重拍」二元基準：

```bash
.venv/bin/python scripts/train_retake_baseline.py
```

目前 246 筆已檢查資料中，排除 31 張不確定資料後，可訓練樣本為 215 張：
可接受 109、需要重拍 106，涵蓋 40 個訂單。驗證依車輛與訂單分組，同一訂單不會同時出現在
訓練集與測試集。5 折各包含 8 個未見訂單及 43 張照片。

第一輪結果為準確率 66.1%、平衡準確率 66.0%、重拍召回率 62.3%。
106 張應重拍照片中仍漏掉 40 張，因此模型產物只供基準比較，**不會接入即時阻擋流程**。
報告位於 `models/retake_baseline/metrics.json`，模型、scaler、特徵快取及資料健檢輸出均不納入 Git。

下一輪應優先增加完整前座與後座的合格照片（目前分別只有 7、6 張），並補充模糊、錯誤角度及
不同光線／車型的重拍反例；至少累積 500 張且獨立測試的重拍召回率達到可接受門檻後，再評估影子模式。

### DINOv2 六角度實驗

首頁 `http://localhost:5173/` 是完整比賽 MVP。測試者只要輸入車牌，系統會在背景建立不顯示於
使用者介面的展示訂單。完成 4 張取車外觀基準照後會先回到租程畫面；使用者實際還車時點擊
「開始還車拍攝」，才會進入「6 張還車照片 → 車牌核對 → 取還車車損比對 → 風險結果與後台預警」。
重新整理頁面時，同一瀏覽器分頁會接續尚未完成的流程。

工程用的單張模型測試頁保留在 `http://localhost:5173/test`，可單獨驗證 DINOv2、Qwen3-VL 與
CV 品質規則，但它不是比賽展示首頁。`http://localhost:5173/capture` 與首頁使用相同完整流程，
管理後台位於 `http://localhost:5173/admin`。

使用相同人工確認資料與相同訂單分組 5 折切法，比較 frozen DINOv2 ViT-S/14 特徵加 OpenCV Linear SVM：

```bash
source .venv/bin/activate
python scripts/train_dinov2_angle_classifier.py --device cuda --batch-size 32
```

官方模型快取與訓練產物會存放在 `models/`，不納入 Git。目前測試環境已將 DINOv2 設為
六角度主要分類器；若需對照舊模型，可在 `.env` 設定
`CARMA_ANGLE_CLASSIFIER_BACKEND=mobilenetv2` 後重啟後端。完整比較見
`docs/dinov2-angle-experiment.md`。

## 已實作功能

雲端正式資料可使用 Cloud SQL PostgreSQL 與 Cloud Storage 掛載；建立與重新部署步驟見
`deploy/cloudrun/README.md`。未設定這兩項時，Cloud Run smoke 版本仍只使用暫存 SQLite 與
`/tmp` 上傳目錄。

- 11 個資料模型、訂單與完整六位置取還車流程。
- Cloud Run CPU 使用 DINOv2 ViT-S/14 ONNX 特徵與 Linear SVM 判斷六個指定位置，
  不需在公開 Web 服務配置 GPU；Qwen 可在後續作為低信心或車損複核服務。
- 六位置站位導引、逐項即時回饋、YOLO 車輛入鏡／距離／裁切檢查，以及模糊、曝光、尺寸、車內範圍、重複與車牌候選規則。
- ORB／Homography 對齊、光照正規化、SSIM、熱區及 bounding box。
- 車內整潔度三級 Demo 分類、集中設定的風險引擎與判斷原因。
- Dashboard 篩選、案件詳情、四種派工、六種任務狀態與人工判定紀錄。
- 統一 API 錯誤格式、Swagger、seed、pytest 與 Docker Compose。

## 目前限制與正式模型替換

六角度目前使用 DINOv2＋Linear SVM，車牌文字由 Qwen3-VL 讀取；YOLO 車框及 VLM OCR 仍可能受遮擋、夜間、反光或特殊車型影響。車損 pipeline 已融合 Qwen 配對判斷與 ORB／SSIM 對齊熱區，但陰影、雨水或角度差異仍需人工複核。尚未加入正式身分驗證、物件儲存、通知與 PostgreSQL migration。

正式模型可實作既有三個 interface，並在 `backend/app/main.py` 的 analyzer 注入位置替換，資料庫與 API 契約不需改變。建議用人工判定資料訓練角度分類、車牌／車體偵測、Siamese change detection 或 segmentation 與整潔度分類器。
