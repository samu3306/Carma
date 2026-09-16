#!/usr/bin/env python3
"""Local-only browser demo for Qwen3-VL photo guidance."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from src.vlm.photo_guidance import EXTERIOR_VIEWS, build_photo_guidance
from src.vlm.prompts import plate_visibility_prompt
from src.vlm.qwen_vl_client import ImageLoadError, InferenceError, ModelLoadError, QwenVLClient
from src.vlm.schemas import DETECTED_VIEWS


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
client = QwenVLClient(MODEL_NAME)
inference_lock = threading.Lock()
app = FastAPI(title="Carma 拍照建議測試", version="1.0")


PAGE = r"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carma 拍照建議測試</title><style>
:root{font-family:Inter,"Noto Sans TC",sans-serif;color:#20242a;background:#f4f5f7}*{box-sizing:border-box}
body{margin:0}.wrap{width:min(1040px,calc(100% - 28px));margin:30px auto}.head{margin-bottom:18px}.head p{color:#68707b}
.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:16px}.card{background:white;border:1px solid #e1e3e7;border-radius:16px;padding:20px;box-shadow:0 8px 28px #20242a12}
.drop{display:grid;place-items:center;min-height:430px;border:2px dashed #c8ccd2;border-radius:13px;overflow:hidden;background:#f8f9fa;cursor:pointer;text-align:center}
.drop img{display:none;width:100%;height:430px;object-fit:contain;background:#202226}.drop span{padding:25px;color:#747b84}
label{display:block;margin-bottom:14px;font-size:13px;font-weight:700}select,input,button{width:100%;min-height:44px;margin-top:7px;border:1px solid #cfd3d8;border-radius:9px;padding:9px;background:white}
button{border:0;color:white;background:#bd1017;font-weight:800;cursor:pointer}button:disabled{opacity:.5}.result{display:none}.badge{display:inline-block;border-radius:99px;padding:6px 10px;font-size:12px;font-weight:800}.pass{color:#08754c;background:#e6f7ef}.review{color:#7a4b00;background:#fff0c7}.fail{color:#a31218;background:#ffeaeb}
.issues{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}.issues span{padding:5px 8px;border-radius:6px;background:#fff0d7;color:#8a5511;font-size:11px}.advice{padding:13px;border-left:4px solid #bd1017;background:#fff3f3;line-height:1.7}.meta{color:#747b84;font-size:12px;line-height:1.8}.error{color:#a31218;white-space:pre-wrap}.notice{font-size:11px;color:#7a818a;line-height:1.6}
@media(max-width:760px){.grid{grid-template-columns:1fr}.drop,.drop img{min-height:300px;height:300px}}
</style></head><body><main class="wrap"><section class="head"><h1>Carma 拍照建議測試</h1><p>上傳單張取還車照片，Qwen3-VL 判斷後由固定模板產生重拍建議。</p></section>
<div class="grid"><section class="card"><label class="drop" for="file"><img id="preview"><span id="hint">點擊或拖曳圖片到此處<br><small>JPG / PNG / WEBP，最多 15 MB</small></span></label><input id="file" type="file" accept="image/*" hidden></section>
<section class="card"><form id="form"><label>預期拍攝角度<select id="view" name="expected_view"><option value="left_front">左前</option><option value="right_front">右前</option><option value="left_rear">左後</option><option value="right_rear">右後</option><option value="interior_front">車前座</option><option value="interior_rear">車後座</option><option value="other">其他／不支援視角</option></select></label><button id="submit" type="submit">開始分析</button></form>
<p class="notice">本頁僅供內部測試。模型可能漏掉輕微模糊，不應直接作為自動退件依據。</p><div id="loading" hidden>模型分析中，請稍候…</div><div id="error" class="error"></div>
<div id="result" class="result"><h2>分析結果</h2><span id="badge" class="badge"></span><div id="issues" class="issues"></div><h3>拍照建議</h3><div id="advice" class="advice"></div><h3>模型判斷</h3><p id="explanation"></p><div id="meta" class="meta"></div></div></section></div></main>
<script>
const file=document.querySelector('#file'),preview=document.querySelector('#preview'),hint=document.querySelector('#hint'),form=document.querySelector('#form'),button=document.querySelector('#submit'),loading=document.querySelector('#loading'),error=document.querySelector('#error'),result=document.querySelector('#result');
file.onchange=()=>{if(!file.files[0])return;preview.src=URL.createObjectURL(file.files[0]);preview.style.display='block';hint.style.display='none'};
document.querySelector('.drop').ondragover=e=>e.preventDefault();document.querySelector('.drop').ondrop=e=>{e.preventDefault();file.files=e.dataTransfer.files;file.onchange()};
form.onsubmit=async e=>{e.preventDefault();error.textContent='';result.style.display='none';if(!file.files[0]){error.textContent='請先選擇圖片';return}button.disabled=true;loading.hidden=false;const body=new FormData();body.append('file',file.files[0]);body.append('expected_view',document.querySelector('#view').value);
try{const response=await fetch('/api/analyze',{method:'POST',body});const data=await response.json();if(!response.ok)throw new Error(data.detail||'分析失敗');const g=data.guidance;result.style.display='block';const badge=document.querySelector('#badge'),decision=g.decision||(g.passed?'pass':'retake');badge.textContent=decision==='pass'?'照片通過':(decision==='review'?'需人工確認':'需要重拍');badge.className='badge '+(decision==='pass'?'pass':(decision==='review'?'review':'fail'));document.querySelector('#issues').innerHTML=g.issues.map(x=>`<span>${x}</span>`).join('');document.querySelector('#advice').textContent=g.retake_instruction||'照片符合目前檢查條件，不需要重拍。';document.querySelector('#explanation').textContent=g.model_explanation||'—';document.querySelector('#meta').innerHTML=`預期角度：${g.expected_view}<br>偵測角度：${g.detected_view}<br>信心：${g.model_confidence}<br>角度方法：${data.angle_method==='manual_review_required'?'人工確認':'zero-shot'}<br>車牌方法：${data.plate_method==='target_vehicle_second_pass'?'目標車二次檢查':'基礎模型'}<br>推論：${data.inference_seconds.toFixed(3)} 秒<br>GPU 峰值：${data.gpu_peak_memory_mb.toFixed(1)} MiB`;}catch(e){error.textContent=e.message}finally{button.disabled=false;loading.hidden=true}};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), expected_view: str = Form(...)) -> dict:
    if expected_view not in DETECTED_VIEWS:
        raise HTTPException(422, "不支援的預期角度")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, "只接受圖片檔")
    body = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(body) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "圖片超過 15 MB")
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="carma-guidance-", suffix=suffix, delete=False) as handle:
            handle.write(body)
            temporary_path = Path(handle.name)
        with Image.open(temporary_path) as image:
            image.verify()
        plate_result = None
        with inference_lock:
            result = client.infer(temporary_path)
            if result.json_valid and result.prediction is not None and expected_view in EXTERIOR_VIEWS:
                plate_result = client.infer(
                    temporary_path,
                    prompt_text=plate_visibility_prompt(expected_view),
                )
        if not result.json_valid or result.prediction is None:
            return {
                "success": False, "json_valid": False, "error": result.error,
                "raw_model_output": result.raw_output,
                "inference_seconds": result.inference_seconds,
                "gpu_peak_memory_mb": result.gpu_peak_memory_mb,
            }
        prediction = dict(result.prediction)
        inference_seconds = result.inference_seconds
        gpu_peak_memory_mb = result.gpu_peak_memory_mb
        plate_method = "base_vlm"
        if plate_result is not None:
            inference_seconds += plate_result.inference_seconds
            gpu_peak_memory_mb = max(gpu_peak_memory_mb, plate_result.gpu_peak_memory_mb)
            if plate_result.json_valid and plate_result.prediction is not None:
                plate_prediction = plate_result.prediction
                original_issues = list(prediction.get("quality_issues", []))
                remaining_issues = [issue for issue in original_issues if issue != "plate_not_visible"]
                prediction["quality_issues"] = remaining_issues
                prediction["plate_visible"] = bool(plate_prediction.get("plate_visible"))
                if not prediction["plate_visible"]:
                    prediction["quality_passed"] = False
                elif original_issues == ["plate_not_visible"]:
                    prediction["quality_passed"] = True
                prediction["explanation"] = (
                    f"目標車牌檢查：{plate_prediction.get('explanation', '')} "
                    f"整體判斷：{prediction.get('explanation', '')}"
                )
                plate_method = "target_vehicle_second_pass"
        angle_review_required = False
        angle_method = "zero_shot"
        return {
            "success": True,
            "json_valid": True,
            "guidance": build_photo_guidance(
                prediction, expected_view, angle_review_required=angle_review_required
            ),
            "model_prediction": prediction,
            "angle_method": angle_method,
            "reference_images": 0,
            "plate_method": plate_method,
            "inference_seconds": inference_seconds,
            "gpu_peak_memory_mb": gpu_peak_memory_mb,
        }
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(422, f"圖片無法開啟：{error}") from error
    except (ModelLoadError, ImageLoadError, InferenceError) as error:
        raise HTTPException(500, str(error)) from error
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
