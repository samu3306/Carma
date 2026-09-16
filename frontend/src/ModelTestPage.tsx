import { useEffect, useState } from "react";
import { api } from "./api";
import "./model-test.css";

const VIEWS = [
  ["left_front", "左前"], ["right_front", "右前"],
  ["left_rear", "左後"], ["right_rear", "右後"],
  ["interior_front", "車前座"], ["interior_rear", "車後座"],
];

export default function ModelTestPage({ notify }:{ notify:(message:string)=>void }) {
  const [view,setView] = useState("left_front");
  const [expectedPlate,setExpectedPlate] = useState("");
  const [preview,setPreview] = useState("");
  const [result,setResult] = useState<any>(null);
  const [busy,setBusy] = useState(false);

  useEffect(()=>()=>{ if (preview) URL.revokeObjectURL(preview); },[preview]);

  async function upload(file?:File) {
    if (!file) return;
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    setResult(null); setBusy(true);
    try {
      const data = await api.testQuality(view,file,expectedPlate);
      setResult(data);
      notify(data.passed ? "模型判定通過" : data.retake_instruction || "模型判定需要重拍");
    } catch(error:any) { notify(error.message); }
    finally { setBusy(false); }
  }

  const checks = result?.metrics?.checks || [];
  return <main className="model-test-page">
    <header><strong>iRent 車況 AI 測試</strong><span>DINOv2＋Qwen3-VL</span></header>
    <section className="model-test-hero"><p>MODEL PLAYGROUND</p><h1>單張照片快速測試</h1><span>不需訂單、不寫入租程；伺服器暫存照片會在分析完成後刪除。</span></section>
    <section className="model-test-controls">
      <label><span>這張照片應該是哪個位置？</span><select value={view} onChange={e=>{setView(e.target.value);setResult(null);}}>{VIEWS.map(([key,label])=><option key={key} value={key}>{label}</option>)}</select></label>
      <label><span>預期車牌（選填）</span><input value={expectedPlate} placeholder="例如 RFY-6613" onChange={e=>setExpectedPlate(e.target.value.toUpperCase())}/></label>
      <label className="model-test-upload">{busy ? "AI 分析中…" : "選擇照片並測試"}<input type="file" accept="image/jpeg,image/png,image/webp" disabled={busy} onChange={e=>upload(e.target.files?.[0])}/></label>
    </section>
    <section className="model-test-workspace">
      <div className="model-test-preview">{preview ? <img src={preview} alt="測試照片"/> : <div><strong>尚未選擇照片</strong><span>支援 JPG、PNG、WebP，最大 15 MB</span></div>}{busy && <i>正在執行 DINOv2 與品質檢查…</i>}</div>
      <div className="model-test-result">
        {!result ? <div className="model-test-empty">結果會顯示在這裡</div> : <>
          <div className={`model-test-verdict ${result.passed?"pass":"fail"}`}><b>{result.passed?"✓":"!"}</b><div><small>判定結果</small><strong>{result.passed?"照片通過":"需要重拍"}</strong><span>預期 {VIEWS.find(x=>x[0]===result.expected_view)?.[1]} · 模型判定 {VIEWS.find(x=>x[0]===result.detected_view)?.[1] || result.detected_view}</span></div><em>{Math.round((result.model_confidence||0)*100)}%</em></div>
          {result.retake_instruction && <p className="model-test-instruction">{result.retake_instruction}</p>}
          <div className="model-test-checks">{checks.map((check:any)=><div className={check.status} key={check.code}><b>{check.status==="pass"?"✓":check.status==="review"?"?":"!"}</b><span><strong>{check.label}</strong><small>{check.instruction || check.message}</small></span><em>{check.status==="pass"?"通過":check.status==="review"?"複核":"重拍"}</em></div>)}</div>
          <footer>模型版本：{result.model_version}</footer>
        </>}
      </div>
    </section>
  </main>;
}
