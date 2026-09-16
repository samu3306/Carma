import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import LabelingPage from "./LabelingPage";
import MobileCapturePage from "./MobileCapturePage";
import ModelTestPage from "./ModelTestPage";

const VIEWS = [
  ["left_front", "左前", "↖"],
  ["right_front", "右前", "↗"],
  ["left_rear", "左後", "↙"],
  ["right_rear", "右後", "↘"],
  ["interior_front", "前車內", "▤"],
  ["interior_rear", "後車內", "▥"]
];
const GUIDES: any = {
  left_front: { title:"站在車輛左前方", detail:"鏡頭對準車頭與左側車身，車牌及四個車輪盡量入鏡", orientation:"直拍／橫拍皆可", marker:"left-front", type:"exterior" },
  right_front: { title:"站在車輛右前方", detail:"鏡頭對準車頭與右側車身；不確定左右時，看俯視圖紅點", orientation:"直拍／橫拍皆可", marker:"right-front", type:"exterior" },
  left_rear: { title:"站在車輛左後方", detail:"拍到車尾、左側車身與後車牌，避免只拍局部鈑金", orientation:"直拍／橫拍皆可", marker:"left-rear", type:"exterior" },
  right_rear: { title:"站在車輛右後方", detail:"拍到車尾、右側車身與後車牌，車身約占畫面一半", orientation:"直拍／橫拍皆可", marker:"right-rear", type:"exterior" },
  interior_front: { title:"拍攝完整前車室", detail:"涵蓋方向盤、中控區，以及至少一張前座的主要椅面與椅背；另一張前座可部分遮擋", orientation:"橫拍較佳，直拍可審核", marker:"front-cabin", type:"interior" },
  interior_rear: { title:"拍攝完整後車室", detail:"拍到整排後座、椅面、椅背及主要腳踏區；以內容完整為準", orientation:"橫拍較佳，直拍可審核", marker:"rear-cabin", type:"interior" }
};

function captureAccepted(item: any) {
  return item?.quality?.passed && item.quality.metrics?.capture_decision === "accept";
}

const riskMeta: any = {
  green: { label: "正常", icon: "✓" },
  yellow: { label: "待複核", icon: "!" },
  red: { label: "高風險", icon: "⚠" }
};

function CaptureGuide({ view }: { view:string }) {
  const guide = GUIDES[view];
  return <div className={`capture-guide ${guide.type}`}>
    {guide.type === "exterior" ? <div className="top-car"><i className={guide.marker}/><span>車頭</span></div> : <div className={`cabin-map ${guide.marker}`}><i/><i/><i/><i/></div>}
    <div><strong>{guide.title}</strong><b className="orientation-tip">↔ {guide.orientation}</b><small>{guide.detail}</small></div>
  </div>;
}

function Toast({ message, onClose }: any) {
  if (!message) return null;
  return <button className="toast" onClick={onClose} aria-label="關閉通知">{message}<span>×</span></button>;
}

function DemoBadge() {
  return <span className="demo-badge">DEMO 規則推論</span>;
}

function QwenBadge() {
  return <span className="qwen-badge">Qwen3-VL 即時審核</span>;
}

function CapturePage({ notify }: any) {
  const [orders, setOrders] = useState<any[]>([]);
  const [orderId, setOrderId] = useState("");
  const [inspectionType, setInspectionType] = useState("return");
  const [inspection, setInspection] = useState<any>(null);
  const [uploads, setUploads] = useState<any>({});
  const [busy, setBusy] = useState("");
  const [result, setResult] = useState<any>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({ order_number: "", plate_number: "", pickup_location: "台北車站 Demo 站" });

  const loadOrders = () => api.orders().then((data) => {
    setOrders(data);
    if (!orderId && data[0]) setOrderId(data[0].id);
  }).catch((error) => notify(error.message));

  useEffect(() => { loadOrders(); }, []);
  const passed = useMemo(() => VIEWS.filter(([key]) => captureAccepted(uploads[key])).length, [uploads]);

  async function createOrder(event: any) {
    event.preventDefault();
    try {
      const created = await api.createOrder(form);
      await loadOrders();
      setOrderId(created.id);
      setShowCreate(false);
      notify("訂單已建立");
    } catch (error: any) { notify(error.message); }
  }

  async function start() {
    if (!orderId) return notify("請先選擇訂單");
    setBusy("start");
    try {
      const data = await api.createInspection({ order_id: orderId, inspection_type: inspectionType, location: "台北車站 Demo 站" });
      setInspection(data); setUploads({}); setResult(null);
      notify("檢查流程已開始，請依序拍攝六個位置");
    } catch (error: any) { notify(error.message); }
    finally { setBusy(""); }
  }

  async function upload(view: string, file?: File) {
    if (!inspection || !file) return;
    setBusy(view);
    try {
      const data = await api.upload(inspection.id, view, file);
      setUploads((old: any) => ({ ...old, [view]: data }));
      notify(data.quality.passed ? "審核通過，照片已保存" : data.quality.retake_instruction);
    } catch (error: any) { notify(error.message); }
    finally { setBusy(""); }
  }

  async function analyze() {
    setBusy("analyze");
    try {
      const data = await api.analyze(inspection.id);
      setResult(data); notify("六個角度分析完成");
    } catch (error: any) { notify(error.message); }
    finally { setBusy(""); }
  }

  return <main className="capture-layout">
    <section className="capture-hero">
      <div><p className="eyebrow">DRIVER INSPECTION</p><h1>取還車，拍完就安心</h1></div>
      <QwenBadge />
    </section>

    <section className="setup-card">
      <div className="field grow"><label>租車訂單／車牌</label><select value={orderId} onChange={e => setOrderId(e.target.value)}>
        <option value="">請選擇訂單</option>{orders.map(order => <option key={order.id} value={order.id}>{order.order_number} · {order.plate_number}</option>)}
      </select></div>
      <div className="segmented"><button className={inspectionType === "pickup" ? "active" : ""} onClick={() => setInspectionType("pickup")}>取車</button><button className={inspectionType === "return" ? "active" : ""} onClick={() => setInspectionType("return")}>還車</button></div>
      <button className="btn primary" onClick={start} disabled={!!busy}>{inspection ? "重新開始" : "開始檢查"}</button>
      <button className="btn ghost" onClick={() => setShowCreate(!showCreate)}>＋ 新訂單</button>
    </section>

    {showCreate && <form className="create-form" onSubmit={createOrder}>
      <input required placeholder="訂單編號" value={form.order_number} onChange={e => setForm({...form, order_number:e.target.value})}/>
      <input required placeholder="車牌號碼" value={form.plate_number} onChange={e => setForm({...form, plate_number:e.target.value})}/>
      <input placeholder="取車地點" value={form.pickup_location} onChange={e => setForm({...form, pickup_location:e.target.value})}/>
      <button className="btn primary">建立訂單</button>
    </form>}

    <div className="progress-block">
      <div className="progress-copy"><span>拍攝進度</span><strong>{passed} / 6 合格</strong></div>
      <div className="progress"><i style={{width:`${passed / 6 * 100}%`}} /></div>
    </div>

    <section className="view-grid">
      {VIEWS.map(([key, label], index) => {
        const item = uploads[key];
        const checks = item?.quality?.metrics?.checks || [];
        const decision = item?.quality?.metrics?.capture_decision;
        const ok = captureAccepted(item);
        const failed = item && !item.quality?.passed;
        const pending = item?.quality?.passed && !ok;
        return <article className={`view-card ${ok ? "passed" : failed ? "failed" : pending ? "pending" : ""}`} key={key}>
          <div className="view-visual">
            {item?.image?.image_url ? <img src={item.image.image_url} alt={label}/> : <CaptureGuide view={key}/>}
            {busy === key && <div className="reviewing-overlay"><span className="spinner"/>AI 即時審核中…</div>}
            <span className="state-icon">{ok ? "✓" : failed ? "!" : pending ? "?" : index + 1}</span>
          </div>
          <div className="view-content"><div><h3>{label}</h3><p>{ok ? `品質 ${item.quality.quality_score} 分 · 已通過` : pending ? "拍攝位置未完全符合，請重新拍攝" : failed ? item.quality.retake_instruction : GUIDES[key].detail}</p></div>
            <label className={`upload-button ${busy === key ? "loading" : ""}`}>
            {item ? "重新拍攝" : "拍攝／上傳"}
              <input type="file" accept="image/*" capture="environment" disabled={!inspection || !!busy} onChange={e => upload(key, e.target.files?.[0])}/>
            </label>
          </div>
          {checks.length > 0 && <div className="capture-review">
          {item && <details className="guide-details" open={pending}><summary>查看 {label} 站位圖</summary><CaptureGuide view={key}/></details>}
            <div className={`review-decision ${decision}`}>
              <strong>{decision === "retake" || decision === "confirm" ? "請重新拍攝" : "AI 檢查通過"}</strong>
              <span>{decision === "retake" || decision === "confirm" ? "拍攝位置必須與官方要求完全相同" : "可進行下一個角度"}</span>
            </div>
            <div className="check-grid">{checks.map((check:any) => <div className={`capture-check ${check.status}`} key={check.code} title={check.message}>
              <span>{check.status === "pass" ? "✓" : check.status === "fail" ? "!" : "?"}</span>
              <div><strong>{check.label}</strong><small>{check.instruction || check.message}</small></div>
              <em>{check.mode === "model_pending" ? "待確認" : check.status === "pass" ? "通過" : check.status === "fail" ? "重拍" : "複核"}</em>
            </div>)}</div>
            <p className="review-limit">拍攝後立即以 Qwen3-VL 檢查六方向、必要內容與目標車牌；解析度、模糊、曝光及重複照由固定規則檢查。車內照片不檢查車牌，且不進行車牌文字比對。</p>
          </div>}
        </article>;
      })}
    </section>

    <section className="finish-card">
      <div><h2>{passed === 6 ? "照片已齊全" : `還差 ${6 - passed} 個合格角度`}</h2><p>缺少必要角度或品質不合格時，系統不會直接完成檢查。</p></div>
      <button className="btn primary large" onClick={analyze} disabled={passed !== 6 || !!busy}>{busy === "analyze" ? "分析中…" : "完成並執行 AI 分析"}</button>
    </section>

    {result && <section className={`result-panel ${result.risk.risk_level}`}>
      <div className="risk-orb"><strong>{result.risk.risk_score}</strong><span>風險分數</span></div>
      <div><p className="eyebrow">分析完成 · DEMO MODE</p><h2>{riskMeta[result.risk.risk_level].icon} {riskMeta[result.risk.risk_level].label}</h2>
        <p>{result.risk.recommended_action}</p><ul>{result.risk.reasons.map((reason:string) => <li key={reason}>{reason}</li>)}</ul></div>
    </section>}
  </main>;
}

function StatCard({ icon, label, value, tone }: any) {
  return <article className={`stat-card ${tone || ""}`}><span>{icon}</span><div><strong>{value ?? "—"}</strong><p>{label}</p></div></article>;
}

function Dashboard({ notify }: any) {
  const [summary, setSummary] = useState<any>(null);
  const [cases, setCases] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [filters, setFilters] = useState({ risk:"", plate:"", order_number:"", anomaly_type:"", status:"" });
  const [busy, setBusy] = useState(false);

  async function load() {
    setBusy(true);
    try {
      const params = new URLSearchParams(Object.entries(filters).filter(([,v]) => v) as any).toString();
      const [s, c] = await Promise.all([api.summary(), api.cases(params ? `?${params}` : "")]);
      setSummary(s); setCases(c);
      if (selected) setSelected(await api.case(selected.id));
    } catch (error:any) { notify(error.message); }
    finally { setBusy(false); }
  }
  useEffect(() => { load(); }, [filters.risk, filters.anomaly_type, filters.status]);

  async function openCase(id:string) {
    setBusy(true);
    try { setSelected(await api.case(id)); } catch(error:any) { notify(error.message); }
    finally { setBusy(false); }
  }

  async function task(type:string) {
    try {
      await api.createTask(selected.id, { task_type:type, priority: type === "reshoot" ? "normal" : "high", notes:"由營運 Dashboard 建立" });
      notify("任務已建立"); setSelected(await api.case(selected.id)); await load();
    } catch(error:any) { notify(error.message); }
  }

  async function review(decision:string) {
    try {
      await api.review(selected.id, { reviewer:"Demo 營運專員", decision, notes:"Dashboard 人工判定", labels:{source:"hackathon_demo"} });
      notify("人工判定已保存，可供未來回訓"); setSelected(await api.case(selected.id));
    } catch(error:any) { notify(error.message); }
  }

  async function acknowledgeAlert(alertId:string) {
    try {
      await api.acknowledgeAlert(alertId);
      notify("預警已確認"); setSelected(await api.case(selected.id)); await load();
    } catch(error:any) { notify(error.message); }
  }

  async function advanceTask(taskItem:any) {
    const sequence:any = {pending:"reviewing", reviewing:"assigned", assigned:"processing", processing:"completed"};
    const next = sequence[taskItem.status] || "completed";
    try {
      await api.updateTask(taskItem.id, {status:next});
      notify(`任務已更新為 ${next}`); setSelected(await api.case(selected.id));
    } catch(error:any) { notify(error.message); }
  }

  return <main className="dashboard">
    <section className="dash-heading"><div><p className="eyebrow">OPERATIONS CONTROL</p><h1>車況營運中心</h1><p>把注意力留給真正需要處理的車。</p></div><div className="live"><i/> 即時監控 <DemoBadge/></div></section>
    <section className="stats">
      <StatCard icon="◎" label="今日檢查" value={summary?.today_inspections}/>
      <StatCard icon="✓" label="正常完成" value={summary?.normal_count} tone="good"/>
      <StatCard icon="!" label="待人工複核" value={summary?.review_count} tone="warn"/>
      <StatCard icon="⚠" label="高風險" value={summary?.high_risk_count} tone="danger"/>
      <StatCard icon="⌁" label="疑似新車損" value={summary?.new_damage_count}/>
      <StatCard icon="✦" label="髒污案件" value={summary?.dirty_count}/>
    </section>
    <section className="case-shell">
      <div className="case-list">
        <div className="section-title"><div><h2>異常案件</h2><p>平均處理 {summary?.average_processing_seconds ?? 0} 秒</p></div><button className="icon-btn" onClick={load}>↻</button></div>
        <div className="filters">
          <select value={filters.risk} onChange={e => setFilters({...filters,risk:e.target.value})}><option value="">全部風險</option><option value="red">高風險</option><option value="yellow">待複核</option><option value="green">正常</option></select>
          <select value={filters.anomaly_type} onChange={e => setFilters({...filters,anomaly_type:e.target.value})}><option value="">全部異常</option><option value="damage">車損</option><option value="cleanliness">髒污</option></select>
          <select value={filters.status} onChange={e => setFilters({...filters,status:e.target.value})}><option value="">全部狀態</option><option value="attention_required">待處理</option><option value="completed">已完成</option></select>
          <input placeholder="搜尋車牌" value={filters.plate} onChange={e => setFilters({...filters,plate:e.target.value})} onKeyDown={e => e.key === "Enter" && load()}/>
          <input placeholder="搜尋訂單" value={filters.order_number} onChange={e => setFilters({...filters,order_number:e.target.value})} onKeyDown={e => e.key === "Enter" && load()}/>
          <button onClick={load}>搜尋</button>
        </div>
        <div className="case-rows">
          {cases.map(item => <button className={`case-row ${selected?.id === item.id ? "selected" : ""}`} key={item.id} onClick={() => openCase(item.id)}>
            <span className={`risk-tag ${item.risk_level}`}>{riskMeta[item.risk_level].icon} {item.risk_score}</span>
            <span><strong>{item.plate_number}</strong><small>{item.order_number}</small></span>
            <span className="case-reason">{item.has_damage ? "疑似新車損" : item.has_cleanliness_issue ? "車內髒污" : item.reasons[0]}</span>
            <b>›</b>
          </button>)}
          {!busy && !cases.length && <div className="empty">沒有符合篩選條件的案件</div>}
        </div>
      </div>

      <aside className="case-detail">
        {!selected ? <div className="empty detail-empty"><span>⌁</span><h3>選擇案件查看 AI 比對</h3><p>照片、熱區、風險原因與派工都會顯示在這裡。</p></div> : <>
          <div className="detail-head"><div><span className={`risk-tag ${selected.risk_level}`}>{riskMeta[selected.risk_level].icon} {riskMeta[selected.risk_level].label}</span><h2>{selected.plate_number}</h2><p>{selected.order_number} · {selected.order?.pickup_location}</p><p>取車 {selected.order?.pickup_time ? new Date(selected.order.pickup_time).toLocaleString("zh-TW") : "未記錄"} · 還車 {selected.order?.return_time ? new Date(selected.order.return_time).toLocaleString("zh-TW") : "未記錄"}</p></div><div className="score"><strong>{selected.risk_score}</strong><small>/ 100</small></div></div>
          <div className="view-completeness">{VIEWS.map(([key,label]) => {
            const returned = selected.order?.inspections?.find((item:any) => item.inspection_type === "return");
            const image = returned?.images?.find((item:any) => item.expected_view === key && item.quality_status === "passed");
            return <span className={image ? "complete" : "missing"} key={key}>{image ? "✓" : "!"} {label}</span>;
          })}</div>

          {selected.alerts?.length > 0 && <div className="alert-list"><h3>異常預警</h3>{selected.alerts.map((alert:any)=><div className={alert.acknowledged ? "acknowledged" : "active"} key={alert.id}><span>⚠</span><p><strong>{alert.alert_type === "new_damage" ? "疑似新車損" : "檢查異常"}</strong><small>{alert.message}</small></p>{alert.acknowledged ? <em>已確認</em> : <button onClick={()=>acknowledgeAlert(alert.id)}>確認預警</button>}</div>)}</div>}
          <div className="reason-box"><strong>AI 判斷理由</strong>{selected.reasons.map((reason:string)=><p key={reason}>• {reason}</p>)}<small>車損由 Qwen 借還照片配對初篩，異常仍須由營運人員確認。</small></div>
          <h3>取車／還車影像比對</h3>
          <div className="comparisons">{selected.damage_comparisons?.map((item:any) => {
            const pickup = selected.order.inspections.find((i:any)=>i.inspection_type==="pickup")?.images.find((im:any)=>im.expected_view===item.view);
            const returned = selected.order.inspections.find((i:any)=>i.inspection_type==="return")?.images.find((im:any)=>im.expected_view===item.view);
            return <div className="comparison" key={item.id}><div><span>取車</span><img src={pickup?.image_url}/></div><div><span>還車</span><img src={returned?.image_url}/></div><div><span>差異熱區</span><img src={item.heatmap_url}/></div><p>{item.explanation} · 信心 {Math.round(item.confidence*100)}%</p></div>;
          })}</div>
          {selected.cleanliness?.length > 0 ? <div className="cleanliness-list"><h3>車內整潔度</h3>{selected.cleanliness.map((item:any,index:number)=><div key={index}><span className={`cleanliness ${item.cleanliness_level}`}>{item.cleanliness_level}</span><strong>{item.cleanliness_score} 分</strong><p>{item.explanation} · 信心 {Math.round(item.confidence*100)}%</p></div>)}</div> : <div className="reason-box"><strong>車內整潔度</strong><p>目前尚未啟用；本階段只分析外觀新車損。</p></div>}
          <h3>營運動作</h3>
          <div className="actions"><button onClick={()=>task("cleaning")}>✦ 建立清潔</button><button onClick={()=>task("repair")}>⌁ 建立維修</button><button onClick={()=>task("claim_review")}>▣ 索賠複核</button><button onClick={()=>task("reshoot")}>↻ 要求補拍</button></div>
          <div className="review-actions"><button className="btn primary" onClick={()=>review("confirmed_damage")}>確認異常</button><button className="btn ghost" onClick={()=>review("approved")}>判定正常</button></div>
          {selected.tasks?.length > 0 && <div className="task-list"><h3>任務追蹤</h3>{selected.tasks.map((item:any)=><button key={item.id} onClick={()=>advanceTask(item)}><span>{item.task_type}</span><strong>{item.status}</strong><small>點擊推進狀態 ›</small></button>)}</div>}
          {selected.reviews?.length > 0 && <div className="audit"><h3>人工判定紀錄</h3>{selected.reviews.map((item:any)=><p key={item.id}><strong>{item.decision}</strong> · {item.reviewer}<br/><small>{item.notes}</small></p>)}</div>}
        </>}
      </aside>
    </section>
  </main>;
}

export default function App() {
  const [route, setRoute] = useState(window.location.pathname);
  const [toast, setToast] = useState("");

  useEffect(() => {
    const syncRoute = () => setRoute(window.location.pathname);
    window.addEventListener("popstate", syncRoute);
    return () => window.removeEventListener("popstate", syncRoute);
  }, []);

  function notify(message:string) {
    setToast(message);
    window.setTimeout(()=>setToast(""), 3800);
  }
  function navigate(path:string) {
    window.history.pushState({}, "", path);
    setRoute(path);
  }

  const isAdmin = route === "/admin" || route.startsWith("/admin/");
  if (route === "/test" || route.startsWith("/test/")) {
    return <><ModelTestPage notify={notify}/><Toast message={toast} onClose={()=>setToast("")}/></>;
  }
  if (route === "/" || route === "/capture" || route.startsWith("/capture/")) {
    return <><MobileCapturePage notify={notify}/><Toast message={toast} onClose={()=>setToast("")}/></>;
  }
  if (!isAdmin) {
    return <><ModelTestPage notify={notify}/><Toast message={toast} onClose={()=>setToast("")}/></>;
  }

  const adminPage = route.startsWith("/admin/labeling") ? "labeling" : "dashboard";
  return <><header className="topbar admin-topbar">
    <button className="brand" onClick={()=>navigate("/admin")}><span>i</span>Rent <b>車況營運後台</b></button>
    <nav>
      <button className={adminPage==="dashboard"?"active":""} onClick={()=>navigate("/admin")}>案件管理</button>
      <button className={adminPage==="labeling"?"active":""} onClick={()=>navigate("/admin/labeling")}>資料標註</button>
      <button onClick={()=>navigate("/test")}>模型測試</button>
    </nav>
    <span className="system-ok">● 管理端系統正常</span>
  </header>
    {adminPage === "dashboard" ? <Dashboard notify={notify}/> : <LabelingPage notify={notify}/>}
    <Toast message={toast} onClose={()=>setToast("")}/></>;
}
