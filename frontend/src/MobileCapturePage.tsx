import { useEffect, useMemo, useState } from "react";
import { api } from "./api";

type ViewKey = "left_front" | "right_front" | "left_rear" | "right_rear" | "interior_front" | "interior_rear";
type Guide = { label:string; title:string; instruction:string; checks:string[]; type:"exterior"|"interior" };

const EXTERIOR: ViewKey[] = ["left_front", "right_front", "left_rear", "right_rear"];
const RETURN: ViewKey[] = ["interior_front", "interior_rear", ...EXTERIOR];
const GUIDE: Record<ViewKey, Guide> = {
  interior_front: { label:"前車內", title:"完整拍下前座空間", instruction:"站在副駕側車門外，讓方向盤、中控台、駕駛座與副駕座主要椅面入鏡。", checks:["方向盤與中控台","前排座椅主要椅面","腳踏區清楚可見"], type:"interior" },
  interior_rear: { label:"後車內", title:"完整拍下後排座椅", instruction:"打開後車門，稍微後退，讓整排椅面、椅背與主要腳踏區入鏡。", checks:["整排後座椅面","後座椅背","主要腳踏區"], type:"interior" },
  left_front: { label:"左前", title:"站在車輛左前方", instruction:"以車輛行進方向判斷左右，斜拍車頭與左側車身，前車牌清楚入鏡。", checks:["完整車頭","車輛左側","前車牌清楚"], type:"exterior" },
  right_front: { label:"右前", title:"站在車輛右前方", instruction:"以車輛行進方向判斷左右，斜拍車頭與右側車身，前車牌清楚入鏡。", checks:["完整車頭","車輛右側","前車牌清楚"], type:"exterior" },
  left_rear: { label:"左後", title:"站在車輛左後方", instruction:"以車輛行進方向判斷左右，斜拍車尾與左側車身，後車牌清楚入鏡。", checks:["完整車尾","車輛左側","後車牌清楚"], type:"exterior" },
  right_rear: { label:"右後", title:"站在車輛右後方", instruction:"以車輛行進方向判斷左右，斜拍車尾與右側車身，後車牌清楚入鏡。", checks:["完整車尾","車輛右側","後車牌清楚"], type:"exterior" }
};

const EXAMPLES: Record<ViewKey,string> = {
  interior_front:"/guides/interior-front-example.webp", interior_rear:"/guides/interior-rear-example.webp",
  left_front:"/guides/left-front-example.webp", right_front:"/guides/right-front-example.webp",
  left_rear:"/guides/left-rear-example.webp", right_rear:"/guides/right-rear-example.webp"
};

function accepted(item:any) {
  return item?.quality?.passed && item.quality.metrics?.capture_decision !== "retake";
}

function Quality({ item }:{item:any}) {
  if (!item) return null;
  const ok = accepted(item);
  return <section className={`mobile-quality ${ok ? "ok" : "bad"}`}>
    <div className="mobile-quality-head"><span>{ok ? "✓" : "!"}</span><div><strong>{ok ? "照片通過" : "需要重拍"}</strong><small>AI 品質分數 {item.quality?.quality_score ?? "—"}</small></div></div>
    {!ok && <p>{item.quality?.retake_instruction}</p>}
  </section>;
}

function EvidencePanel({ items, busy, onUpload, title="有其他車損要補充嗎？" }:{
  items:any[]; busy:boolean; onUpload:(purpose:string,file?:File)=>void; title?:string;
}) {
  return <section className="mobile-evidence">
    <div><strong>{title}</strong><small>不計入必拍張數，照片會連同時間與本次訂單保存。</small></div>
    <div className="mobile-evidence-actions">
      <label>＋ 車損位置照<input type="file" accept="image/*" capture="environment" disabled={busy} onChange={(e)=>onUpload("damage_context",e.target.files?.[0])}/></label>
      <label>＋ 車損近照<input type="file" accept="image/*" capture="environment" disabled={busy} onChange={(e)=>onUpload("damage_closeup",e.target.files?.[0])}/></label>
    </div>
    {busy && <p>照片上傳中…</p>}
    {items.length>0 && <div className="mobile-evidence-list">{items.map((item)=><span key={item.id}><img src={item.image_url}/><b>{item.purpose==="damage_context"?"位置照":"近照"}</b></span>)}</div>}
  </section>;
}

export default function MobileCapturePage
({ notify }:{notify:(message:string)=>void}) {
  const [order,setOrder] = useState<any>(null);
  const [orderId,setOrderId] = useState("");
  const [orderState,setOrderState] = useState<"loading"|"ready"|"missing"|"error">("loading");
  const [demoPlate,setDemoPlate] = useState("");
  const [kind,setKind] = useState<"pickup"|"return">("pickup");
  const [session,setSession] = useState<any>(null);
  const [uploads,setUploads] = useState<Record<string,any>>({});
  const [step,setStep] = useState(0);
  const [busy,setBusy] = useState("");
  const [result,setResult] = useState<any>(null);
  const [knownDamages,setKnownDamages] = useState<any>({items:[],all_acknowledged:true});
  const [evidence,setEvidence] = useState<any[]>([]);
  const [evidenceBusy,setEvidenceBusy] = useState(false);
  const sequence = kind === "pickup" ? EXTERIOR : RETURN;
  const view = sequence[step];
  const item = uploads[view];
  const done = useMemo(()=>sequence.filter((key)=>accepted(uploads[key])).length,[uploads,kind]);

  function applyOrder(data:any) {
    const pickupCompleted=data.inspections?.some((inspection:any)=>inspection.inspection_type==="pickup" && inspection.status==="completed");
    setOrder(data); setOrderId(data.id); setKind(pickupCompleted?"return":"pickup"); setOrderState("ready");
  }
  useEffect(()=>{
    const linkedOrderId=new URLSearchParams(window.location.search).get("order_id") || window.localStorage.getItem("carma_mvp_order_id");
    if (!linkedOrderId) { setOrderState("missing"); return; }
    api.order(linkedOrderId).then(applyOrder).catch((error:any)=>{
      window.localStorage.removeItem("carma_mvp_order_id");
      setOrderState("error"); notify(error.message);
    });
  },[]);
  useEffect(()=>{
    if (!orderId) return setKnownDamages({items:[],all_acknowledged:true});
    api.knownDamages(orderId).then(setKnownDamages).catch((error:any)=>notify(error.message));
  },[orderId]);

  async function acknowledgeDamageHistory() {
    try {
      await api.acknowledgeKnownDamages(orderId);
      setKnownDamages((old:any)=>({...old,all_acknowledged:true,items:old.items.map((item:any)=>({...item,acknowledged:true}))}));
      notify("既有車損已確認並留下時間紀錄");
    } catch(error:any) { notify(error.message); }
  }

  async function uploadEvidence(purpose:string,file?:File) {
    if (!file || !session) return;
    setEvidenceBusy(true);
    try {
      const data=await api.uploadEvidence(session.id,purpose,file);
      setEvidence((old)=>[...old,data]);
      notify("補充蒐證已保存，不計入必拍張數");
    } catch(error:any) { notify(error.message); }
    finally { setEvidenceBusy(false); }
  }

  async function start() {
    if (!orderId || !order) return notify("請先輸入車牌，開始完整測試");
    if (kind==="pickup" && knownDamages.items.length && !knownDamages.all_acknowledged) return notify("請先確認前次既有車損");
    setBusy("start");
    try {
      const data=await api.createInspection({order_id:orderId,inspection_type:kind,location:"黑客松 MVP 展示站"});
      setSession(data); setUploads({}); setEvidence([]); setStep(0); setResult(null);
      notify(`${kind==="pickup"?"取車":"還車"}拍攝開始，共 ${sequence.length} 張`);
    } catch(e:any){notify(e.message);} finally{setBusy("");}
  }
  async function startDemo() {
    const plate=demoPlate.trim().toUpperCase().replace(/\s+/g,"");
    if (plate.length<4) return notify("請輸入有效的測試車牌");
    setBusy("start"); setOrderState("loading");
    try {
      const existing=await api.ordersByPlate(plate);
      for (const candidate of existing.filter((item:any)=>item.plate_number===plate).slice(0,5)) {
        const detail=await api.order(candidate.id);
        const returnCompleted=detail.inspections?.some((inspection:any)=>inspection.inspection_type==="return" && inspection.status==="completed");
        if (!returnCompleted) {
          window.localStorage.setItem("carma_mvp_order_id",detail.id);
          applyOrder(detail);
          setKnownDamages(await api.knownDamages(detail.id));
          notify("已找回這台車尚未完成的租程，接續上次進度");
          return;
        }
      }
      const created=await api.createOrder({
        order_number:`MVP-${Date.now()}-${Math.random().toString(36).slice(2,7).toUpperCase()}`,
        plate_number:plate,
        make_model:"iRent 展示車輛",
        pickup_location:"黑客松 MVP 展示站",
        return_location:"黑客松 MVP 展示站"
      });
      window.localStorage.setItem("carma_mvp_order_id",created.id);
      setOrder(created); setOrderId(created.id); setKind("pickup"); setOrderState("ready");
      const history=await api.knownDamages(created.id);
      setKnownDamages(history);
      if (history.items?.length && !history.all_acknowledged) {
        notify("請先確認這台車的既有車損，再開始取車拍攝");
        return;
      }
      const inspection=await api.createInspection({order_id:created.id,inspection_type:"pickup",location:"黑客松 MVP 展示站"});
      setSession(inspection); setUploads({}); setEvidence([]); setStep(0); setResult(null);
      notify("完整 MVP 已開始：先拍攝 4 張取車基準照");
    } catch(error:any) {
      setOrderState("error"); notify(error.message);
    } finally { setBusy(""); }
  }
  async function upload(file?:File) {
    if (!file || !session) return;
    setBusy("upload");
    try {
      const data=await api.upload(session.id,view,file);
      setUploads((old)=>({...old,[view]:data}));
      notify(data.quality.passed ? "完成即時審核" : data.quality.retake_instruction);
    } catch(e:any){notify(e.message);} finally{setBusy("");}
  }
  async function finish() {
    setBusy("finish");
    try { const data=await api.analyze(session.id); setResult(data); notify(kind==="pickup"?"取車基準照已保存":"還車分析與預警已完成"); }
    catch(e:any){notify(e.message);} finally{setBusy("");}
  }
  function reset() {
    setSession(null); setUploads({}); setResult(null); setStep(0);
    if (orderId) api.order(orderId).then(applyOrder).catch((error:any)=>notify(error.message));
  }
  async function continueAfterResult() {
    if (kind==="pickup") {
      setSession(null); setUploads({}); setEvidence([]); setStep(0); setResult(null); setKind("return");
      if (orderId) api.order(orderId).then(applyOrder).catch((error:any)=>notify(error.message));
      notify("取車已完成；實際還車時，請點擊「開始還車拍攝」");
      return;
    }
    window.localStorage.removeItem("carma_mvp_order_id");
    setOrder(null); setOrderId(""); setKind("pickup"); setSession(null); setUploads({}); setEvidence([]); setStep(0); setResult(null); setDemoPlate(""); setOrderState("missing");
    notify("完整租程測試已完成，可以開始下一台車");
  }

  if (result) return <main className="mobile-capture">
    <div className="mobile-appbar"><button onClick={reset}>‹</button><strong>{kind==="pickup"?"確認取車":"完成還車"}</strong><span/></div>
    <section className="mobile-final">
      <span className={result.risk?.risk_level === "red" ? "danger" : "success"}>{result.risk?.risk_level === "red" ? "!" : "✓"}</span>
      <h1>{kind==="pickup"?"取車基準照已完成":result.risk?.risk_level==="green"?"車況檢查完成":"已送交營運人員確認"}</h1>
      <p>{kind==="pickup"?"四個車身角度已保存，還車時會以相同角度進行新車損比對。":result.risk?.reasons?.join("、")}</p>
      <div className="mobile-save-receipt"><span>✓</span><div><strong>紀錄已安全保存</strong><small>{order?.plate_number} · {new Date().toLocaleString("zh-TW")}</small></div></div>
      {kind==="return" && <div className="mobile-damage-notice"><strong>後台預警流程</strong><p>疑似新車損會建立預警，營運人員可比對取車／還車照片、確認異常並建立維修或索賠複核任務。</p></div>}
      {kind==="return" && result.risk?.risk_level!=="green" && <EvidencePanel title="請補充疑似車損照片" items={evidence} busy={evidenceBusy} onUpload={uploadEvidence}/>}
      <button className="mobile-primary" onClick={continueAfterResult} disabled={!!busy}>{kind==="pickup"?"返回租程畫面":"開始新的完整測試"}</button>
    </section>
  </main>;

  if (!session) return <main className="mobile-capture">
    <div className="mobile-appbar mobile-homebar"><span className="mobile-brand"><i>i</i>Rent</span><strong>車況紀錄</strong><span className="mobile-secure">租程專用</span></div>
    <section className="mobile-hero mobile-home-hero"><div><span>AI VEHICLE CHECK</span><h1>{kind==="pickup"?"取車前，先留下安心紀錄":"準備還車了嗎？"}</h1><p>{kind==="pickup"?"跟著站位引導拍攝，AI 當下檢查角度、清晰度與車牌。":"點擊開始後，依序完成車內與車外拍攝；系統會與取車紀錄逐張比對。"}</p></div><span className="mobile-ai-badge"><i/>拍攝即檢查</span></section>
    <section className="mobile-setup-card mobile-home-card">
      {orderState==="loading" && <div className="mobile-order-state">正在準備完整 MVP 流程…</div>}
      {(orderState==="missing" || orderState==="error") && <section className="mobile-demo-start"><small>COMPETITION MVP</small><h2>開始車況紀錄</h2><p>輸入車牌即可開始；若雲端已有未完成租程，系統會自動接續。</p><label><span>車牌號碼</span><input value={demoPlate} onChange={(event)=>setDemoPlate(event.target.value)} onKeyDown={(event)=>event.key==="Enter"&&startDemo()} placeholder="例如 RFY-6613" autoCapitalize="characters"/></label><button className="mobile-primary" type="button" onClick={startDemo} disabled={!!busy}>{busy?"正在確認租程…":"確認車輛並開始"}</button><div className="mobile-capabilities"><span><i>01</i>角度防呆</span><span><i>02</i>車牌核對</span><span><i>03</i>車損比對</span></div></section>}
      {order && kind==="return" && <section className="mobile-rental-status"><span className="mobile-live-dot"/><div><small>目前租程</small><strong>車輛使用中</strong><p>取車基準照已完成。準備還車時，再啟動下方檢查。</p></div><b>取車完成</b></section>}
      {order && <section className="mobile-vehicle-card"><div className="mobile-card-heading"><div><small>本次測試車輛</small><strong>{order.make_model || "iRent 租賃車輛"}</strong></div><span>完整 MVP</span></div><div className="mobile-plate">{order.plate_number}</div><div className="mobile-order-meta"><span><small>目前階段</small><b>{kind==="pickup"?"取車基準":"還車檢查"}</b></span><span><small>展示地點</small><b>{order.pickup_location || "黑客松 MVP 展示站"}</b></span></div></section>}
      {order && <>
      {kind==="pickup" && knownDamages.items.length>0 && <section className="mobile-known-damage"><div className="mobile-known-head"><span>!</span><div><strong>此車已有 {knownDamages.items.length} 筆車損紀錄</strong><small>這些是前次已確認車損，不會歸責於本次租程。</small></div></div><div className="mobile-known-list">{knownDamages.items.map((damage:any)=><p key={damage.id}><b>{GUIDE[damage.view as ViewKey]?.label || damage.view}</b><span>{damage.description}</span><time>{new Date(damage.confirmed_at).toLocaleString("zh-TW")}</time></p>)}</div>{knownDamages.all_acknowledged?<em>✓ 已留下確認紀錄</em>:<button type="button" onClick={acknowledgeDamageHistory}>我已確認上述既有車損</button>}</section>}
      <section className="mobile-journey"><div className={kind==="pickup"?"active":"done"}><i>{kind==="pickup"?"1":"✓"}</i><span><b>取車紀錄</b><small>4 張外觀基準照</small></span></div><em/><div className={kind==="return"?"active":"locked"}><i>2</i><span><b>還車檢查</b><small>6 張照片與車損比對</small></span></div></section>
      <section className="mobile-task"><div className="mobile-section-heading"><span><small>接下來</small><strong>{kind==="pickup"?"拍攝 4 張取車基準照":"拍攝 6 張還車檢查照"}</strong></span><b>約 {kind==="pickup"?"2":"3"} 分鐘</b></div><div className="mobile-flow-preview">{(kind==="pickup"?EXTERIOR:RETURN).map((key,index)=><span key={key}><i>{index+1}</i><b>{GUIDE[key].label}</b><small>{GUIDE[key].type==="exterior"?"整車＋車牌":"座椅＋腳踏區"}</small></span>)}</div></section>
      <button className="mobile-primary" onClick={start} disabled={orderState!=="ready" || !!busy || (kind==="pickup" && knownDamages.items.length>0 && !knownDamages.all_acknowledged)}>{busy?"建立流程中…":`開始${kind==="pickup"?"取車":"還車"}拍攝`}</button>
      </>}
  
    </section>
    <p className="mobile-disclaimer">雲端加密保存 · 僅供本次租程車況紀錄與爭議查核</p>
  </main>;

  if (step >= sequence.length) return <main className="mobile-capture">
    <div className="mobile-appbar"><button onClick={()=>setStep(sequence.length-1)}>‹</button><strong>確認照片</strong><span>{done}/{sequence.length}</span></div>
    <section className="mobile-summary-head"><span>✓</span><h1>所有照片已通過</h1><p>送出後，{kind==="pickup"?"將保存為還車比對基準。":"系統會比對取車照片並建立必要預警。"}</p></section>
    <section className="mobile-summary-grid">{sequence.map((key,index)=><button key={key} onClick={()=>setStep(index)}><img src={uploads[key]?.image?.image_url}/><span>✓ {GUIDE[key].label}</span></button>)}</section>
    {kind==="return" && <EvidencePanel items={evidence} busy={evidenceBusy} onUpload={uploadEvidence}/>}
    <div className="mobile-bottom-action"><button className="mobile-primary" onClick={finish} disabled={!!busy}>{busy?"AI 分析中…":kind==="pickup"?"確認取車":"送出並完成還車分析"}</button></div>
  </main>;

  const guide=GUIDE[view];
  const ok=accepted(item);
  return <main className="mobile-capture">
    <div className="mobile-appbar"><button onClick={()=>step?setStep(step-1):reset()}>‹</button><strong>{kind==="pickup"?"取車拍照":"還車拍照"}</strong><span>{step+1}/{sequence.length}</span></div>
    <div className="mobile-progress"><i style={{width:`${(step+1)/sequence.length*100}%`}}/></div>
    <div className="mobile-step-rail">{sequence.map((key,index)=><span key={key} className={index<step?"done":index===step?"active":""}><i>{index<step?"✓":index+1}</i><small>{GUIDE[key].label}</small></span>)}</div>
    <section className="mobile-shot-title"><small>第 {step+1} 張 · {guide.label}</small><h1>{guide.title}</h1><p>{guide.instruction}</p></section>
    <section className="mobile-camera-card">
      {item?.image?.image_url ? <img className="mobile-captured-image" src={item.image.image_url} alt={guide.label}/> : <div className="mobile-example-art"><img src={EXAMPLES[view]} alt={`${guide.label}正確成品範例`}/><span className="correct-badge">✓ 正確成品範例</span><strong>{guide.type==="interior"?"照這個範圍拍攝":"整車入鏡・四周留白"}</strong></div>}
      {busy==="upload" && <div className="mobile-analyzing"><i/>AI 正在檢查照片…</div>}
    </section>
    <section className="mobile-instruction">
      <strong>照片中要包含</strong>
      <div className="mobile-checks">{guide.checks.map((check)=><span key={check}>{check}</span>)}</div>
    </section>
    <Quality item={item}/>
    <div className="mobile-capture-actions">
      {item && <label className="mobile-secondary">重新拍攝<input type="file" accept="image/*" capture="environment" onChange={(e)=>upload(e.target.files?.[0])}/></label>}
      {!item && <label className="mobile-camera-button"><span>開啟相機拍照</span><small>拍完會立即進行 AI 檢查</small><input type="file" accept="image/*" capture="environment" onChange={(e)=>upload(e.target.files?.[0])}/></label>}
      {ok && <button className="mobile-primary" onClick={()=>setStep(step+1)}>{step===sequence.length-1?"確認全部照片":"使用這張，下一步"}</button>}
    </div>
  </main>;
}
