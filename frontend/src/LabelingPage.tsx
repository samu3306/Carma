import { useEffect, useState } from "react";
import { api } from "./api";
import "./labeling.css";

const LABEL_OPTIONS = [
  ["left_front", "左前", "1"], ["right_front", "右前", "2"],
  ["left_rear", "左後", "3"], ["right_rear", "右後", "4"],
  ["interior_front", "車前座", "5"], ["interior_rear", "車後座", "6"],
  ["other", "其他", "7"]
] as const;
const LABEL_NAMES: Record<string, string> = Object.fromEntries(
  LABEL_OPTIONS.map(([value, label]) => [value, label])
);
const RETAKE_REASONS = [
  ["wrong_scene", "拍錯位置／場景"], ["wrong_angle", "角度不符"],
  ["partial", "車輛或車室不完整"], ["blur", "影像模糊"],
  ["too_dark", "太暗"], ["overexposed", "過曝"],
  ["duplicate", "重複照片"], ["non_vehicle", "非車輛照片"],
  ["corrupt", "檔案損壞"], ["other", "其他"]
] as const;

function LabelStat({ icon, label, value, tone = "" }: any) {
  return <article className={`stat-card ${tone}`}><span>{icon}</span><div><strong>{value ?? "—"}</strong><p>{label}</p></div></article>;
}

export default function LabelingPage({ notify }: any) {
  const [stats, setStats] = useState<any>(null);
  const [items, setItems] = useState<any[]>([]);
  const [queueTotal, setQueueTotal] = useState(0);
  const [status, setStatus] = useState("unreviewed");
  const [category, setCategory] = useState("");
  const [search, setSearch] = useState("");
  const [selectedLabel, setSelectedLabel] = useState("other");
  const [reason, setReason] = useState("");
  const [notes, setNotes] = useState("");
  const [rotation, setRotation] = useState(0);
  const [busy, setBusy] = useState(false);
  const [lastAction, setLastAction] = useState<any>(null);
  const current = items[0];

  async function load(showBusy = true) {
    if (showBusy) setBusy(true);
    try {
      const params = new URLSearchParams({ review_status: status, limit: "50" });
      if (category) params.set("category", category);
      if (search.trim()) params.set("search", search.trim());
      const [nextStats, result] = await Promise.all([api.labelStats(), api.labelItems(params.toString())]);
      setStats(nextStats);
      setItems(result.items);
      setQueueTotal(result.total);
    } catch (error:any) {
      notify(error.message);
    } finally {
      if (showBusy) setBusy(false);
    }
  }

  useEffect(() => { load(); }, [status, category]);
  useEffect(() => {
    if (!current) return;
    setSelectedLabel(current.actual_label || current.original_label || "other");
    setReason(current.retake_reason || "");
    setNotes(current.notes || "");
    setRotation(0);
  }, [current?.id]);

  function previousPayload(item:any) {
    return {
      actual_label: item.actual_label, acceptable: item.acceptable,
      retake_reason: item.retake_reason, review_status: item.review_status,
      notes: item.notes, reviewer: item.reviewer || "本機標註員"
    };
  }

  async function save(action:"approve"|"invalid"|"excluded") {
    if (!current || busy) return;
    if (action === "invalid" && !reason) return notify("請先選擇需要重拍的原因");
    const body = action === "approve" ? {
      actual_label: selectedLabel, acceptable: true, retake_reason: null,
      review_status: selectedLabel === current.original_label ? "approved" : "relabelled",
      notes: notes || null, reviewer: "本機標註員"
    } : {
      actual_label: null, acceptable: false,
      retake_reason: action === "invalid" ? reason : null,
      review_status: action === "invalid" ? "invalid" : "excluded",
      notes: notes || null, reviewer: "本機標註員"
    };
    setBusy(true);
    try {
      await api.updateLabel(current.id, body);
      setLastAction({ item: current, payload: previousPayload(current) });
      setItems(old => old.slice(1));
      setQueueTotal(old => Math.max(0, old - 1));
      const nextStats = await api.labelStats();
      setStats(nextStats);
      if (status === "unreviewed" && items.length === 1 && queueTotal > 1) await load(false);
      notify(action === "approve" ? "標註已儲存" : action === "invalid" ? "已標記需要重拍" : "已排除，原圖仍保留");
    } catch (error:any) {
      notify(error.message);
    } finally {
      setBusy(false);
    }
  }

  function skip() {
    if (items.length > 1) setItems(old => [...old.slice(1), old[0]]);
  }

  async function undo() {
    if (!lastAction || busy) return;
    setBusy(true);
    try {
      await api.updateLabel(lastAction.item.id, lastAction.payload);
      setLastAction(null);
      await load(false);
      notify("已復原上一筆標註");
    } catch (error:any) {
      notify(error.message);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    function onKey(event:KeyboardEvent) {
      const target = event.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      const option = LABEL_OPTIONS.find(([, , key]) => key === event.key);
      if (option) setSelectedLabel(option[0]);
      else if (event.key === "Enter") { event.preventDefault(); save("approve"); }
      else if (event.key.toLowerCase() === "r") save("invalid");
      else if (event.key.toLowerCase() === "x") save("excluded");
      else if (event.key.toLowerCase() === "s" || event.key === "ArrowRight") skip();
      else if (event.key.toLowerCase() === "z") undo();
      else if (event.key.toLowerCase() === "q") setRotation(value => value - 90);
      else if (event.key.toLowerCase() === "e") setRotation(value => value + 90);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current?.id, selectedLabel, reason, notes, busy, items, lastAction]);

  const reviewed = stats?.reviewed || 0;
  const total = stats?.total || 0;
  return <main className="labeling-page">
    <section className="label-head">
      <div><p className="eyebrow">LOCAL DATA REVIEW</p><h1>影像資料標註</h1><p>確認預分類、標記需要重拍的照片；資料只寫入標註表，原圖不會移動或刪除。</p></div>
      <div className="label-head-actions"><a className="btn ghost" href="/api/labeling/export.csv">匯出 CSV</a><button className="btn primary" disabled={busy} onClick={async()=>{ await api.syncLabels(); await load(); notify("資料清單已同步"); }}>同步資料</button></div>
    </section>
    <section className="label-stats">
      <LabelStat icon="▣" label="照片總數" value={total}/>
      <LabelStat icon="✓" label="已完成" value={reviewed} tone="good"/>
      <LabelStat icon="…" label="待檢查" value={stats?.remaining || 0} tone="warn"/>
      <LabelStat icon="!" label="需重拍" value={stats?.by_status?.invalid || 0} tone="danger"/>
    </section>
    <section className="label-progress"><div style={{width:`${stats?.progress_percent || 0}%`}}/><span>{stats?.progress_percent || 0}% · {reviewed} / {total}</span></section>
    <section className="label-filters">
      <select value={status} onChange={event=>setStatus(event.target.value)}><option value="unreviewed">待檢查</option><option value="">全部</option><option value="approved">分類正確</option><option value="relabelled">已改分類</option><option value="invalid">需要重拍</option><option value="excluded">已排除</option></select>
      <select value={category} onChange={event=>setCategory(event.target.value)}><option value="">全部目前分類</option>{LABEL_OPTIONS.map(([value,label])=><option value={value} key={value}>{label}</option>)}</select>
      <input value={search} onChange={event=>setSearch(event.target.value)} onKeyDown={event=>event.key==="Enter"&&load()} placeholder="搜尋車牌、訂單或檔名"/>
      <button onClick={()=>load()}>搜尋</button><span>此篩選剩餘 {queueTotal} 張</span>
    </section>
    <section className="label-shell">
      <div className="label-image-stage">
        {current ? <img className="label-image" src={current.image_url} style={{transform:`rotate(${rotation}deg)`}} alt="待標註車輛"/> : <div className="empty label-empty"><span>✓</span><h2>這個清單已完成</h2><p>可切換篩選條件，或匯出 CSV 檢查結果。</p></div>}
        {current && <><div className="label-counter">目前佇列 {items.length} / {queueTotal}</div><div className="rotate-tools"><button onClick={()=>setRotation(value=>value-90)}>Q ↶</button><button onClick={()=>setRotation(value=>value+90)}>E ↷</button></div></>}
      </div>
      <aside className="label-controls">
        {current ? <>
          <div className="label-meta"><span>目前選擇</span><strong>{LABEL_NAMES[selectedLabel] || selectedLabel}</strong><small>原始預分類：{LABEL_NAMES[current.original_label] || current.original_label}</small><p title={current.source_path}>{current.source_path}</p></div>
          <h3>這張照片實際是哪一類？</h3>
          <div className="label-options">{LABEL_OPTIONS.map(([value,label,key])=><button className={selectedLabel===value?"active":""} key={value} onClick={()=>setSelectedLabel(value)}><kbd>{key}</kbd>{label}</button>)}</div>
          <label className="label-field">若不合格，選擇原因<select value={reason} onChange={event=>setReason(event.target.value)}><option value="">請選擇重拍原因</option>{RETAKE_REASONS.map(([value,label])=><option value={value} key={value}>{label}</option>)}</select></label>
          <label className="label-field">備註（選填）<textarea value={notes} onChange={event=>setNotes(event.target.value)} rows={3} placeholder="例如：只拍到車頭、前座誤放車外照"/></label>
          <div className="label-actions"><button className="approve" disabled={busy} onClick={()=>save("approve")}><kbd>Enter</kbd> 分類正確／接受</button><button className="invalid" disabled={busy} onClick={()=>save("invalid")}><kbd>R</kbd> 需要重拍</button><button disabled={busy} onClick={()=>save("excluded")}><kbd>X</kbd> 排除資料</button></div>
          <div className="label-secondary"><button onClick={skip}><kbd>S / →</kbd> 稍後再看</button><button disabled={!lastAction||busy} onClick={undo}><kbd>Z</kbd> 復原上一筆</button></div>
          <p className="label-safety">「需要重拍」與「排除」只改標註狀態，不會刪除原始照片。</p>
        </> : <div className="empty"><p>沒有符合條件的照片。</p></div>}
      </aside>
    </section>
  </main>;
}
