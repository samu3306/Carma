const jsonHeaders = { "Content-Type": "application/json" };

async function request(path: string, options: RequestInit = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload?.error?.message || "系統暫時無法完成操作");
  }
  return payload;
}

export const api = {
  orders: () => request("/api/orders"),
  ordersByPlate: (plate: string) => request(`/api/orders?plate=${encodeURIComponent(plate)}`),
  createOrder: (body: any) => request("/api/orders", { method: "POST", headers: jsonHeaders, body: JSON.stringify(body) }),
  order: (id: string) => request(`/api/orders/${id}`),
  knownDamages: (orderId: string) => request(`/api/orders/${orderId}/known-damages`),
  acknowledgeKnownDamages: (orderId: string) => request(`/api/orders/${orderId}/known-damages/acknowledge`, { method: "POST" }),
  createInspection: (body: any) => request("/api/inspections", { method: "POST", headers: jsonHeaders, body: JSON.stringify(body) }),
  uploadEvidence: (inspectionId: string, purpose: string, file: File, relatedView?: string, notes?: string) => {
    const body = new FormData();
    body.append("purpose", purpose);
    if (relatedView) body.append("related_view", relatedView);
    if (notes) body.append("notes", notes);
    body.append("file", file);
    return request(`/api/inspections/${inspectionId}/evidence`, { method: "POST", body });
  },
  upload: (inspectionId: string, view: string, file: File) => {
    const body = new FormData();
    body.append("expected_view", view);
    body.append("file", file);
    return request(`/api/inspections/${inspectionId}/images`, { method: "POST", body });
  },
  testQuality: (view: string, file: File, expectedPlate = "") => {
    const body = new FormData();
    body.append("expected_view", view);
    if (expectedPlate.trim()) body.append("expected_plate", expectedPlate.trim());
    body.append("file", file);
    return request("/api/model-test/quality", { method: "POST", body });
  },
  analyze: (id: string) => request(`/api/inspections/${id}/analyze`, { method: "POST" }),
  summary: () => request("/api/dashboard/summary"),
  cases: (query = "") => request(`/api/cases${query}`),
  case: (id: string) => request(`/api/cases/${id}`),
  review: (id: string, body: any) => request(`/api/cases/${id}/review`, { method: "POST", headers: jsonHeaders, body: JSON.stringify(body) }),
  createTask: (id: string, body: any) => request(`/api/cases/${id}/tasks`, { method: "POST", headers: jsonHeaders, body: JSON.stringify(body) }),
  updateTask: (id: string, body: any) => request(`/api/tasks/${id}`, { method: "PATCH", headers: jsonHeaders, body: JSON.stringify(body) }),
  acknowledgeAlert: (id: string) => request(`/api/alerts/${id}/acknowledge`, { method: "PATCH" }),
  labelStats: () => request("/api/labeling/stats"),
  labelItems: (query = "") => request(`/api/labeling/items${query ? `?${query}` : ""}`),
  syncLabels: () => request("/api/labeling/sync", { method: "POST" }),
  updateLabel: (id: string, body: any) => request(`/api/labeling/items/${id}`, { method: "PATCH", headers: jsonHeaders, body: JSON.stringify(body) })
};
