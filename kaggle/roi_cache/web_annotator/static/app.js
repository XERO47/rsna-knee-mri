"use strict";

/**
 * Remote ROI landmark annotator client.
 *
 * All scientific state (patient-space XYZ, revisions, resume position) is
 * computed and owned by the server. This script only maps pointer clicks to
 * fractional source-pixel coordinates and renders whatever the server
 * returns; it never computes or trusts a patient coordinate itself.
 */

const state = {
  task: null,
  guidance: null,
  imageBitmap: null,
  imageRows: 0,
  imageColumns: 0,
  sopUid: null,
  sliceIndex: 0,
  zoom: 1,
  panX: 0,
  panY: 0,
  pendingPoint: null, // {row, column} in source pixel space
  saveState: "unsaved", // unsaved | saving | saved | rejected | offline
  isOnline: true,
  pendingRequestId: null,
  detailsOpen: false,
};

const canvas = document.getElementById("slice-canvas");
const ctx = canvas.getContext("2d");

function el(id) {
  return document.getElementById(id);
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  return response;
}

function setSaveState(next) {
  state.saveState = next;
  const badge = el("save-state");
  badge.className = "save-state save-state-" + next;
  const labels = {
    unsaved: "Unsaved point",
    saving: "Saving…",
    saved: "Saved",
    rejected: "Point not saved",
    offline: "Offline—not saved",
  };
  badge.textContent = labels[next] || next;
  el("offline-banner").classList.toggle("hidden", next !== "offline");
  el("rejected-banner").classList.toggle("hidden", next !== "rejected");
}

function landmarkLabel(name) {
  if (state.guidance && state.guidance[name]) return state.guidance[name].title;
  return name;
}

function renderHeader() {
  const task = state.task;
  if (!task) return;
  el("cohort-name").textContent = task.cohort_phase === "pilot" ? "Pilot" : "Final";
  el("reader-name").textContent = task.reader;
  el("patient-progress").textContent = `Patient ${task.patient_index + 1}/${task.patient_count}`;
  el("patients-remaining").textContent = `${task.remaining_patient_count} remaining`;
  el("landmark-progress").textContent = `Landmark ${task.landmark_index + 1}/${task.landmark_count}`;
  el("landmark-name").textContent = landmarkLabel(task.landmark);

  const statesContainer = el("landmark-states");
  statesContainer.innerHTML = "";
  task.landmark_states.forEach((entry, index) => {
    const span = document.createElement("span");
    span.className = "landmark-state state-" + entry.state;
    if (index === task.landmark_index) span.classList.add("state-current");
    span.textContent = `${index + 1}. ${landmarkLabel(entry.landmark)}`;
    statesContainer.appendChild(span);
  });

  const details = el("details-panel");
  details.innerHTML =
    `<span class="progress-text">Study UID: ${task.study_uid}</span>` +
    (task.series
      ? `<span class="progress-text">Series: ${task.series.plane} fluid=${task.series.fluid_sensitive}</span>`
      : "");

  const instructionPanel = el("instruction-panel");
  const guide = state.guidance ? state.guidance[task.landmark] : null;
  const knee = task.laterality === "L" ? "LEFT KNEE" : task.laterality === "R" ? "RIGHT KNEE" : "KNEE SIDE UNKNOWN";
  const anatomySide = task.image_medial_side
    ? ` In this displayed coronal image, MEDIAL is on the ${task.image_medial_side.toUpperCase()} side.`
    : "";
  const correction = task.previous_point
    ? " CORRECTION REQUIRED: the old point is amber; place the corrected point in red, then save."
    : "";
  instructionPanel.textContent = `${knee}.${anatomySide}${correction}` + (guide ? ` ${guide.instruction}` : "");

  el("replace-point").classList.toggle("hidden", task.state !== "pass");
}

function resizeCanvasToContainer() {
  const viewport = el("viewport");
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = viewport.clientWidth;
  const cssHeight = viewport.clientHeight;
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
}

function computeBaseTransform() {
  // Letterbox the source image inside the canvas, preserving aspect ratio.
  const rows = state.imageRows;
  const columns = state.imageColumns;
  if (!rows || !columns) return { scale: 1, offsetX: 0, offsetY: 0 };
  const scale = Math.min(canvas.width / columns, canvas.height / rows);
  const drawWidth = columns * scale;
  const drawHeight = rows * scale;
  const offsetX = (canvas.width - drawWidth) / 2;
  const offsetY = (canvas.height - drawHeight) / 2;
  return { scale, offsetX, offsetY };
}

function drawScene() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!state.imageBitmap) return;
  const base = computeBaseTransform();
  const scale = base.scale * state.zoom;
  const drawWidth = state.imageColumns * scale;
  const drawHeight = state.imageRows * scale;
  const offsetX = base.offsetX - (drawWidth - state.imageColumns * base.scale) / 2 + state.panX;
  const offsetY = base.offsetY - (drawHeight - state.imageRows * base.scale) / 2 + state.panY;
  // The web viewport is deliberately larger than many source MRI matrices
  // (for example 384×384).  Preserve the source pixels and coordinate system,
  // but use the browser's best interpolation while scaling so radiologists do
  // not see a blocky nearest-neighbour enlargement.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(state.imageBitmap, offsetX, offsetY, drawWidth, drawHeight);
  // Keep the previously saved green marker visible while a new red candidate
  // is placed. Saving creates an append-only replacement event on the server;
  // it never deletes the historical point.
  if (state.task && state.task.saved_point) {
    const [x, y] = sourceToCanvas(state.task.saved_point.row_px, state.task.saved_point.column_px);
    drawCrosshair(x, y, "#3fbf7f");
  }
  if (state.task && state.task.previous_point) {
    const [x, y] = sourceToCanvas(state.task.previous_point.row_px, state.task.previous_point.column_px);
    drawCrosshair(x, y, "#e0b93f");
  }
  if (state.pendingPoint) {
    const [x, y] = sourceToCanvas(state.pendingPoint.row, state.pendingPoint.column);
    drawCrosshair(x, y, "#ff3b3b");
  }
}

function drawCrosshair(x, y, color) {
  // Draw dimensions in CSS pixels so the target stays equally visible on
  // standard and Retina displays. A dark outer stroke keeps it legible over
  // bright marrow while the colored inner stroke remains visible on black.
  const dpr = window.devicePixelRatio || 1;
  const outerRadius = 10 * dpr;
  const armLength = 34 * dpr;
  const armGap = 13 * dpr;

  function traceTarget() {
    ctx.beginPath();
    ctx.arc(x, y, outerRadius, 0, Math.PI * 2);
    ctx.moveTo(x - armLength, y);
    ctx.lineTo(x - armGap, y);
    ctx.moveTo(x + armGap, y);
    ctx.lineTo(x + armLength, y);
    ctx.moveTo(x, y - armLength);
    ctx.lineTo(x, y - armGap);
    ctx.moveTo(x, y + armGap);
    ctx.lineTo(x, y + armLength);
  }

  ctx.save();
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(0, 0, 0, 0.9)";
  ctx.lineWidth = 5 * dpr;
  traceTarget();
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5 * dpr;
  traceTarget();
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 2.5 * dpr, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function sourceToCanvas(row, column) {
  const base = computeBaseTransform();
  const scale = base.scale * state.zoom;
  const drawWidth = state.imageColumns * scale;
  const drawHeight = state.imageRows * scale;
  const offsetX = base.offsetX - (drawWidth - state.imageColumns * base.scale) / 2 + state.panX;
  const offsetY = base.offsetY - (drawHeight - state.imageRows * base.scale) / 2 + state.panY;
  return [offsetX + column * scale, offsetY + row * scale];
}

function canvasToSource(canvasX, canvasY) {
  const base = computeBaseTransform();
  const scale = base.scale * state.zoom;
  const drawWidth = state.imageColumns * scale;
  const drawHeight = state.imageRows * scale;
  const offsetX = base.offsetX - (drawWidth - state.imageColumns * base.scale) / 2 + state.panX;
  const offsetY = base.offsetY - (drawHeight - state.imageRows * base.scale) / 2 + state.panY;
  const column = (canvasX - offsetX) / scale;
  const row = (canvasY - offsetY) / scale;
  return { row, column };
}

function clientToCanvasCoordinates(event) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
  return { x, y };
}

canvas.addEventListener("click", (event) => {
  if (!state.imageBitmap) return;
  const { x, y } = clientToCanvasCoordinates(event);
  const { row, column } = canvasToSource(x, y);
  if (row < 0 || column < 0 || row >= state.imageRows || column >= state.imageColumns) {
    return; // click outside the actual image is rejected client-side too
  }
  state.pendingPoint = { row, column };
  setSaveState("unsaved");
  el("save-next").disabled = false;
  drawScene();
});

async function loadGuidance() {
  try {
    const response = await fetch("/static/guidance.json");
    state.guidance = await response.json();
  } catch (error) {
    state.guidance = null;
  }
}

async function loadTask(params = {}) {
  const query = new URLSearchParams();
  if (params.patient_index !== undefined) query.set("patient_index", params.patient_index);
  if (params.landmark !== undefined) query.set("landmark", params.landmark);
  const response = await apiFetch("/api/task?" + query.toString());
  if (response.status === 401) {
    window.location.reload();
    return;
  }
  if (response.status === 404) {
    showNoAssignments();
    return;
  }
  const task = await response.json();
  state.task = task;
  state.pendingPoint = null;
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  setSaveState(task.state === "pass" ? "saved" : "unsaved");
  el("save-next").disabled = task.state !== "pass";
  renderHeader();
  if (task.series) {
    await loadSlice(task.default_slice_index);
    const slider = el("slice-slider");
    slider.min = 0;
    slider.max = task.series.slice_count - 1;
    slider.value = task.default_slice_index;
    el("degraded-banner").classList.add("hidden");
  } else {
    el("degraded-banner").classList.toggle("hidden", task.reference_error !== "data_volume_unavailable");
  }
  await maybeShowCompletion();
}

async function loadSlice(index) {
  state.sliceIndex = index;
  const response = await apiFetch(`/api/slice?task_id=${encodeURIComponent(state.task.task_id)}&index=${index}`);
  if (!response.ok) {
    el("degraded-banner").classList.remove("hidden");
    return;
  }
  state.imageRows = parseInt(response.headers.get("X-Slice-Rows"), 10);
  state.imageColumns = parseInt(response.headers.get("X-Slice-Columns"), 10);
  state.sopUid = response.headers.get("X-Slice-Sop-Uid");
  const blob = await response.blob();
  state.imageBitmap = await createImageBitmap(blob);
  resizeCanvasToContainer();
  drawScene();
  el("slice-label").textContent = `Slice ${index + 1}/${response.headers.get("X-Slice-Count")}`;
}

function generateRequestId() {
  if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

async function saveCurrentPoint() {
  if (!state.pendingPoint || !state.task) return;
  const requestId = state.pendingRequestId || generateRequestId();
  state.pendingRequestId = requestId;
  setSaveState("saving");
  const body = {
    task_id: state.task.task_id,
    sop_uid: state.sopUid,
    row_px: state.pendingPoint.row,
    column_px: state.pendingPoint.column,
    expected_revision: state.task.revision,
    request_id: requestId,
  };
  let response;
  try {
    response = await apiFetch("/api/annotations/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (networkError) {
    setSaveState("offline");
    return;
  }
  if (response.status === 409) {
    const payload = await response.json();
    alert("This task changed since it was loaded. Reloading the latest saved point.");
    state.pendingRequestId = null;
    await loadTask({ patient_index: state.task.patient_index, landmark: state.task.landmark });
    return;
  }
  if (!response.ok) {
    let message = "The point was rejected and was not saved.";
    try {
      const payload = await response.json();
      if (payload.message) message = payload.message;
    } catch (error) {
      // Keep the safe generic explanation when an intermediary returned a
      // non-JSON error response.
    }
    setSaveState("rejected");
    alert(`Point not saved: ${message}`);
    return;
  }
  state.pendingRequestId = null;
  setSaveState("saved");
  await advanceAfterSave();
}

async function advanceAfterSave() {
  const task = state.task;
  if (task.landmark_index < task.landmark_count - 1) {
    await loadTask({ patient_index: task.patient_index, landmark: nextLandmarkName(task) });
  } else if (task.patient_index < task.patient_count - 1) {
    await loadTask({ patient_index: task.patient_index + 1 });
  } else {
    await maybeShowCompletion(true);
  }
}

function nextLandmarkName(task) {
  return task.landmark_states[task.landmark_index + 1].landmark;
}

async function submitUnable() {
  const reason = el("unable-reason").value;
  const notes = el("unable-notes").value;
  const requestId = generateRequestId();
  const response = await apiFetch("/api/annotations/unable", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      task_id: state.task.task_id,
      reason,
      notes: notes || null,
      expected_revision: state.task.revision,
      request_id: requestId,
    }),
  });
  el("unable-modal").classList.add("hidden");
  if (response.status === 409) {
    alert("This task changed since it was loaded. Reloading.");
  }
  await loadTask({ patient_index: state.task.patient_index, landmark: state.task.landmark });
}

async function maybeShowCompletion(force = false) {
  const response = await apiFetch("/api/progress");
  const progress = await response.json();
  if (progress.all_complete && (force || state.task === null)) {
    el("completion-summary").textContent =
      `${progress.completed_patient_count}/${progress.patient_count} patients saved. ` +
      `${progress.flagged_task_count} flagged tasks.`;
    const list = el("completion-review-list");
    list.innerHTML = "";
    progress.patients_needing_review.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = `Patient ${item.patient_index + 1}: ${item.flagged_landmarks.join(", ")}`;
      list.appendChild(li);
    });
    el("completion-screen").classList.remove("hidden");
  }
}

function showCompletion() {
  el("completion-screen").classList.remove("hidden");
}

function showNoAssignments() {
  el("main").innerHTML = "<p>No studies are assigned to this account yet.</p>";
}

function bindControls() {
  el("save-next").addEventListener("click", saveCurrentPoint);
  el("replace-point").addEventListener("click", () => {
    state.pendingPoint = null;
    setSaveState("unsaved");
    el("save-next").disabled = true;
    drawScene();
  });
  el("unable-button").addEventListener("click", () => el("unable-modal").classList.remove("hidden"));
  el("unable-cancel").addEventListener("click", () => el("unable-modal").classList.add("hidden"));
  el("unable-submit").addEventListener("click", submitUnable);

  el("previous-landmark").addEventListener("click", () => {
    const task = state.task;
    if (task.landmark_index > 0) {
      loadTask({ patient_index: task.patient_index, landmark: task.landmark_states[task.landmark_index - 1].landmark });
    }
  });
  el("next-landmark").addEventListener("click", () => {
    const task = state.task;
    if (task.landmark_index < task.landmark_count - 1) {
      loadTask({ patient_index: task.patient_index, landmark: nextLandmarkName(task) });
    }
  });
  el("previous-patient").addEventListener("click", () => {
    const task = state.task;
    if (task.patient_index > 0) loadTask({ patient_index: task.patient_index - 1 });
  });
  el("next-patient").addEventListener("click", () => {
    const task = state.task;
    if (task.patient_index < task.patient_count - 1) loadTask({ patient_index: task.patient_index + 1 });
  });

  el("slice-slider").addEventListener("input", (event) => {
    state.pendingPoint = null;
    setSaveState(state.task.state === "pass" ? "saved" : "unsaved");
    loadSlice(parseInt(event.target.value, 10));
  });
  el("slice-prev").addEventListener("click", () => {
    const slider = el("slice-slider");
    slider.value = Math.max(0, parseInt(slider.value, 10) - 1);
    slider.dispatchEvent(new Event("input"));
  });
  el("slice-next").addEventListener("click", () => {
    const slider = el("slice-slider");
    slider.value = Math.min(parseInt(slider.max, 10), parseInt(slider.value, 10) + 1);
    slider.dispatchEvent(new Event("input"));
  });

  el("zoom-in").addEventListener("click", () => { state.zoom = Math.min(6, state.zoom * 1.25); drawScene(); });
  el("zoom-out").addEventListener("click", () => { state.zoom = Math.max(1, state.zoom / 1.25); drawScene(); });
  el("reset-view").addEventListener("click", () => { state.zoom = 1; state.panX = 0; state.panY = 0; drawScene(); });

  el("details-toggle").addEventListener("click", () => {
    state.detailsOpen = !state.detailsOpen;
    el("details-panel").classList.toggle("hidden", !state.detailsOpen);
  });
  el("logout-button").addEventListener("click", async () => {
    await apiFetch("/logout", { method: "POST" });
    window.location.reload();
  });
  el("completion-signout").addEventListener("click", async () => {
    await apiFetch("/logout", { method: "POST" });
    window.location.reload();
  });
  el("completion-review-again").addEventListener("click", () => {
    el("completion-screen").classList.add("hidden");
  });

  document.addEventListener("keydown", (event) => {
    const tag = document.activeElement.tagName;
    if (tag === "TEXTAREA" || tag === "SELECT" || tag === "INPUT") return;
    if (event.key === "ArrowLeft") el("slice-prev").click();
    if (event.key === "ArrowRight") el("slice-next").click();
  });

  window.addEventListener("resize", () => { resizeCanvasToContainer(); drawScene(); });
  window.addEventListener("online", () => { if (state.saveState === "offline") saveCurrentPoint(); });
}

async function main() {
  bindControls();
  await loadGuidance();
  await loadTask();
}

main();
