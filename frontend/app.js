(() => {
  const feedImg = document.getElementById("feed");
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  const videoWrap = document.getElementById("videoWrap");

  const statSource = document.getElementById("statSource");
  const statFps = document.getElementById("statFps");
  const statLatency = document.getElementById("statLatency");
  const statDevice = document.getElementById("statDevice");
  const statUptime = document.getElementById("statUptime");
  const statAlerts = document.getElementById("statAlerts");
  const weaponBanner = document.getElementById("weaponBanner");
  const weaponBannerText = document.getElementById("weaponBannerText");
  const objectsBar = document.getElementById("objectsBar");
  const objectsList = document.getElementById("objectsList");
  const videoEmpty = document.getElementById("videoEmpty");
  const liveBadge = document.getElementById("liveBadge");
  const WEAPON_CLASSES = new Set(["gun", "guns", "pistol", "rifle", "knife"]);

  const RULE_ICONS = {
    WEAPON: "⚠",
    RESTRICTED_ZONE_INTRUSION: "⛔",
    CROWD_SURGE: "👥",
    CROWD_THRESHOLD: "👥",
    UNATTENDED_OBJECT: "🎒",
    WRONG_DIRECTION: "↩",
    LOITERING: "⏱",
  };

  // Recommended response actions shown when a CRITICAL alert fires. This is
  // a display checklist, not an integration -- it demonstrates the response
  // workflow without depending on any real external system (radio, PA, door
  // locks) that this project doesn't actually control.
  const RESPONSE_ACTIONS = {
    WEAPON: ["Alert on-site security immediately", "Do not approach — maintain distance", "Evacuate the immediate area", "Notify law enforcement"],
    RESTRICTED_ZONE_INTRUSION: ["Dispatch security to the zone", "Verify occupant authorization", "Review recent zone access"],
    CROWD_SURGE: ["Deploy crowd control to the zone", "Open additional exits if available", "Monitor for further buildup"],
    CROWD_THRESHOLD: ["Monitor the zone for further buildup", "Prepare crowd control if it keeps rising"],
    UNATTENDED_OBJECT: ["Cordon off the immediate area", "Do not touch the object", "Notify security / bomb disposal per protocol"],
    WRONG_DIRECTION: ["Alert nearest staff member", "Check for blocked or malfunctioning exit"],
    LOITERING: ["Dispatch patrol to verify", "Review zone camera history"],
  };
  const actionPanel = document.getElementById("actionPanel");
  const actionPanelRule = document.getElementById("actionPanelRule");
  const actionPanelList = document.getElementById("actionPanelList");
  const actionPanelDismiss = document.getElementById("actionPanelDismiss");
  actionPanelDismiss.addEventListener("click", () => (actionPanel.hidden = true));

  function showActionPanel(alert) {
    const actions = RESPONSE_ACTIONS[alert.rule] || ["Review the alert and dispatch an appropriate response"];
    actionPanelRule.textContent = ruleLabel(alert.rule);
    actionPanelList.innerHTML = actions.map((a) => `<li>${a}</li>`).join("");
    actionPanel.hidden = false;
  }

  // Web Audio siren -- synthesized, no audio file to ship/host. Browsers
  // block audio before any user interaction with the page; the first
  // gesture (clicking Webcam/Connect/a sample clip) satisfies that.
  let audioCtx = null;
  function playAlarmSound() {
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const now = audioCtx.currentTime;
      for (let i = 0; i < 3; i++) {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = "square";
        const start = now + i * 0.5;
        osc.frequency.setValueAtTime(880, start);
        osc.frequency.linearRampToValueAtTime(660, start + 0.25);
        gain.gain.setValueAtTime(0.0001, start);
        gain.gain.exponentialRampToValueAtTime(0.25, start + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.45);
        osc.connect(gain).connect(audioCtx.destination);
        osc.start(start);
        osc.stop(start + 0.46);
      }
    } catch (e) {
      // Web Audio unavailable/blocked -- the visual action panel still shows.
    }
  }

  function formatUptime(seconds) {
    seconds = Math.max(0, Math.floor(seconds || 0));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
    if (m > 0) return `${m}m ${String(s).padStart(2, "0")}s`;
    return `${s}s`;
  }

  const btnWebcam = document.getElementById("btnWebcam");
  const cameraSourceInput = document.getElementById("cameraSourceInput");
  const btnConnectCamera = document.getElementById("btnConnectCamera");
  const uploadInput = document.getElementById("uploadInput");
  const sampleSelect = document.getElementById("sampleSelect");

  const btnRecordScreen = document.getElementById("btnRecordScreen");
  const btnRecordScreenLabel = document.getElementById("btnRecordScreenLabel");
  const recDot = document.getElementById("recDot");
  let mediaRecorder = null;
  let recordedChunks = [];

  async function startScreenRecording() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 }, audio: false });
    } catch (e) {
      return; // user cancelled the share picker, or the browser blocked it
    }
    recordedChunks = [];
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm";
    mediaRecorder = new MediaRecorder(stream, { mimeType });
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      a.href = url;
      a.download = `vigil-recording-${stamp}.webm`;
      a.click();
      URL.revokeObjectURL(url);
      setRecordingUI(false);
    };
    // If the user stops sharing via the browser's own "Stop sharing" control
    // (not our button), treat that the same as clicking Stop here.
    stream.getVideoTracks()[0].addEventListener("ended", () => {
      if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    });
    mediaRecorder.start();
    setRecordingUI(true);
  }

  function stopScreenRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    }
  }

  function setRecordingUI(isRecording) {
    recDot.hidden = !isRecording;
    btnRecordScreenLabel.textContent = isRecording ? "Stop Recording" : "Record Screen";
    btnRecordScreen.classList.toggle("recording", isRecording);
  }

  btnRecordScreen.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      stopScreenRecording();
    } else {
      startScreenRecording();
    }
  });

  const btnDrawZone = document.getElementById("btnDrawZone");
  const btnFinishZone = document.getElementById("btnFinishZone");
  const btnCancelZone = document.getElementById("btnCancelZone");
  const zoneListEl = document.getElementById("zoneList");

  const alertListEl = document.getElementById("alertList");
  const alertCountEl = document.getElementById("alertCount");

  const modalBackdrop = document.getElementById("modalBackdrop");
  const modalBody = document.getElementById("modalBody");
  const modalClose = document.getElementById("modalClose");

  const zoneModalBackdrop = document.getElementById("zoneModalBackdrop");
  const zoneModalClose = document.getElementById("zoneModalClose");
  const zoneName = document.getElementById("zoneName");
  const zoneRestricted = document.getElementById("zoneRestricted");
  const zoneCrowdEnabled = document.getElementById("zoneCrowdEnabled");
  const zoneCrowdThreshold = document.getElementById("zoneCrowdThreshold");
  const zoneLoiterEnabled = document.getElementById("zoneLoiterEnabled");
  const zoneLoiterSeconds = document.getElementById("zoneLoiterSeconds");
  const zoneSaveBtn = document.getElementById("zoneSaveBtn");

  let frameSize = { w: 1280, h: 720 };
  let zones = [];
  let drawing = false;
  let currentPoints = [];
  let pendingPolygon = null;
  let alertsById = {};
  let seenAlertIds = null; // null = first fetch not done yet (don't alarm on pre-existing history)

  const ZONE_LINE_COLOR = "#5b8cff";
  const ZONE_RESTRICTED_COLOR = "#ff4d5e";
  const DRAFT_COLOR = "#ffd24d";

  // ---------- coordinate mapping (video-pixel space <-> canvas-pixel space) ----------
  function resizeCanvas() {
    canvas.width = videoWrap.clientWidth;
    canvas.height = videoWrap.clientHeight;
    redraw();
  }
  function videoToCanvas([x, y]) {
    return [x * (canvas.width / frameSize.w), y * (canvas.height / frameSize.h)];
  }
  function canvasToVideo(x, y) {
    return [x * (frameSize.w / canvas.width), y * (frameSize.h / canvas.height)];
  }

  function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const zone of zones) {
      drawPolygon(zone.polygon.map(videoToCanvas), zone.restricted ? ZONE_RESTRICTED_COLOR : ZONE_LINE_COLOR, zone.name);
    }
    if (drawing && currentPoints.length > 0) {
      drawPolygon(currentPoints.map(videoToCanvas), DRAFT_COLOR, null, false);
    }
  }

  function drawPolygon(points, color, label, closed = true) {
    if (points.length === 0) return;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
    if (closed && points.length > 2) ctx.closePath();
    ctx.stroke();
    for (const [x, y] of points) {
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    if (label) {
      ctx.font = "12px sans-serif";
      ctx.fillText(label, points[0][0] + 4, points[0][1] - 6);
    }
  }

  window.addEventListener("resize", resizeCanvas);
  feedImg.addEventListener("load", resizeCanvas);

  canvas.addEventListener("click", (e) => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    currentPoints.push(canvasToVideo(x, y));
    redraw();
  });

  // ---------- zone drawing controls ----------
  btnDrawZone.addEventListener("click", () => {
    drawing = true;
    currentPoints = [];
    btnDrawZone.hidden = true;
    btnFinishZone.hidden = false;
    btnCancelZone.hidden = false;
    redraw();
  });

  btnCancelZone.addEventListener("click", () => {
    drawing = false;
    currentPoints = [];
    btnDrawZone.hidden = false;
    btnFinishZone.hidden = true;
    btnCancelZone.hidden = true;
    redraw();
  });

  btnFinishZone.addEventListener("click", () => {
    if (currentPoints.length < 3) {
      alert("Click at least 3 points to define a zone polygon.");
      return;
    }
    pendingPolygon = currentPoints;
    drawing = false;
    btnDrawZone.hidden = false;
    btnFinishZone.hidden = true;
    btnCancelZone.hidden = true;
    openZoneModal();
  });

  function openZoneModal() {
    zoneName.value = "";
    zoneRestricted.checked = false;
    zoneCrowdEnabled.checked = false;
    zoneCrowdThreshold.value = 5;
    zoneLoiterEnabled.checked = false;
    zoneLoiterSeconds.value = 8;
    zoneModalBackdrop.hidden = false;
  }
  zoneModalClose.addEventListener("click", () => {
    zoneModalBackdrop.hidden = true;
    pendingPolygon = null;
    currentPoints = [];
    redraw();
  });

  zoneSaveBtn.addEventListener("click", async () => {
    const name = zoneName.value.trim() || `Zone ${zones.length + 1}`;
    const newZone = {
      id: `zone_${Date.now()}`,
      name,
      polygon: pendingPolygon,
      restricted: zoneRestricted.checked,
      crowd_threshold: zoneCrowdEnabled.checked ? parseInt(zoneCrowdThreshold.value, 10) : null,
      loiter_seconds: zoneLoiterEnabled.checked ? parseFloat(zoneLoiterSeconds.value) : null,
      allowed_direction: null,
    };
    const updated = [...zones, newZone];
    await fetch("/api/zones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updated),
    });
    zoneModalBackdrop.hidden = true;
    pendingPolygon = null;
    currentPoints = [];
    await fetchZones();
  });

  async function fetchZones() {
    const res = await fetch("/api/zones");
    zones = await res.json();
    renderZoneChips();
    redraw();
  }

  function renderZoneChips() {
    zoneListEl.innerHTML = "";
    for (const zone of zones) {
      const chip = document.createElement("div");
      chip.className = "zone-chip";
      const dot = document.createElement("span");
      dot.className = "dot";
      dot.style.background = zone.restricted ? ZONE_RESTRICTED_COLOR : ZONE_LINE_COLOR;
      const label = document.createElement("span");
      const tags = [];
      if (zone.restricted) tags.push("restricted");
      if (zone.crowd_threshold) tags.push(`crowd≥${zone.crowd_threshold}`);
      if (zone.loiter_seconds) tags.push(`loiter>${zone.loiter_seconds}s`);
      label.textContent = `${zone.name}${tags.length ? " · " + tags.join(", ") : ""}`;
      const del = document.createElement("button");
      del.textContent = "×";
      del.title = "Remove zone";
      del.addEventListener("click", async () => {
        const updated = zones.filter((z) => z.id !== zone.id);
        await fetch("/api/zones", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updated),
        });
        await fetchZones();
      });
      chip.append(dot, label, del);
      zoneListEl.appendChild(chip);
    }
  }

  // ---------- crowd surge settings (GET/POST /api/thresholds) ----------
  const surgeModalBackdrop = document.getElementById("surgeModalBackdrop");
  const SURGE_FIELDS = {
    surge_min_increase: document.getElementById("surgeMinIncrease"),
    surge_window_s: document.getElementById("surgeWindowS"),
    surge_avg_multiplier: document.getElementById("surgeAvgMultiplier"),
    surge_min_people: document.getElementById("surgeMinPeople"),
  };
  document.getElementById("btnSurgeSettings").addEventListener("click", async () => {
    const res = await fetch("/api/thresholds");
    const t = await res.json();
    for (const [key, input] of Object.entries(SURGE_FIELDS)) input.value = t[key];
    surgeModalBackdrop.hidden = false;
  });
  document.getElementById("surgeModalClose").addEventListener("click", () => (surgeModalBackdrop.hidden = true));
  document.getElementById("surgeSaveBtn").addEventListener("click", async () => {
    const body = {};
    for (const [key, input] of Object.entries(SURGE_FIELDS)) {
      const v = parseFloat(input.value);
      if (!Number.isNaN(v)) body[key] = v;
    }
    const res = await fetch("/api/thresholds", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      alert("Invalid surge settings: " + (await res.text()));
      return;
    }
    surgeModalBackdrop.hidden = true;
  });

  // ---------- status polling ----------
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      const status = await res.json();
      const isActive = !!(status.running && status.frame_width && status.frame_height);
      videoEmpty.hidden = isActive;
      liveBadge.hidden = !isActive;
      videoWrap.classList.toggle("live", isActive);

      statSource.textContent = status.source === "none" ? "No source" : status.source;
      statFps.textContent = isActive ? status.fps : "-";
      const capture = status.capture_ms ?? 0;
      const inference = status.inference_ms ?? 0;
      statLatency.textContent = isActive ? `${Math.round(capture + inference)}ms` : "-";
      statLatency.title = `capture ${capture.toFixed(0)}ms + inference ${inference.toFixed(0)}ms`;
      statDevice.textContent = (status.device || "cpu").toUpperCase();
      statUptime.textContent = formatUptime(status.uptime_seconds);
      statAlerts.textContent = status.total_alerts ?? 0;

      if (status.weapon_detector_enabled) {
        weaponBanner.hidden = true;
      } else {
        weaponBanner.hidden = false;
        weaponBannerText.textContent =
          "Weapon detection is offline (no model loaded) - behavior-based detection (zones, crowd, loitering, direction) is fully active.";
      }

      if (status.frame_width && status.frame_height) {
        frameSize = { w: status.frame_width, h: status.frame_height };
      }
      renderObjects(status.objects || {});
    } catch (e) {
      statSource.textContent = "unavailable";
    }
  }

  function renderObjects(objects) {
    const entries = Object.entries(objects);
    if (entries.length === 0) {
      objectsBar.hidden = true;
      return;
    }
    objectsBar.hidden = false;
    objectsList.innerHTML = entries
      .map(([cls, count]) => {
        const isWeapon = WEAPON_CLASSES.has(cls.toLowerCase());
        return `<span class="object-chip${isWeapon ? " weapon-chip" : ""}">${cls} × ${count}</span>`;
      })
      .join(" ");
  }

  // ---------- alerts polling ----------
  function ruleLabel(rule) {
    return rule.replace(/_/g, " ");
  }
  function timeAgo(ts) {
    const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ago`;
  }

  async function fetchAlerts() {
    try {
      const res = await fetch("/api/alerts?limit=50");
      const alerts = await res.json();
      alertCountEl.textContent = alerts.length;

      if (seenAlertIds === null) {
        // First load: these are pre-existing alerts, not new events -- record
        // them silently so the alarm doesn't fire retroactively on page load.
        seenAlertIds = new Set(alerts.map((a) => a.id));
      } else {
        const newCritical = alerts.filter((a) => a.band === "CRITICAL" && !seenAlertIds.has(a.id));
        for (const a of alerts) seenAlertIds.add(a.id);
        if (newCritical.length > 0) {
          playAlarmSound();
          showActionPanel(newCritical[0]);
        }
      }

      alertsById = {};
      if (alerts.length === 0) {
        alertListEl.innerHTML = '<div class="empty-state">No alerts yet. Anomalies will appear here, ranked by severity.</div>';
        return;
      }
      alertListEl.innerHTML = "";
      for (const alert of alerts) {
        alertsById[alert.id] = alert;
        const card = document.createElement("div");
        card.className = `alert-card ${alert.band}`;
        const icon = RULE_ICONS[alert.rule] || "●";
        card.innerHTML = `
          <div class="alert-icon">${icon}</div>
          <div class="alert-body">
            <div class="alert-top">
              <span class="alert-rule">${ruleLabel(alert.rule)}</span>
              <span class="alert-score ${alert.band}">${alert.band} · ${alert.score}</span>
            </div>
            <div class="alert-message">${alert.message}</div>
            <div class="alert-meta">${alert.zone_name ? alert.zone_name + " · " : ""}${timeAgo(alert.timestamp)}</div>
          </div>
        `;
        card.addEventListener("click", () => showAlertModal(alert));
        alertListEl.appendChild(card);
      }
    } catch (e) {
      // keep last known list on transient errors
    }
  }

  function showAlertModal(alert) {
    const imgHtml = alert.has_evidence
      ? `<img src="/api/alerts/${alert.id}/evidence" alt="evidence frame" />`
      : "";
    modalBody.innerHTML = `
      ${imgHtml}
      <h3>${ruleLabel(alert.rule)}</h3>
      <div class="detail-row"><span>Message</span><span>${alert.message}</span></div>
      <div class="detail-row"><span>Severity</span><span>${alert.band} (${alert.score}/100)</span></div>
      <div class="detail-row"><span>Zone</span><span>${alert.zone_name || "-"}</span></div>
      <div class="detail-row"><span>Track IDs</span><span>${alert.track_ids.join(", ") || "-"}</span></div>
      <div class="detail-row"><span>Time</span><span>${new Date(alert.timestamp * 1000).toLocaleTimeString()}</span></div>
      <div class="modal-actions">
        <button class="btn accent" id="modalReportBtn">Generate Incident Report</button>
      </div>
    `;
    modalBackdrop.hidden = false;
    document.getElementById("modalReportBtn").addEventListener("click", (e) => {
      e.target.textContent = "Generating...";
      generateIncidentReport(alert).finally(() => (e.target.textContent = "Generate Incident Report"));
    });
  }
  modalClose.addEventListener("click", () => (modalBackdrop.hidden = true));
  modalBackdrop.addEventListener("click", (e) => {
    if (e.target === modalBackdrop) modalBackdrop.hidden = true;
  });

  // ---------- incident report ----------
  const RESPONSE_ACTIONS_FOR_REPORT = RESPONSE_ACTIONS; // same map used for the live action panel

  async function generateIncidentReport(alert) {
    const generatedAt = new Date().toLocaleString();
    const actions = RESPONSE_ACTIONS_FOR_REPORT[alert.rule] || ["Review the alert and dispatch an appropriate response"];

    let evidenceImgTag = "";
    if (alert.has_evidence) {
      try {
        const res = await fetch(`/api/alerts/${alert.id}/evidence`);
        const blob = await res.blob();
        const dataUrl = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
        evidenceImgTag = `<img src="${dataUrl}" alt="evidence" />`;
      } catch (e) {
        // evidence fetch failed -- report still generates without the image
      }
    }

    const html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8" /><title>Incident Report - Alert ${alert.id}</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Inter, Arial, sans-serif; background:#f4f5f8; color:#1a1d29; margin:0; padding:40px; }
  .sheet { max-width: 720px; margin: 0 auto; background:#fff; border-radius:12px; box-shadow:0 2px 20px rgba(0,0,0,0.08); overflow:hidden; }
  .head { background:#0a0c12; color:#fff; padding:28px 36px; }
  .head h1 { margin:0; font-size:20px; letter-spacing:2px; }
  .head p { margin:4px 0 0; color:#8b90a6; font-size:12px; }
  .band { display:inline-block; margin-top:14px; padding:4px 14px; border-radius:999px; font-weight:700; font-size:12px; letter-spacing:0.5px; }
  .band.CRITICAL { background:#ff4d5e; color:#fff; }
  .band.HIGH { background:#ff9a4d; color:#1a1d29; }
  .band.MEDIUM { background:#ffd24d; color:#1a1d29; }
  .band.LOW { background:#6b7690; color:#fff; }
  .body { padding: 28px 36px; }
  table { width:100%; border-collapse: collapse; margin-bottom: 8px; }
  td { padding:8px 0; border-bottom:1px solid #e8e9ee; font-size:14px; vertical-align:top; }
  td.label { color:#6b7690; width:160px; font-weight:600; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:0.6px; color:#6b7690; margin:28px 0 10px; }
  ul { margin:0; padding-left:20px; }
  li { margin-bottom:6px; font-size:14px; }
  img { width:100%; border-radius:8px; border:1px solid #e8e9ee; display:block; }
  .footer { padding:16px 36px; border-top:1px solid #e8e9ee; font-size:11px; color:#a2a6b8; }
  @media print { body { background:#fff; padding:0; } .sheet { box-shadow:none; border-radius:0; } }
</style></head>
<body>
  <div class="sheet">
    <div class="head">
      <h1>VIGIL - INCIDENT REPORT</h1>
      <p>Threat Detection &amp; Situational Awareness</p>
      <span class="band ${alert.band}">${alert.band} - ${alert.score}/100</span>
    </div>
    <div class="body">
      <table>
        <tr><td class="label">Incident ID</td><td>#${alert.id}</td></tr>
        <tr><td class="label">Type</td><td>${ruleLabel(alert.rule)}</td></tr>
        <tr><td class="label">Description</td><td>${alert.message}</td></tr>
        <tr><td class="label">Zone</td><td>${alert.zone_name || "N/A"}</td></tr>
        <tr><td class="label">Detected at</td><td>${new Date(alert.timestamp * 1000).toLocaleString()}</td></tr>
        <tr><td class="label">Report generated</td><td>${generatedAt}</td></tr>
      </table>
      ${evidenceImgTag ? `<h2>Evidence</h2>${evidenceImgTag}` : ""}
      <h2>Recommended Response</h2>
      <ul>${actions.map((a) => `<li>${a}</li>`).join("")}</ul>
    </div>
    <div class="footer">Auto-generated by Vigil at time of detection. For internal incident triage -- does not replace direct verification by on-site personnel.</div>
  </div>
</body></html>`;

    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    window.open(url, "_blank");
  }

  // ---------- source controls ----------
  btnWebcam.addEventListener("click", async () => {
    await fetch("/api/source/webcam", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index: 0 }),
    });
  });

  async function connectCamera() {
    const source = cameraSourceInput.value.trim();
    if (!source) return;
    await fetch("/api/source/camera", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
  }
  btnConnectCamera.addEventListener("click", connectCamera);
  cameraSourceInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") connectCamera();
  });

  uploadInput.addEventListener("change", async () => {
    const file = uploadInput.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    await fetch("/api/source/upload", { method: "POST", body: formData });
    uploadInput.value = "";
  });

  async function fetchSamples() {
    try {
      const res = await fetch("/api/samples");
      const names = await res.json();
      for (const name of names) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sampleSelect.appendChild(opt);
      }
    } catch (e) {
      // no samples available; leave placeholder only
    }
  }
  sampleSelect.addEventListener("change", async () => {
    if (!sampleSelect.value) return;
    await fetch("/api/source/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: sampleSelect.value }),
    });
  });

  // ---------- init ----------
  resizeCanvas();
  fetchStatus();
  fetchZones();
  fetchSamples();
  fetchAlerts();
  setInterval(fetchStatus, 3000);
  setInterval(fetchAlerts, 1500);
})();
