# VIGIL backend API

The contract is whatever `frontend/index.html` calls. The backend matches it; if the two ever
disagree, the frontend wins and this file gets updated.

- Base URL: `http://localhost:8000`
- All request and response bodies are JSON unless noted.
- Errors: `400` for bad input, `404` for missing resources. The body is always `{"detail": "readable message"}`, including body validation errors (FastAPI's default 422 is converted to 400).
- The server starts idle. Nothing plays until a source is chosen.

| Method | Path | Used by dashboard |
|---|---|---|
| GET | `/` | page itself |
| GET | `/video_feed` | video panel + zone canvas background |
| GET | `/api/status` | metric cards, polled every 3 s |
| POST | `/api/source/camera` | Network Camera / DroidCam connect |
| POST | `/api/source/sample` | Sample Video connect |
| GET | `/api/source/samples` | not yet (lists valid sample names) |
| POST | `/api/source/upload` | Upload button |
| POST | `/api/source/stop` | not yet |
| GET | `/api/alerts?limit=50` | alert sidebar, polled every 3 s |
| DELETE | `/api/alerts` | Clear Alerts button |
| GET | `/api/alerts/{id}/evidence` | evidence modal image |
| GET / POST | `/api/zones` | zone list / save / clear |
| GET / POST | `/api/thresholds` | thresholds panel |
| POST | `/api/audit` | audit log mirror (optional) |

---

## GET /video_feed

MJPEG stream (`multipart/x-mixed-replace; boundary=frame`) of the annotated video: person/object
boxes with track IDs, zones (restricted red, others amber, with names), raw weapon boxes, and a
`CRITICAL THREAT: <CLASS>` banner only while a weapon is confirmed (5 of the last 8 inference
frames at conf >= 0.5). Shows a placeholder frame when no source is running.

```html
<img src="/video_feed?t=1727251200000">
```

## GET /api/status

```json
{
  "running": true,
  "source": "sample: vtest.avi",
  "object_count": 6,
  "fps": 9.8,
  "latency_ms": 51.1,
  "device": "cuda:0",
  "alert_count": 26
}
```

- `source` is readable: `"Webcam 0"`, `"camera: http://192.168.1.20:4747/video"` (any `user:password@` is shown as `***@`), `"sample: vtest.avi"`, `"upload: clip.mp4"`, or `null` when idle.
- `fps` is the pipeline rate. File sources play at their native frame rate (vtest.avi is 10 fps).
- `latency_ms` is the processing time of the last frame (detection, tracking, rules, weapon model, drawing, encoding).
- `device` is `"cuda:0"` or `"cpu"`.
- `object_count` is the number of tracked objects in the current frame (all COCO classes).

Idle:

```json
{"running": false, "source": null, "object_count": 0, "fps": 0.0, "latency_ms": 0.0, "device": "cuda:0", "alert_count": 0}
```

## POST /api/source/camera

All digits means a local webcam index; otherwise it must be an `http(s)://`, `rtsp(s)://` or
`rtmp://` URL (DroidCam WiFi: `http://PHONE-IP:4747/video`). URL sources time out after 5 s if
unreachable.

```json
{"source": "0"}
```
```json
{"source": "http://192.168.1.20:4747/video"}
```

Response: same JSON as `/api/status`.

```json
{"running": true, "source": "Webcam 0", "object_count": 0, "fps": 0.0, "latency_ms": 0.0, "device": "cuda:0", "alert_count": 0}
```

Errors (400):

```json
{"detail": "source must be a webcam index like 0, or a stream URL like http://PHONE-IP:4747/video or rtsp://..."}
```
```json
{"detail": "could not open source camera: http://192.168.1.20:4747/video"}
```

## POST /api/source/sample

Plays `test_videos/<name>` on loop. `name` must be a plain file name from `/api/source/samples`;
paths (`../`, `\`, drive letters) are rejected.

```json
{"name": "vtest.avi"}
```

Response: same JSON as `/api/status` (`"source": "sample: vtest.avi"`).

Errors (400):

```json
{"detail": "name must be a plain file name from /api/source/samples"}
```
```json
{"detail": "sample 'nope.mp4' not found; available: confusers.mp4, vtest.avi"}
```

## GET /api/source/samples

```json
["confusers.mp4", "vtest.avi"]
```

## POST /api/source/upload

`multipart/form-data` with field `file`. Accepted: `.mp4 .avi .mov .mkv .webm .m4v`, up to 1 GB.
Saved to `uploads/` (gitignored) and played on loop.

```bash
curl -F "file=@clip.mp4" http://localhost:8000/api/source/upload
```

Response: same JSON as `/api/status` (`"source": "upload: clip.mp4"`).

Errors (400): `{"detail": "unsupported file type '.txt'; use .avi, .m4v, .mkv, .mov, .mp4, .webm"}`

## POST /api/source/stop

No body. Response: `/api/status` JSON with `"running": false, "source": null`.

## GET /api/alerts?limit=50

`limit` 1-200 (default 50). Sorted by severity (critical, high, medium, low), then newest first.
At most 200 alerts are kept. Zone rules (intrusion, loitering, crowd) are rate-limited per
(rule, zone): at most one alert per zone every 12 s however many people walk in. Weapon alerts
have their own per-class 12 s cooldown and are never suppressed by other alert types.

```json
[
  {
    "id": 26,
    "type": "Restricted Zone Intrusion",
    "rule": "restricted_intrusion",
    "severity": "high",
    "description": "3 people in restricted zone 'Walkway' (latest #137)",
    "message": "3 people in restricted zone 'Walkway' (latest #137)",
    "timestamp": "14:10:03",
    "created_at": "2026-09-25T14:10:03+05:30",
    "zone": "Walkway",
    "track_id": 137,
    "has_evidence": true
  }
]
```

| rule | type | severity | fires when |
|---|---|---|---|
| `weapon` | Weapon Detected | critical | pistol/knife confirmed in 5 of the last 8 inference frames |
| `restricted_intrusion` | Restricted Zone Intrusion | high | a person's bottom-center point enters a restricted zone |
| `crowd_surge` | Crowd Surge | medium | people in a zone >= `crowd_threshold` (whole frame if no zones) |
| `wrong_direction` | Wrong Direction | medium | reserved, not implemented |
| `loitering` | Loitering | low | a person stays in a zone >= `loiter_seconds` |

- `timestamp` is local `HH:MM:SS` (the dashboard shows it as-is); `created_at` is ISO 8601.
- `zone` is the zone name or `null`; `track_id` is the latest person to trigger it, `null` for weapon and crowd alerts.

## DELETE /api/alerts

Clears every alert (and resets cooldowns). Response: `{"ok": true, "alert_count": 0}`.

## GET /api/alerts/{id}/evidence

The annotated frame (`image/jpeg`) captured when the alert fired. `404 {"detail": "no evidence for this alert"}`
for unknown ids, including the dashboard's simulated `demo-...` ids.

## GET /api/zones

```json
[
  {
    "id": "zone-1790325564476",
    "name": "Walkway",
    "polygon": [[0.2994, 0.3487], [0.8489, 0.3487], [0.8489, 0.7480], [0.2994, 0.7480]],
    "restricted": true
  }
]
```

## POST /api/zones

Takes the FULL list and replaces everything; `[]` clears all zones. Response: the saved list.
Persisted to `data/zones.json` (gitignored).

- `polygon`: at least 3 `[x, y]` points, normalized 0-1 of the frame width/height.
- `id` and `name` are strings (generated if missing); `restricted` defaults to `false`.

```json
[{"id": "zone-1", "name": "Door", "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5]], "restricted": true}]
```

Errors (400): `{"detail": "zone zone-1: points must be normalized to 0-1"}`,
`{"detail": "zone zone-1: polygon needs at least 3 points"}`

Note: the dashboard draws zones over the video with `object-fit: cover` in a 16:9 box. For
non-16:9 sources (vtest.avi is 4:3) the canvas crops the frame, so drawn zones sit slightly
off from where the backend places them. 16:9 sources line up exactly.

## GET /api/thresholds

```json
{"loiter_seconds": 30.0, "crowd_threshold": 10, "unattended_seconds": 20.0, "wrong_direction_angle_deg": 45.0}
```

## POST /api/thresholds

Full or partial update, applied live to the running pipeline. Response: all thresholds.

```json
{"loiter_seconds": 12, "crowd_threshold": 4, "unattended_seconds": 25, "wrong_direction_angle_deg": 90}
```

| field | type | range |
|---|---|---|
| `loiter_seconds` | number | 1-3600 |
| `crowd_threshold` | whole number | 1-1000 |
| `unattended_seconds` | number | 1-3600 (stored; rule not implemented) |
| `wrong_direction_angle_deg` | number | 0-180 (stored; rule not implemented) |

Errors (400): `{"detail": "crowd_threshold must be between 1 and 1000"}`,
`{"detail": "unknown threshold(s): bogus"}`. A rejected update changes nothing.

## POST /api/audit

Accepts any JSON, logs one line on the server, returns:

```json
{"ok": true}
```
