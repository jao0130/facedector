import { FaceLandmarker, FilesetResolver } from
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

// ── Constants ────────────────────────────────────────────────────────────────
const INPUT_SIZE      = 72;
const LARGE_BOX_COEF  = 1.5;
const FACE_THRESHOLD  = 0.4;
const MODEL_URL       = 'models/rppg_fcatt_v2.onnx';
const ORT_WASM        = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const MP_WASM         = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MP_MODEL        = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const LM_COLORS       = ['#4488ff', '#4488ff', '#00ff88', '#ff8844', '#ff8844'];

// ── DOM ──────────────────────────────────────────────────────────────────────
const video       = document.getElementById('webcam');
const overlay     = document.getElementById('overlay');
const ctx         = overlay.getContext('2d');
const connDot     = document.getElementById('conn-dot');
const connLabel   = document.getElementById('conn-label');
const fpsDisplay  = document.getElementById('fps-display');
const hrValue     = document.getElementById('hr-value');
const spo2Value   = document.getElementById('spo2-value');
const cardSpo2    = document.getElementById('card-spo2');
const bufferBar   = document.getElementById('buffer-bar');
const bufferPct   = document.getElementById('buffer-pct');
const noFace      = document.getElementById('no-face');
const confValue   = document.getElementById('conf-value');
const pipelineFps = document.getElementById('pipeline-fps');

// ── State ────────────────────────────────────────────────────────────────────
let faceLandmarker = null;
let worker         = null;
let workerReady    = false;
let latestVitals   = { hr_bpm: 0, spo2: 0 };
let bufFilled      = 0;
let bufTotal       = 128;
let fpsCount       = 0;
let fpsLast        = performance.now();
let lastProcessed  = 0;

// Hidden canvases for ROI extraction
const captureCanvas = document.createElement('canvas');
const captureCtx    = captureCanvas.getContext('2d', { willReadFrequently: true });
const roiCanvas     = document.createElement('canvas');
roiCanvas.width     = INPUT_SIZE;
roiCanvas.height    = INPUT_SIZE;
const roiCtx        = roiCanvas.getContext('2d', { willReadFrequently: true });

// ── Initialization ───────────────────────────────────────────────────────────
async function init() {
  setStatus('Loading models...', 'loading');
  console.log(`[VitalSense] crossOriginIsolated=${self.crossOriginIsolated}`);

  // Start MediaPipe and inference worker in parallel
  const [lm] = await Promise.all([initMediaPipe(), initWorker()]);
  faceLandmarker = lm;
  console.log('[VitalSense] Models loaded.');

  setStatus('Starting camera...', 'loading');
  await startCamera();
  console.log(`[VitalSense] Camera started: ${video.videoWidth}x${video.videoHeight}`);

  setStatus('Live', 'connected');
  requestAnimationFrame(mainLoop);
}

async function initMediaPipe() {
  const fs = await FilesetResolver.forVisionTasks(MP_WASM);
  return FaceLandmarker.createFromOptions(fs, {
    baseOptions: { modelAssetPath: MP_MODEL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: FACE_THRESHOLD,
    minFacePresenceConfidence: FACE_THRESHOLD,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}

function initWorker() {
  return new Promise((resolve, reject) => {
    worker = new Worker('inference-worker.js');

    worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === 'ready') {
        workerReady = true;
        console.log('[VitalSense] Inference worker ready.');
        resolve();
      } else if (msg.type === 'result') {
        latestVitals = { hr_bpm: msg.hr_bpm, spo2: msg.spo2 };
        console.log(`[VitalSense] HR=${msg.hr_bpm} BPM, SpO2=${msg.spo2}%`);
      } else if (msg.type === 'buffer') {
        bufFilled = msg.filled;
        bufTotal  = msg.total;
      } else if (msg.type === 'error') {
        console.error('[Worker]', msg.message);
        if (!workerReady) reject(new Error(msg.message));
      }
    };

    worker.onerror = (err) => {
      console.error('[Worker] Fatal:', err);
      if (!workerReady) reject(err);
    };

    // Send init message with model URL
    worker.postMessage({
      type:     'init',
      modelUrl: MODEL_URL,
      wasmPaths: ORT_WASM,
    });
  });
}

// ── Camera ───────────────────────────────────────────────────────────────────
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user', frameRate: { ideal: 30 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  resizeOverlay();
  window.addEventListener('resize', resizeOverlay);
}

function resizeOverlay() {
  const panel = overlay.parentElement;
  overlay.width  = panel.clientWidth;
  overlay.height = panel.clientHeight;
}

// ── Face Detection ───────────────────────────────────────────────────────────
function detectFace(timestamp) {
  if (!video.videoWidth) return null;

  const result = faceLandmarker.detectForVideo(video, timestamp);
  if (!result.faceLandmarks || result.faceLandmarks.length === 0) return null;

  const lms = result.faceLandmarks[0];

  // 5-point landmarks
  const fivePt = [
    lms.length > 468 ? [lms[468].x, lms[468].y] : [(lms[133].x + lms[33].x) / 2, (lms[133].y + lms[33].y) / 2],
    lms.length > 473 ? [lms[473].x, lms[473].y] : [(lms[362].x + lms[263].x) / 2, (lms[362].y + lms[263].y) / 2],
    [lms[1].x, lms[1].y],
    [lms[61].x, lms[61].y],
    [lms[291].x, lms[291].y],
  ];

  // Bounding box from landmarks
  let xMin = 1, xMax = 0, yMin = 1, yMax = 0;
  for (const l of lms) {
    if (l.x < xMin) xMin = l.x;
    if (l.x > xMax) xMax = l.x;
    if (l.y < yMin) yMin = l.y;
    if (l.y > yMax) yMax = l.y;
  }
  const pad = 0.2;
  const bw = xMax - xMin, bh = yMax - yMin;
  xMin = Math.max(0, xMin - bw * pad);
  yMin = Math.max(0, yMin - bh * pad);
  xMax = Math.min(1, xMax + bw * pad);
  yMax = Math.min(1, yMax + bh * pad);

  return { bbox: [xMin, yMin, xMax, yMax], landmarks: fivePt, confidence: 0.95 };
}

// ── Face ROI Crop ────────────────────────────────────────────────────────────
function cropFaceROI(bbox) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;

  if (captureCanvas.width !== vw || captureCanvas.height !== vh) {
    captureCanvas.width = vw;
    captureCanvas.height = vh;
  }

  captureCtx.drawImage(video, 0, 0, vw, vh);

  let [x1, y1, x2, y2] = bbox;
  x1 *= vw; y1 *= vh; x2 *= vw; y2 *= vh;

  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  const bw = (x2 - x1) * LARGE_BOX_COEF, bh = (y2 - y1) * LARGE_BOX_COEF;

  const rx1 = Math.max(0, Math.floor(cx - bw / 2));
  const ry1 = Math.max(0, Math.floor(cy - bh / 2));
  const rx2 = Math.min(vw, Math.floor(cx + bw / 2));
  const ry2 = Math.min(vh, Math.floor(cy + bh / 2));
  const rw = rx2 - rx1, rh = ry2 - ry1;
  if (rw <= 0 || rh <= 0) return null;

  roiCtx.drawImage(captureCanvas, rx1, ry1, rw, rh, 0, 0, INPUT_SIZE, INPUT_SIZE);
  return roiCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
}

// ── Canvas Overlay ───────────────────────────────────────────────────────────
function drawOverlay(face) {
  const cw = overlay.width, ch = overlay.height;
  ctx.clearRect(0, 0, cw, ch);
  if (!face) return;

  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(cw / vw, ch / vh);
  const dw = vw * scale, dh = vh * scale;
  const ox = (cw - dw) / 2, oy = (ch - dh) / 2;

  const [x1n, y1n, x2n, y2n] = face.bbox;
  const px = n => ox + n * dw;
  const py = n => oy + n * dh;

  const x1 = px(x1n), y1 = py(y1n);
  const bw = (x2n - x1n) * dw, bh = (y2n - y1n) * dh;

  // Corner brackets
  const len = Math.min(bw, bh) * 0.2;
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 2;
  ctx.lineCap = 'square';

  const corners = [
    [x1, y1, 1, 1],
    [x1 + bw, y1, -1, 1],
    [x1, y1 + bh, 1, -1],
    [x1 + bw, y1 + bh, -1, -1],
  ];

  ctx.beginPath();
  for (const [cx, cy, dx, dy] of corners) {
    ctx.moveTo(cx, cy + dy * len);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx + dx * len, cy);
  }
  ctx.stroke();

  // Scanline
  const centerY = y1 + bh / 2;
  ctx.strokeStyle = 'rgba(0, 255, 136, 0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x1, centerY);
  ctx.lineTo(x1 + bw, centerY);
  ctx.stroke();

  // Landmarks
  face.landmarks.forEach(([lxn, lyn], i) => {
    const lx = px(lxn), ly = py(lyn);
    ctx.beginPath();
    ctx.arc(lx, ly, 5, 0, Math.PI * 2);
    ctx.strokeStyle = LM_COLORS[i];
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(lx, ly, 2, 0, Math.PI * 2);
    ctx.fillStyle = LM_COLORS[i];
    ctx.fill();
  });

  // Confidence
  ctx.font = '10px monospace';
  ctx.fillStyle = 'rgba(0, 255, 136, 0.6)';
  ctx.fillText(`${Math.round(face.confidence * 100)}%`, x1, y1 - 6);
}

// ── UI Update ────────────────────────────────────────────────────────────────
function updateUI(face) {
  noFace.classList.toggle('hidden', !!face);

  confValue.textContent = face ? `${Math.round(face.confidence * 100)}%` : '--';

  // Buffer
  const fill = Math.round((bufFilled / bufTotal) * 100);
  bufferBar.style.width = `${fill}%`;
  bufferPct.textContent = `${fill}%`;
  if (fill < 50) bufferBar.style.background = 'var(--accent-amber)';
  else if (fill < 100) bufferBar.style.background = 'var(--accent-cyan)';
  else bufferBar.style.background = 'var(--accent-green)';

  // Vitals
  const hr = latestVitals.hr_bpm;
  hrValue.textContent = hr > 0 ? Math.round(hr) : '--';

  const spo2 = latestVitals.spo2;
  spo2Value.textContent = spo2 > 0 ? spo2.toFixed(1) : '--';
  if (spo2 > 0 && spo2 < 95) cardSpo2.classList.add('alert');
  else cardSpo2.classList.remove('alert');
}

// ── Status helper ────────────────────────────────────────────────────────────
function setStatus(label, state) {
  connLabel.textContent = label;
  connDot.className = 'dot ' + state;
}

// ── Main Loop ────────────────────────────────────────────────────────────────
function mainLoop(timestamp) {
  // ~30 FPS processing
  if (timestamp - lastProcessed < 30) {
    requestAnimationFrame(mainLoop);
    return;
  }
  lastProcessed = timestamp;

  const face = detectFace(timestamp);
  drawOverlay(face);

  if (face && workerReady) {
    const rgbaData = cropFaceROI(face.bbox);
    if (rgbaData) {
      // Send RGBA pixel data to worker (structured clone, ~21 KB)
      worker.postMessage({ type: 'frame', rgba: rgbaData });
    }
  }

  updateUI(face);
  fpsCount++;

  requestAnimationFrame(mainLoop);
}

// FPS counter
setInterval(() => {
  const now = performance.now();
  const dt = (now - fpsLast) / 1000;
  const fps = fpsCount / Math.max(dt, 0.001);
  fpsDisplay.textContent = `${Math.round(fps)} FPS`;
  pipelineFps.textContent = `${Math.round(fps)}`;
  fpsCount = 0;
  fpsLast = now;
}, 1000);

// ── Boot ─────────────────────────────────────────────────────────────────────
init().catch(err => {
  console.error('[VitalSense] Init failed:', err);
  setStatus('Error', 'disconnected');
});
