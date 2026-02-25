/* eslint-env worker */
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js');

// ── Constants ─────────────────────────────────────────────────────────────────
const BUFFER_SIZE   = 128;
const INPUT_SIZE    = 72;
const PREDICT_EVERY = 30;
const HW            = INPUT_SIZE * INPUT_SIZE;
const FRAME_BYTES   = HW * 3;

// ── Frame buffer (managed entirely in worker) ────────────────────────────────
const frameBuffer = new Array(BUFFER_SIZE);
for (let i = 0; i < BUFFER_SIZE; i++) frameBuffer[i] = new Float32Array(FRAME_BYTES);
let bufWriteIdx        = 0;
let bufFilled          = 0;
let totalFrames        = 0;
let firstPredictionDone = false;

let session          = null;
let inferenceRunning = false;

// ── Message handler ──────────────────────────────────────────────────────────
self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === 'init') {
    try {
      ort.env.wasm.wasmPaths = msg.wasmPaths;
      session = await ort.InferenceSession.create(msg.modelUrl, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', message: 'ONNX init failed: ' + err.message });
    }
    return;
  }

  if (msg.type === 'frame') {
    // RGBA Uint8 → RGB Float32 into circular buffer
    const rgba  = msg.rgba;          // Uint8ClampedArray or Uint8Array
    const frame = frameBuffer[bufWriteIdx % BUFFER_SIZE];
    let j = 0;
    for (let i = 0; i < rgba.length; i += 4) {
      frame[j++] = rgba[i];
      frame[j++] = rgba[i + 1];
      frame[j++] = rgba[i + 2];
    }
    bufWriteIdx++;
    if (bufFilled < BUFFER_SIZE) bufFilled++;
    totalFrames++;

    // Report buffer progress
    self.postMessage({ type: 'buffer', filled: bufFilled, total: BUFFER_SIZE });

    // Decide whether to run inference
    if (bufFilled < BUFFER_SIZE || inferenceRunning || !session) return;

    let shouldInfer = false;
    if (!firstPredictionDone) {
      firstPredictionDone = true;
      shouldInfer = true;
    } else {
      shouldInfer = totalFrames % PREDICT_EVERY === 0;
    }
    if (!shouldInfer) return;

    // ── Run inference (non-blocking for main thread) ───────────────────────
    inferenceRunning = true;
    try {
      const inputData = toRawTensor();
      const tensor  = new ort.Tensor('float32', inputData, [1, 3, BUFFER_SIZE, INPUT_SIZE, INPUT_SIZE]);
      const results = await session.run({ video: tensor });

      const waveOut = results['rppg_wave'] || results[Object.keys(results)[0]];
      const spo2Out = results['spo2']      || results[Object.keys(results)[1]];

      if (!waveOut || !spo2Out) {
        self.postMessage({ type: 'error', message: 'Missing outputs: ' + Object.keys(results) });
        return;
      }

      const rppgWave = waveOut.data;
      const spo2Raw  = spo2Out.data[0];
      const hr       = estimateHR(rppgWave, 30);
      const spo2     = (isNaN(spo2Raw) || spo2Raw === 0) ? 0 : Math.max(85, Math.min(100, spo2Raw));

      self.postMessage({
        type:   'result',
        hr_bpm: Math.round(hr * 10) / 10,
        spo2:   Math.round(spo2 * 10) / 10,
      });
    } catch (err) {
      self.postMessage({ type: 'error', message: 'Inference error: ' + err.message });
    } finally {
      inferenceRunning = false;
    }
  }
};

// ── Raw Tensor (V2 model has built-in DiffNormalizeLayer) ────────────────────
// Input: raw RGB pixels 0-255, shape [1, 3, T, H, W] in CHW planar layout
function toRawTensor() {
  const T   = BUFFER_SIZE;
  const THW = T * HW;
  const output = new Float32Array(3 * THW);
  const startIdx = bufWriteIdx - BUFFER_SIZE;

  for (let t = 0; t < T; t++) {
    const frame = frameBuffer[(startIdx + t + BUFFER_SIZE) % BUFFER_SIZE];
    const baseR = t * HW;
    const baseG = THW + t * HW;
    const baseB = 2 * THW + t * HW;

    for (let px = 0; px < HW; px++) {
      const i = px * 3;
      output[baseR + px] = frame[i];        // R
      output[baseG + px] = frame[i + 1];    // G
      output[baseB + px] = frame[i + 2];    // B
    }
  }
  return output;
}

// ── FFT Heart Rate ──────────────────────────────────────────────────────────
function estimateHR(signal, fs) {
  const N = signal.length;
  const K = Math.floor(N / 2) + 1;
  let maxMag = 0, peakFreq = 0;

  for (let k = 0; k < K; k++) {
    const freq = k * fs / N;
    if (freq < 0.75 || freq > 2.5) continue;

    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const angle = -2 * Math.PI * k * n / N;
      re += signal[n] * Math.cos(angle);
      im += signal[n] * Math.sin(angle);
    }

    const mag = re * re + im * im;
    if (mag > maxMag) { maxMag = mag; peakFreq = freq; }
  }

  return peakFreq * 60;
}
