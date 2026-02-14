"""
Export face detection model to TensorFlow.js format for web deployment.
"""

import argparse
import shutil
from pathlib import Path

import tensorflow as tf
import yaml

from models.face_detector import create_face_detector


def export_to_tfjs(
    model_path: str,
    config_path: str,
    output_dir: str,
    quantize: bool = True,
):
    """
    Export model to TensorFlow.js format.

    Args:
        model_path: Path to model weights
        config_path: Path to config file
        output_dir: Directory to save TFJS model
        quantize: Whether to apply quantization
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    input_size = config.get('model', {}).get('input_size', 224)

    # Create and load model (pretrained=False since we load trained weights)
    model = create_face_detector(config, pretrained=False)
    # Forward pass to build all layers including FPN
    dummy_input = tf.zeros((1, input_size, input_size, 3))
    model(dummy_input, training=False)
    model.load_weights(model_path)

    print(f"Model loaded from {model_path}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as SavedModel first
    saved_model_dir = output_path / 'saved_model_temp'

    # Create a simplified model for TFJS
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    outputs = model(inputs)

    # Create export model with named outputs
    export_model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            outputs['bbox'],
            outputs['landmarks'],
            outputs['confidence'],
        ],
    )

    # Rename outputs for clarity
    export_model.output_names = ['bbox', 'landmarks', 'confidence']

    # Save as SavedModel
    export_model.save(str(saved_model_dir))
    print(f"SavedModel exported to {saved_model_dir}")

    # Convert to TFJS using tensorflowjs_converter
    # Note: This requires tensorflowjs package to be installed
    try:
        import tensorflowjs as tfjs

        quantization_dtype = 'uint16' if quantize else None

        tfjs.converters.convert_tf_saved_model(
            str(saved_model_dir),
            str(output_path),
            quantization_dtype_map={quantization_dtype: '*'} if quantize else None,
        )

        print(f"TFJS model exported to {output_path}")

        # Cleanup temp SavedModel
        shutil.rmtree(saved_model_dir)

    except ImportError:
        print("\nTensorFlow.js converter not installed.")
        print("Please install with: pip install tensorflowjs")
        print(f"\nThen run: tensorflowjs_converter --input_format=tf_saved_model "
              f"--output_format=tfjs_graph_model {saved_model_dir} {output_path}")

        # Keep SavedModel for manual conversion
        print(f"\nSavedModel kept at: {saved_model_dir}")

    # Generate HTML demo file
    create_demo_html(output_path, input_size)


def create_demo_html(output_dir: Path, input_size: int):
    """Create a production-ready HTML demo for testing the TFJS model."""
    demo_html = f'''<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Detection Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: system-ui, sans-serif; background: #111; color: #eee; }}

        .container {{
            max-width: 720px;
            margin: 0 auto;
            padding: 16px;
        }}

        h1 {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 12px;
        }}

        .video-wrap {{
            position: relative;
            background: #222;
            border-radius: 8px;
            overflow: hidden;
            aspect-ratio: 4 / 3;
        }}

        #canvas {{
            display: block;
            width: 100%;
            height: 100%;
        }}

        #video {{
            display: none;
        }}

        .overlay {{
            position: absolute;
            top: 12px;
            left: 12px;
            font-size: 13px;
            font-family: monospace;
            background: rgba(0, 0, 0, 0.6);
            padding: 4px 8px;
            border-radius: 4px;
        }}

        #status {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 15px;
            text-align: center;
            color: #aaa;
        }}

        .controls {{
            display: flex;
            gap: 12px;
            margin-top: 12px;
            align-items: center;
        }}

        button {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            min-height: 44px;
            transition: opacity 0.2s;
        }}

        button:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}

        #startBtn {{
            background: #2563eb;
            color: white;
        }}

        #stopBtn {{
            background: #dc2626;
            color: white;
        }}

        .threshold-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-left: auto;
            font-size: 13px;
        }}

        #thresholdRange {{
            width: 100px;
        }}

        .stats {{
            display: flex;
            gap: 16px;
            margin-top: 12px;
            font-size: 13px;
            font-family: monospace;
            color: #aaa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Detection Demo</h1>

        <div class="video-wrap">
            <canvas id="canvas" width="640" height="480"></canvas>
            <video id="video" playsinline autoplay muted></video>
            <div id="status">Loading model...</div>
        </div>

        <div class="controls">
            <button id="startBtn" disabled>Start Camera</button>
            <button id="stopBtn" disabled>Stop</button>
            <div class="threshold-group">
                <label for="thresholdRange">Threshold</label>
                <input type="range" id="thresholdRange" min="0" max="100" value="50">
                <span id="thresholdVal">0.50</span>
            </div>
        </div>

        <div class="stats">
            <span id="fpsInfo">FPS: -</span>
            <span id="inferInfo">Inference: -</span>
            <span id="confInfo">Confidence: -</span>
            <span id="backendInfo">Backend: -</span>
        </div>
    </div>

    <script>
        const MODEL_INPUT_SIZE = {input_size};
        const LANDMARK_COLORS = ['#3b82f6', '#3b82f6', '#22c55e', '#ef4444', '#ef4444'];
        const LANDMARK_NAMES = ['L-Eye', 'R-Eye', 'Nose', 'L-Mouth', 'R-Mouth'];

        let model = null;
        let isRunning = false;
        let confidenceThreshold = 0.5;

        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        // --- Model ---
        async function loadModel() {{
            await tf.setBackend('webgl');
            await tf.ready();
            document.getElementById('backendInfo').textContent = 'Backend: ' + tf.getBackend();

            model = await tf.loadGraphModel('model.json');

            // Warmup: first inference is slow due to shader compilation
            const warmup = tf.zeros([1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3]);
            await model.executeAsync(warmup);
            warmup.dispose();

            document.getElementById('status').textContent = 'Model ready — click Start Camera';
            document.getElementById('startBtn').disabled = false;
        }}

        // --- Detection ---
        async function detect(source) {{
            const t0 = performance.now();

            let bbox, landmarks, confidence;

            const inputTensor = tf.tidy(() => {{
                return tf.browser.fromPixels(source)
                    .resizeBilinear([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE])
                    .div(255.0)
                    .expandDims(0);
            }});

            const outputs = await model.executeAsync(inputTensor);
            inputTensor.dispose();

            // outputs is an array: [bbox, landmarks, confidence]
            const [bboxT, landmarksT, confT] = Array.isArray(outputs) ? outputs : [outputs];

            bbox = await bboxT.data();
            landmarks = await landmarksT.data();
            confidence = (await confT.data())[0];

            bboxT.dispose();
            landmarksT.dispose();
            confT.dispose();

            const inferMs = performance.now() - t0;
            return {{ bbox, landmarks, confidence, inferMs }};
        }}

        // --- Drawing ---
        function drawResults(result, w, h) {{
            const {{ bbox, landmarks, confidence, inferMs }} = result;

            if (confidence >= confidenceThreshold) {{
                const x1 = bbox[0] * w, y1 = bbox[1] * h;
                const x2 = bbox[2] * w, y2 = bbox[3] * h;

                // Bbox
                ctx.strokeStyle = '#22c55e';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Confidence label
                ctx.fillStyle = '#22c55e';
                ctx.font = 'bold 14px monospace';
                ctx.fillText(confidence.toFixed(3), x1, y1 - 6);

                // Landmarks
                for (let i = 0; i < 5; i++) {{
                    const lx = landmarks[i * 2] * w;
                    const ly = landmarks[i * 2 + 1] * h;
                    ctx.fillStyle = LANDMARK_COLORS[i];
                    ctx.beginPath();
                    ctx.arc(lx, ly, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }}
            }}

            // Update stats
            const fps = 1000 / inferMs;
            document.getElementById('fpsInfo').textContent = 'FPS: ' + fps.toFixed(1);
            document.getElementById('inferInfo').textContent = 'Inference: ' + inferMs.toFixed(1) + 'ms';
            document.getElementById('confInfo').textContent = 'Confidence: ' + confidence.toFixed(3);
        }}

        // --- Main Loop ---
        async function processFrame() {{
            if (!isRunning) return;

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            try {{
                const result = await detect(canvas);
                drawResults(result, canvas.width, canvas.height);
            }} catch (err) {{
                console.error('Detection error:', err);
            }}

            requestAnimationFrame(processFrame);
        }}

        // --- Camera ---
        async function startCamera() {{
            try {{
                const stream = await navigator.mediaDevices.getUserMedia({{
                    video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 480 }}, facingMode: 'user' }}
                }});
                video.srcObject = stream;
                await video.play();

                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;

                isRunning = true;
                document.getElementById('status').style.display = 'none';
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                processFrame();
            }} catch (err) {{
                document.getElementById('status').textContent = 'Camera error: ' + err.message;
                console.error(err);
            }}
        }}

        function stopCamera() {{
            isRunning = false;
            const stream = video.srcObject;
            if (stream) stream.getTracks().forEach(t => t.stop());
            video.srcObject = null;

            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('status').style.display = 'block';
            document.getElementById('status').textContent = 'Camera stopped';
        }}

        // --- Events ---
        document.getElementById('startBtn').addEventListener('click', startCamera);
        document.getElementById('stopBtn').addEventListener('click', stopCamera);
        document.getElementById('thresholdRange').addEventListener('input', (e) => {{
            confidenceThreshold = e.target.value / 100;
            document.getElementById('thresholdVal').textContent = confidenceThreshold.toFixed(2);
        }});

        // --- Init ---
        loadModel().catch(err => {{
            document.getElementById('status').textContent = 'Failed to load model: ' + err.message;
            console.error(err);
        }});
    </script>
</body>
</html>
'''

    demo_path = output_dir / 'demo.html'
    with open(demo_path, 'w') as f:
        f.write(demo_html)

    print(f"Demo HTML created at {demo_path}")


def main():
    parser = argparse.ArgumentParser(description='Export face detection model to TFJS')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model weights',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='export/tfjs_model',
        help='Output directory for TFJS model',
    )
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Disable quantization',
    )

    args = parser.parse_args()

    export_to_tfjs(
        args.model,
        args.config,
        args.output,
        quantize=not args.no_quantize,
    )


if __name__ == '__main__':
    main()
