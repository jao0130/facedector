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

    input_size = config.get('model', {}).get('input_size', 256)

    # Create and load model
    model = create_face_detector(config)
    model.build((None, input_size, input_size, 3))
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
    """Create a simple HTML demo for testing the TFJS model."""
    demo_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Face Detection Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.0.0"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #canvas {{ border: 1px solid #ccc; }}
        #video {{ display: none; }}
        .controls {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>Face Detection Demo</h1>
    <div class="controls">
        <button id="startBtn">Start Camera</button>
        <button id="stopBtn" disabled>Stop Camera</button>
    </div>
    <canvas id="canvas" width="640" height="480"></canvas>
    <video id="video" width="640" height="480" autoplay></video>
    <p id="fps">FPS: -</p>

    <script>
        const MODEL_INPUT_SIZE = {input_size};
        let model = null;
        let isRunning = false;
        let lastTime = 0;

        async function loadModel() {{
            console.log('Loading model...');
            model = await tf.loadGraphModel('model.json');
            console.log('Model loaded');
        }}

        async function detect(imageData) {{
            // Preprocess
            const tensor = tf.browser.fromPixels(imageData)
                .resizeBilinear([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE])
                .div(255.0)
                .expandDims(0);

            // Run inference
            const outputs = await model.predict(tensor);

            // Parse outputs
            const bbox = await outputs[0].data();
            const landmarks = await outputs[1].data();
            const confidence = await outputs[2].data();

            tensor.dispose();
            outputs.forEach(o => o.dispose());

            return {{ bbox, landmarks, confidence: confidence[0] }};
        }}

        function drawResults(ctx, result, width, height) {{
            const {{ bbox, landmarks, confidence }} = result;

            // Draw bbox
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.strokeRect(
                bbox[0] * width,
                bbox[1] * height,
                (bbox[2] - bbox[0]) * width,
                (bbox[3] - bbox[1]) * height
            );

            // Draw landmarks
            const colors = ['blue', 'blue', 'green', 'red', 'red'];
            for (let i = 0; i < 5; i++) {{
                ctx.fillStyle = colors[i];
                ctx.beginPath();
                ctx.arc(
                    landmarks[i * 2] * width,
                    landmarks[i * 2 + 1] * height,
                    4, 0, 2 * Math.PI
                );
                ctx.fill();
            }}

            // Draw confidence
            ctx.fillStyle = '#00ff00';
            ctx.font = '16px Arial';
            ctx.fillText(
                `Confidence: ${{confidence.toFixed(3)}}`,
                bbox[0] * width,
                bbox[1] * height - 10
            );
        }}

        async function processFrame() {{
            if (!isRunning) return;

            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            // Draw video frame
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Run detection
            const result = await detect(canvas);
            drawResults(ctx, result, canvas.width, canvas.height);

            // Calculate FPS
            const now = performance.now();
            const fps = 1000 / (now - lastTime);
            lastTime = now;
            document.getElementById('fps').textContent = `FPS: ${{fps.toFixed(1)}}`;

            requestAnimationFrame(processFrame);
        }}

        async function startCamera() {{
            const video = document.getElementById('video');
            const stream = await navigator.mediaDevices.getUserMedia({{
                video: {{ width: 640, height: 480 }}
            }});
            video.srcObject = stream;

            isRunning = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;

            lastTime = performance.now();
            processFrame();
        }}

        function stopCamera() {{
            isRunning = false;
            const video = document.getElementById('video');
            const stream = video.srcObject;
            if (stream) {{
                stream.getTracks().forEach(track => track.stop());
            }}
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }}

        // Initialize
        document.getElementById('startBtn').addEventListener('click', startCamera);
        document.getElementById('stopBtn').addEventListener('click', stopCamera);

        loadModel().then(() => {{
            console.log('Ready to start camera');
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
