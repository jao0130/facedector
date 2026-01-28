"""
Export face detection model to TensorFlow Lite format for mobile deployment.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import yaml

from models.face_detector import create_face_detector


def export_to_tflite(
    model_path: str,
    config_path: str,
    output_path: str,
    quantize: bool = True,
    representative_data: Optional[np.ndarray] = None,
):
    """
    Export model to TensorFlow Lite format.

    Args:
        model_path: Path to model weights
        config_path: Path to config file
        output_path: Path to save TFLite model
        quantize: Whether to apply INT8 quantization
        representative_data: Representative dataset for quantization calibration
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

    # Create concrete function for conversion
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, input_size, input_size, 3], dtype=tf.float32)
    ])
    def inference_func(x):
        outputs = model(x, training=False)
        # Flatten outputs for TFLite compatibility
        return {
            'bbox': outputs['bbox'],
            'landmarks': tf.reshape(outputs['landmarks'], [1, -1]),
            'confidence': outputs['confidence'],
        }

    # Get concrete function
    concrete_func = inference_func.get_concrete_function()

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    if quantize:
        # Enable quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_data is not None:
            # Full INT8 quantization with representative dataset
            def representative_dataset():
                for i in range(min(100, len(representative_data))):
                    yield [representative_data[i:i+1].astype(np.float32)]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.float32
            print("Using full INT8 quantization")
        else:
            # Dynamic range quantization (no calibration data needed)
            print("Using dynamic range quantization")
    else:
        print("No quantization applied (FP32)")

    # Convert
    tflite_model = converter.convert()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # Print model size
    model_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {model_size_mb:.2f} MB")

    return output_path


def verify_tflite_model(
    tflite_path: str,
    input_size: int = 256,
):
    """
    Verify TFLite model by running inference.

    Args:
        tflite_path: Path to TFLite model
        input_size: Model input size
    """
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\nModel verification:")
    print(f"  Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: {detail['name']} {detail['shape']} {detail['dtype']}")

    # Run test inference
    test_input = np.random.rand(1, input_size, input_size, 3).astype(np.float32)

    # Adjust input type if quantized
    if input_details[0]['dtype'] == np.uint8:
        test_input = (test_input * 255).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    # Get outputs
    outputs = {}
    for detail in output_details:
        output = interpreter.get_tensor(detail['index'])
        outputs[detail['name']] = output
        print(f"  {detail['name']}: {output.shape}, range [{output.min():.4f}, {output.max():.4f}]")

    print("Model verification passed!")

    return outputs


def main():
    parser = argparse.ArgumentParser(description='Export face detection model to TFLite')
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
        default='export/face_detector.tflite',
        help='Output TFLite model path',
    )
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Disable quantization',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify exported model',
    )

    args = parser.parse_args()

    # Export
    output_path = export_to_tflite(
        args.model,
        args.config,
        args.output,
        quantize=not args.no_quantize,
    )

    # Verify
    if args.verify:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        input_size = config.get('model', {}).get('input_size', 256)
        verify_tflite_model(str(output_path), input_size)


if __name__ == '__main__':
    main()
