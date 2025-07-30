import torch
import torch.onnx
import numpy as np
import time
import onnxruntime as ort
from faceversev4 import FaceVerseRecon
import argparse

def load_faceverse(device):
    """
    run.py dan olingan load_faceverse funksiyasi
    """
    fvr = FaceVerseRecon(
        "data/faceverse_v4_2.npy",
        "data/faceverse_resnet50.pth",
        device
    )
    return fvr

def convert_to_float32_onnx():
    """
    Create Float32 ONNX model (main)
    """
    print("=== Creating Float32 ONNX ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model (from run.py)
    fvr = load_faceverse(device)
    fvr.reconnet.eval()
    
    # Test input (like in run.py)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    print("Exporting to ONNX...")
    torch.onnx.export(
        fvr.reconnet,
        dummy_input,
        "data/faceverse_resnet50_float32.onnx",
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False,
        keep_initializers_as_inputs=False
    )
    
    print("✅ Float32 ONNX saved: data/faceverse_resnet50_float32.onnx")

def convert_to_quantized_onnx():
    """
    Create Quantized (INT8) ONNX model
    """
    print("\n=== Creating Quantized ONNX ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    fvr = load_faceverse(device)
    fvr.reconnet.eval()
    
    # Dynamic quantization (INT8)
    print("Performing quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        fvr.reconnet, 
        {torch.nn.Linear, torch.nn.Conv2d}, 
        dtype=torch.qint8
    )
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    print("Exporting to ONNX...")
    torch.onnx.export(
        quantized_model,
        dummy_input,
        "data/faceverse_resnet50_int8.onnx",
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False,
        keep_initializers_as_inputs=False
    )
    
    print("✅ Quantized ONNX saved: data/faceverse_resnet50_int8.onnx")

def benchmark_models():
    """
    PyTorch vs ONNX comparison
    """
    print("\n=== Speed Comparison ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # PyTorch model
    pytorch_model = load_faceverse(device)
    pytorch_model.reconnet.eval()
    
    # Test input
    test_input = torch.randn(1, 3, 256, 256).to(device)
    test_input_np = test_input.cpu().numpy()
    
    # PyTorch benchmark
    with torch.no_grad():
        # Warmup
        for _ in range(20):
            _ = pytorch_model.reconnet(test_input)
        
        # Benchmark
        torch_times = []
        for _ in range(100):
            start_time = time.time()
            _ = pytorch_model.reconnet(test_input)
            torch_times.append(time.time() - start_time)
        
        torch_avg = np.mean(torch_times)
    
    # ONNX models
    onnx_models = [
        ("faceverse_resnet50_float32.onnx", "Float32 ONNX"),
        ("faceverse_resnet50_int8.onnx", "Quantized ONNX")
    ]
    
    results = []
    
    for model_path, model_name in onnx_models:
        try:
            # ONNX session
            ort_session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            print(f"\n{model_name}:")
            print(f"  Provider: {ort_session.get_providers()}")
            
            # Warmup
            for _ in range(20):
                _ = ort_session.run(['output'], {'input': test_input_np})
            
            # Benchmark
            onnx_times = []
            for _ in range(100):
                start_time = time.time()
                _ = ort_session.run(['output'], {'input': test_input_np})
                onnx_times.append(time.time() - start_time)
            
            onnx_avg = np.mean(onnx_times)
            speedup = torch_avg / onnx_avg
            
            results.append((model_name, onnx_avg, speedup))
            print(f"  Time: {onnx_avg*1000:.2f}ms")
            print(f"  FPS: {1/onnx_avg:.1f}")
            print(f"  Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"{model_name}: Error - {e}")
    
    # Results
    print(f"\n=== Final Results ===")
    print(f"PyTorch: {torch_avg*1000:.2f}ms ({1/torch_avg:.1f} FPS) - baseline")
    
    for model_name, onnx_time, speedup in results:
        print(f"{model_name}: {onnx_time*1000:.2f}ms ({1/onnx_time:.1f} FPS) - {speedup:.2f}x")
    
    # Find best result
    if results:
        best_model = max(results, key=lambda x: x[2])
        print(f"\n🏆 Best: {best_model[0]} ({best_model[2]:.2f}x speedup)")

def test_accuracy():
    """
    Accuracy comparison
    """
    print("\n=== Accuracy Comparison ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # PyTorch model
    pytorch_model = load_faceverse(device)
    pytorch_model.reconnet.eval()
    
    # Test input
    test_input = torch.randn(1, 3, 256, 256).to(device)
    test_input_np = test_input.cpu().numpy()
    
    # PyTorch result
    with torch.no_grad():
        torch_output = pytorch_model.reconnet(test_input)
    
    # ONNX results
    onnx_models = [
        ("faceverse_resnet50_float32.onnx", "Float32 ONNX"),
        ("faceverse_resnet50_int8.onnx", "Quantized ONNX")
    ]
    
    for model_path, model_name in onnx_models:
        try:
            ort_session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            onnx_output = ort_session.run(['output'], {'input': test_input_np})[0]
            
            # Calculate difference
            diff = np.abs(torch_output.cpu().numpy() - onnx_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"\n{model_name}:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            
            if max_diff < 0.01:
                print("  ✅ Accuracy well preserved")
            elif max_diff < 0.1:
                print("  ⚠️ Accuracy slightly reduced")
            else:
                print("  ❌ Accuracy significantly reduced")
                
        except Exception as e:
            print(f"{model_name}: Error - {e}")

def show_quantization_info():
    """
    Information about quantization
    """
    print("\n=== Quantization Information ===")
    print("Quantization - used to reduce model size and speed up inference")
    print("\nAdvantages:")
    print("✅ Model size reduced by 75%")
    print("✅ Memory usage reduced by 75%")
    print("✅ Speed increased by 2-4x")
    print("✅ Saves battery (on mobile devices)")
    
    print("\nDisadvantages:")
    print("❌ Accuracy reduced by 1-2%")
    print("❌ Slower on some GPUs")
    
    print("\nWhen to use:")
    print("✅ Real-time video processing")
    print("✅ Mobile devices")
    print("✅ Edge computing")
    print("✅ Limited memory")
    
    print("\nWhen not to use:")
    print("❌ When high accuracy is required")
    print("❌ When powerful GPU is available")
    print("❌ Offline processing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceVerse PyTorch to ONNX Converter")
    parser.add_argument("--float32", action="store_true", help="Create Float32 ONNX")
    parser.add_argument("--quantized", action="store_true", help="Create Quantized ONNX")
    parser.add_argument("--benchmark", action="store_true", help="Speed comparison")
    parser.add_argument("--accuracy", action="store_true", help="Accuracy comparison")
    parser.add_argument("--all", action="store_true", help="Run everything")
    parser.add_argument("--info", action="store_true", help="Quantization information")
    
    args = parser.parse_args()
    
    print("🎯 FaceVerse PyTorch → ONNX Converter")
    print("=" * 50)
    
    if args.info:
        show_quantization_info()
    
    elif args.all or (not args.float32 and not args.quantized and not args.benchmark and not args.accuracy):
        # Run everything
        convert_to_float32_onnx()
        convert_to_quantized_onnx()
        benchmark_models()
        test_accuracy()
        show_quantization_info()
    
    else:
        # Run selected operations
        if args.float32:
            convert_to_float32_onnx()
        
        if args.quantized:
            convert_to_quantized_onnx()
        
        if args.benchmark:
            benchmark_models()
        
        if args.accuracy:
            test_accuracy()
    
    print("\n🎉 Conversion completed!")
    print("\n💡 Usage:")
    print("python convert_faceverse_to_onnx.py --all")
    print("python convert_faceverse_to_onnx.py --float32 --benchmark")
    print("python convert_faceverse_to_onnx.py --info") 