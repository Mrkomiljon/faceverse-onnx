#!/usr/bin/env python3
"""
Check if FaceVerse-ONNX model files exist
"""

import os
import sys

REQUIRED_MODELS = [
    "data/faceverse_resnet50.pth",
    "data/faceverse_v4_2.npy", 
    "data/face_landmarker.task"
]

OPTIONAL_MODELS = [
    "faceverse_resnet50_float32.onnx",
    "faceverse_resnet50_int8.onnx"
]

def check_models():
    """Check if required model files exist"""
    print("🔍 Checking FaceVerse-ONNX model files...")
    print("=" * 50)
    
    missing_required = []
    missing_optional = []
    
    # Check required models
    for model_path in REQUIRED_MODELS:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {model_path} - MISSING")
            missing_required.append(model_path)
    
    # Check optional models
    for model_name in OPTIONAL_MODELS:
        model_path = os.path.join("data", model_name)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"⚠️ {model_path} - Optional (not found)")
            missing_optional.append(model_name)
    
    print("=" * 50)
    
    if missing_required:
        print("❌ Missing required model files!")
        print("\n📥 To download models, run:")
        print("python download_models.py")
        return False
    else:
        print("🎉 All required models found!")
        if missing_optional:
            print(f"\n💡 Optional models missing: {', '.join(missing_optional)}")
            print("To download optional models: python download_models.py --models " + " ".join(missing_optional))
        return True

if __name__ == "__main__":
    success = check_models()
    sys.exit(0 if success else 1) 