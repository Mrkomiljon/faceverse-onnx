#!/usr/bin/env python3
"""
Download FaceVerse-ONNX model weights automatically
"""

import os
import requests
import zipfile
from tqdm import tqdm
import argparse

# GitHub release URL
RELEASE_URL = "https://github.com/Mrkomiljon/faceverse-onnx/releases/latest/download"

# Model files to download
MODEL_FILES = {
    "faceverse_resnet50.pth": "PyTorch model weights",
    "faceverse_v4_2.npy": "FaceVerse data file", 
    "face_landmarker.task": "MediaPipe face landmarker",
    "faceverse_resnet50_float32.onnx": "ONNX Float32 model",
    "faceverse_resnet50_int8.onnx": "ONNX INT8 quantized model"
}

def download_file(url, filename, description):
    """Download a file with progress bar"""
    print(f"📥 Downloading {description}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        print(f"✅ {description} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download FaceVerse-ONNX model weights")
    parser.add_argument("--output", default="data", help="Output directory (default: data)")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_FILES.keys()), 
                       help="Specific models to download (default: all)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine which models to download
    models_to_download = args.models if args.models else list(MODEL_FILES.keys())
    
    print("🚀 FaceVerse-ONNX Model Downloader")
    print("=" * 50)
    
    success_count = 0
    total_count = len(models_to_download)
    
    for model_file in models_to_download:
        if model_file in MODEL_FILES:
            url = f"{RELEASE_URL}/{model_file}"
            output_path = os.path.join(args.output, model_file)
            description = MODEL_FILES[model_file]
            
            if download_file(url, output_path, description):
                success_count += 1
        else:
            print(f"⚠️ Unknown model: {model_file}")
    
    print("=" * 50)
    print(f"📊 Download Summary: {success_count}/{total_count} models downloaded")
    
    if success_count == total_count:
        print("🎉 All models downloaded successfully!")
        print("\n📝 Next steps:")
        print("1. Run: python run_onnx.py --input webcam --save_video True")
        print("2. Or process a video: python run_onnx.py --input video.mp4 --save_video True")
    else:
        print("⚠️ Some models failed to download. Please check your internet connection.")

if __name__ == "__main__":
    main() 