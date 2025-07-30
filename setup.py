from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="faceverse-onnx",
    version="4.1.0",
    authors="Mrkomiljon, Lizhen Wang, Zhiyuan Chen, Tao Yu, Chenguang Ma, Liang Li, Yebin Liu",
    author_email="komiljon19950813@gmail.com",
    description="ONNX-optimized implementation of FaceVerse V4 for fast 3D face reconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mrkomiljon/faceverse-onnx",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "mediapipe>=0.8.0",
        "onnxruntime-gpu>=1.12.0",
        "scipy>=1.7.0",
        "pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    keywords="faceverse, onnx, 3d-face, face-reconstruction, computer-vision, deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/Mrkomiljon/faceverse-onnx/issues",
        "Source": "https://github.com/Mrkomiljon/faceverse-onnx",
        "Documentation": "https://github.com/Mrkomiljon/faceverse-onnx#readme",
    },
) 