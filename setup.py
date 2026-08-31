from setuptools import setup, find_packages

setup(
    name="speechsight",
    version="1.0.0",
    description="SpeechSight AI - Visual intelligence for human speech",
    author="SpeechSight Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "pydantic>=2.6.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "opencv-python-headless>=4.8.0",
        "pillow>=10.0.0",
        "python-multipart>=0.0.9"
    ],
    extras_require={
        "train": [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "torchvision>=0.15.0"
        ],
        "test": [
            "pytest>=7.4.0",
            "httpx>=0.25.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "speechsight-server=speechsight.server.app:main",
            "speechsight-eval=speechsight.eval.evaluate:main"
        ]
    }
)
