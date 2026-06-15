from setuptools import setup

setup(
    name="liveness-detector",
    version="1.0.0",
    description="Webcam liveness detector using passive texture analysis and active challenges",
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "liveness-detector=main:main",
        ],
    },
)
