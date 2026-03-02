from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ocrmypdf-paddleocr",
    use_scm_version=True,
    author="Your Name",
    author_email="your.email@example.com",
    description="PaddleOCR plugin for OCRmyPDF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ocrmypdf-paddleocr",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics :: Capture :: Scanners",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "ocrmypdf>=14.0.0",
        "paddlepaddle>=2.5.0",
        "paddlex==3.3.*",
        "paddleocr==3.3.*",
        "pillow>=9.0.0",
    ],
    extras_require={
        # PaddleOCR-VL-1.5 (0.9B VLM) support - requires additional doc-parser deps
        "vl": [
            "paddleocr[doc-parser]>=3.3.0",
        ],
    },
    entry_points={
        "ocrmypdf": [
            "paddleocr = ocrmypdf_paddleocr.plugin:get_plugin",
        ],
    },
)
