{ pkgs ? import <nixpkgs-unstable> {} }:

let
  # Use Python 3.12 (paddlepaddle 3.0.0 in unstable supports 3.12)
  python = pkgs.python312;
  pythonPackages = python.pkgs;

  # Override paddlex to include OCR optional dependencies
  paddlex-with-ocr = pythonPackages.paddlex.overridePythonAttrs (old: {
    dependencies = (old.dependencies or []) ++ (old.optional-dependencies.ocr or []);
  });

  # Override paddleocr to use our paddlex-with-ocr
  paddleocr-with-ocr = pythonPackages.paddleocr.override {
    paddlex = paddlex-with-ocr;
  };

  # Build the ocrmypdf-paddleocr plugin
  ocrmypdf-paddleocr = pythonPackages.buildPythonPackage rec {
    pname = "ocrmypdf-paddleocr";
    version = "0.1.0";
    format = "pyproject";

    src = ./.;

    nativeBuildInputs = with pythonPackages; [
      setuptools
      setuptools-scm
      wheel
    ];

    propagatedBuildInputs = with pythonPackages; [
      (ocrmypdf.override {
        # Skip tests for img2pdf which is failing in unstable
        img2pdf = img2pdf.overridePythonAttrs (old: { doCheck = false; });
      })
      paddlepaddle
      paddleocr-with-ocr  # PaddleOCR with PaddleX[ocr]
      pillow
    ];

    # PaddlePaddle tests require GPU or specific CPU instructions
    doCheck = false;

    pythonImportsCheck = [
      "ocrmypdf_paddleocr"
    ];
  };

in
pkgs.mkShell {
  buildInputs = [
    # Python with the plugin and all dependencies
    (python.withPackages (ps: [
      ocrmypdf-paddleocr
      # Explicitly include the overridden packages with OCR dependencies
      paddlex-with-ocr
      paddleocr-with-ocr
      # Missing PaddleX OCR dependencies not in nixpkgs paddlex optional-dependencies
      ps.python-bidi
      ps.sentencepiece
      # Development tools
      ps.pytest
      ps.ipython
    ]))

    # Additional tools that might be useful
    pkgs.ghostscript
    pkgs.pngquant
    pkgs.unpaper
  ];

  shellHook = ''
    # Workaround: PaddleX checks for opencv-contrib-python and premailer but we have opencv4
    # and don't need premailer (only used for HTML table to Excel conversion, not OCR)
    # Create dummy modules and metadata so the import checks pass
    export PYTHONPATH="$PWD/.python-shims:$PYTHONPATH"
    mkdir -p .python-shims
    cat > .python-shims/opencv_contrib_python.py << 'EOF'
# Dummy module to satisfy PaddleX's opencv-contrib-python check
# We use opencv4 from nixpkgs instead
from cv2 import *
EOF
    cat > .python-shims/premailer.py << 'EOF'
# Dummy module to satisfy PaddleX's premailer check
# Premailer is only used for HTML table to Excel conversion, not for OCR
class Premailer:
    pass
EOF
    # Create fake package metadata for importlib.metadata
    mkdir -p .python-shims/opencv_contrib_python-4.10.0.84.dist-info
    cat > .python-shims/opencv_contrib_python-4.10.0.84.dist-info/METADATA << 'EOF'
Metadata-Version: 2.1
Name: opencv-contrib-python
Version: 4.10.0.84
EOF
    mkdir -p .python-shims/premailer-3.10.0.dist-info
    cat > .python-shims/premailer-3.10.0.dist-info/METADATA << 'EOF'
Metadata-Version: 2.1
Name: premailer
Version: 3.10.0
EOF

    echo "OCRmyPDF-PaddleOCR development environment"
    echo "========================================="
    echo ""
    echo "Python: $(python --version)"
    echo "OCRmyPDF: $(python -c 'import ocrmypdf; print(ocrmypdf.__version__)' 2>/dev/null || echo 'not found')"
    echo "PaddleOCR: $(python -c 'import paddleocr; print(paddleocr.__version__)' 2>/dev/null || echo 'not found')"
    echo ""
    echo "Usage:"
    echo "  ocrmypdf --plugin ocrmypdf_paddleocr input.pdf output.pdf"
    echo ""
    echo "Plugin installed at:"
    echo "  ${ocrmypdf-paddleocr}/lib/python3.12/site-packages/ocrmypdf_paddleocr"
    echo ""
  '';
}
