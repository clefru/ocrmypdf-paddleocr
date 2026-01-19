# OCRmyPDF-PaddleOCR

A PaddleOCR plugin for OCRmyPDF, enabling the use of PaddleOCR as an alternative OCR engine to Tesseract.

## Features

- Drop-in replacement for Tesseract OCR in OCRmyPDF
- Support for multiple languages including Chinese, Japanese, Korean, and many others
- GPU acceleration support
- Text orientation detection
- Configurable text detection and recognition models
- **Optimized bounding boxes** for accurate text selection in PDF output

## Installation

### NixOS

```nix
# In your NixOS configuration
{ pkgs }:

let
  ocrmypdf-paddleocr = pkgs.callPackage ./path/to/ocrmypdf-paddleocr/default.nix {};
in
{
  environment.systemPackages = [
    ocrmypdf-paddleocr
  ];
}
```

### Using pip

```bash
# Install from source
pip install .

# Or in development mode
pip install -e .
```

### Dependencies

- Python >= 3.8
- OCRmyPDF >= 14.0.0
- PaddlePaddle >= 2.5.0
- PaddleOCR >= 2.7.0
- Pillow >= 9.0.0

## Usage

### Command Line

Use PaddleOCR as the OCR engine with the `--plugin` flag:

```bash
ocrmypdf --plugin ocrmypdf_paddleocr input.pdf output.pdf
```

### With Language Selection

```bash
# English
ocrmypdf --plugin ocrmypdf_paddleocr -l eng input.pdf output.pdf

# Chinese Simplified
ocrmypdf --plugin ocrmypdf_paddleocr -l chi_sim input.pdf output.pdf

# Multiple languages (uses first language)
ocrmypdf --plugin ocrmypdf_paddleocr -l eng+fra input.pdf output.pdf
```

### With GPU Acceleration

```bash
ocrmypdf --plugin ocrmypdf_paddleocr --paddle-use-gpu input.pdf output.pdf
```

### Python API

```python
import ocrmypdf

ocrmypdf.ocr(
    'input.pdf',
    'output.pdf',
    plugins=['ocrmypdf_paddleocr'],
    language='eng'
)

# With GPU
ocrmypdf.ocr(
    'input.pdf',
    'output.pdf',
    plugins=['ocrmypdf_paddleocr'],
    language='chi_sim',
    paddle_use_gpu=True
)
```

## Command Line Options

The plugin adds the following PaddleOCR-specific options:

- `--paddle-use-gpu`: Use GPU acceleration (requires GPU-enabled PaddlePaddle)
- `--paddle-no-angle-cls`: Disable text orientation classification
- `--paddle-show-log`: Show PaddleOCR internal logging
- `--paddle-det-model-dir DIR`: Path to custom text detection model directory
- `--paddle-rec-model-dir DIR`: Path to custom text recognition model directory
- `--paddle-cls-model-dir DIR`: Path to custom text orientation classification model directory

## Supported Languages

PaddleOCR supports many languages. The plugin maps common Tesseract language codes to PaddleOCR codes:

| Tesseract Code | PaddleOCR Code | Language |
|---------------|----------------|----------|
| eng | en | English |
| chi_sim | ch | Chinese Simplified |
| chi_tra | chinese_cht | Chinese Traditional |
| fra | fr | French |
| deu | german | German |
| spa | spanish | Spanish |
| rus | ru | Russian |
| jpn | japan | Japanese |
| kor | korean | Korean |
| ara | ar | Arabic |
| hin | hi | Hindi |
| por | pt | Portuguese |
| ita | it | Italian |
| tur | tr | Turkish |
| vie | vi | Vietnamese |
| tha | th | Thai |

And many more! See PaddleOCR documentation for the complete list.

## Examples

### Basic OCR

```bash
ocrmypdf --plugin ocrmypdf_paddleocr input.pdf output.pdf
```

### Force OCR on all pages

```bash
ocrmypdf --plugin ocrmypdf_paddleocr --force-ocr input.pdf output.pdf
```

### Skip pages that already have text

```bash
ocrmypdf --plugin ocrmypdf_paddleocr --skip-text input.pdf output.pdf
```

### Optimize output file size

```bash
ocrmypdf --plugin ocrmypdf_paddleocr --optimize 3 input.pdf output.pdf
```

### Chinese document with GPU

```bash
ocrmypdf --plugin ocrmypdf_paddleocr -l chi_sim --paddle-use-gpu input.pdf output.pdf
```

## Development

### Running Tests

```bash
pytest tests/
```

### Building from Source

```bash
# Install in development mode
pip install -e .

# Build distribution
python -m build
```

## How It Works

The plugin implements the OCRmyPDF `OcrEngine` interface, which requires:

1. **Language support**: Maps OCRmyPDF/Tesseract language codes to PaddleOCR codes
2. **Text detection**: Uses PaddleOCR to detect text regions in images
3. **Text recognition**: Recognizes text within detected regions
4. **hOCR generation**: Converts PaddleOCR output to hOCR format for OCRmyPDF to overlay on PDFs

PaddleOCR processes each page image and returns bounding boxes with recognized text and confidence scores. The plugin converts this to hOCR (HTML-based OCR) format, which OCRmyPDF uses to create a searchable PDF.

## Bounding Box Accuracy

This plugin includes optimized bounding box calculation for accurate text selection in the output PDF:

### Native Word-Level Boxes (PaddleOCR 3.x)

The plugin uses PaddleOCR 3.x's native `return_word_box=True` parameter to get accurate word-level bounding boxes directly from the OCR engine:
- Native word boxes provide precise boundaries for each word
- Automatic merging of split tokens (handles German umlauts, punctuation, etc.)
- Falls back to estimation algorithm when word boxes aren't available (e.g., blank pages)

**Result**: Word bounding boxes are now pixel-accurate, matching exactly what PaddleOCR detected.

### Polygon-Based Vertical Bounds

Instead of using simple min/max coordinates, the plugin uses PaddleOCR's 4-point polygon geometry:
- For horizontal text, points 0-1 define the top edge and points 2-3 define the bottom edge
- Averaging these edge points provides tighter vertical bounds
- Falls back to min/max for non-standard polygon shapes

**Result**: Line heights are reduced by 2-3 pixels (3-6%), providing tighter text selection without clipping.

These improvements make text selection in the output PDF more precise and visually aligned with the actual text in the document. For technical details, see [CLAUDE.md](CLAUDE.md).

## Troubleshooting

### Import Error: PaddleOCR not found

Make sure PaddlePaddle and PaddleOCR are installed:

```bash
pip install paddlepaddle paddleocr
```

For GPU support:

```bash
# CUDA 11.x
pip install paddlepaddle-gpu

# CUDA 12.x
pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

### Poor OCR Quality

Try these options:

1. Increase image quality: `--oversample 300`
2. Preprocess images: `--clean` or `--deskew`
3. Disable angle classification if it's causing issues: `--paddle-no-angle-cls`

### GPU Not Being Used

Verify PaddlePaddle GPU installation:

```python
import paddle
print(paddle.device.is_compiled_with_cuda())  # Should return True
print(paddle.device.get_device())  # Should show GPU
```

## License

MPL-2.0 - Same as OCRmyPDF

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Credits

- [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) - PDF OCR tool
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Multilingual OCR toolkit
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - Deep learning framework

## See Also

- [OCRmyPDF Documentation](https://ocrmypdf.readthedocs.io/)
- [OCRmyPDF Plugin Development](https://ocrmypdf.readthedocs.io/en/latest/plugins.html)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/main/README_en.md)
