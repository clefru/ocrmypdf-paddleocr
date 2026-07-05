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

### ARM64

Although `paddlepaddle` `aarch64` builds are available, you may still need to compile from source due to [incompatbility issues between hardware](https://github.com/PaddlePaddle/Paddle/issues/78616):

<details>

<summary>
Docker
</summary>

```bash
# (most of the instructions are Chinese-only, although I followed along with Google Translate version, it was clear enough)
# https://www.paddlepaddle.org.cn/documentation/docs/en/install/compile/linux-compile-by-make.html#span-id-compile-from-host-span

docker run -it ubuntu:24.04 /bin/bash

apt update
apt upgrade -y
apt install -y bzip2 make
apt install -y cmake gcc
apt install -y python3-dev

find `dirname $(dirname $(which python3))` -name "libpython3*.so"
export PYTHON_LIBRARY=/usr/lib/python3.12/config-3.12-aarch64-linux-gnu/

find `dirname $(dirname $(which python3))`/include -name "python3.12"

# https://www.ecosia.org/ai-chat/2159e6c5bc2a1300433646a0af43f77ccabec4a5d946e5d2edbced5ea4468884
# export PYTHON_INCLUDE_DIRS=/usr/include/aarch64-linux-gnu/python3.12/
export PYTHON_INCLUDE_DIRS=/usr/include/python3.12/
export PATH=/usr/bin/:$PATH

apt install -y python3-virtualenvwrapper

find / -name virtualenvwrapper.sh
apt install -y nano
nano /usr/share/virtualenvwrapper/virtualenvwrapper.sh

mkdir $HOME/.virtualenvs
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "source /usr/share/virtualenvwrapper/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
workon

export VIRTUALENVWRAPPER_PYTHON=/usr/bin/:$PATH

mkvirtualenv paddle-venv
workon paddle-venv

# https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#third_party

apt install -y patchelf
apt install -y git

git clone --recursive https://github.com/PaddlePaddle/Paddle.git
cd Paddle

git checkout develop
mkdir build && cd build

pip install -r ../python/requirements.txt
pip install wheel

# https://github.com/PaddlePaddle/Paddle/issues/78616#issuecomment-4221894029

cmake .. -DPY_VERSION=3.12 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
-DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF -DWITH_ARM=ON -DTARGET_AARCH64=ON

make -j$(nproc)
```
</details>
If the build was successful you will end up with a `.whl` file inside `/Paddle/build/python/dist/`: you can then `docker cp running-container-id:/Paddle/build/python/dist/*.whl desired/location/on/device` (check the container id with `docker ps`). 

After that create a virtual environment that matches the version you built (I built on Python `3.12`, which is why my wheel contains the string `cp312`, so I created an environment with `uv venv -p 3.12`). Then `source .venv/bin/activate` and `uv pip install -U paddlepaddle*.whl`: now you can `cd` to a local clone of this repo and continue installation as usual (`uv pip install .`).

To make sure `paddlepaddle` is working properly before installing `ocrmypdf`, you can [test your installation with these steps from the PaddlePaddle docs](https://www.paddlepaddle.org.cn/documentation/docs/en/install/compile/linux-compile-by-make.html#yanzhenganzhuang):
```bash
python # enters a subshell into which you can execute Python code for testing
import paddle
paddle.utils.run_check()
```

After you exit your container, if you ever want to fire it up again, remember you can always run `docker start container-id-or-name` `docker exec -it container-id-or-name /bin/bash` (check container id with `docker ps -a` this time as the container isn't running).

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
