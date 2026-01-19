"""PaddleOCR engine plugin for OCRmyPDF."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

log = logging.getLogger(__name__)


@hookimpl
def add_options(parser):
    """Add PaddleOCR-specific options to the argument parser."""
    paddle = parser.add_argument_group(
        "PaddleOCR",
        "Options for PaddleOCR engine"
    )
    paddle.add_argument(
        '--paddle-use-gpu',
        action='store_true',
        help='Use GPU acceleration for PaddleOCR (requires GPU-enabled PaddlePaddle)',
    )
    paddle.add_argument(
        '--paddle-no-angle-cls',
        action='store_false',
        dest='paddle_use_angle_cls',
        default=True,
        help='Disable text orientation classification',
    )
    paddle.add_argument(
        '--paddle-show-log',
        action='store_true',
        help='Show PaddleOCR internal logging',
    )
    paddle.add_argument(
        '--paddle-det-model-dir',
        metavar='DIR',
        help='Path to text detection model directory',
    )
    paddle.add_argument(
        '--paddle-rec-model-dir',
        metavar='DIR',
        help='Path to text recognition model directory',
    )
    paddle.add_argument(
        '--paddle-cls-model-dir',
        metavar='DIR',
        help='Path to text orientation classification model directory',
    )


@hookimpl
def check_options(options):
    """Validate PaddleOCR options."""
    if PaddleOCR is None:
        from ocrmypdf.exceptions import MissingDependencyError
        raise MissingDependencyError(
            "PaddleOCR is not installed. "
            "Install it with: pip install paddlepaddle paddleocr"
        )


class PaddleOCREngine(OcrEngine):
    """Implements OCR with PaddleOCR."""

    # Mapping from Tesseract/OCRmyPDF language codes to PaddleOCR codes
    LANGUAGE_MAP = {
        'eng': 'en',
        'chi_sim': 'ch',
        'chi_tra': 'chinese_cht',
        'fra': 'fr',
        'deu': 'german',
        'jpn': 'japan',
        'kor': 'korean',
        'spa': 'spanish',
        'rus': 'ru',
        'ara': 'ar',
        'hin': 'hi',
        'por': 'pt',
        'ita': 'it',
        'tur': 'tr',
        'vie': 'vi',
        'tha': 'th',
    }

    @staticmethod
    def version():
        """Return PaddleOCR version."""
        try:
            import paddleocr
            return paddleocr.__version__
        except (ImportError, AttributeError):
            return "2.7.0"

    @staticmethod
    def creator_tag(options):
        """Return the creator tag to identify this software."""
        return f"PaddleOCR {PaddleOCREngine.version()}"

    def __str__(self):
        """Return name of OCR engine and version."""
        return f"PaddleOCR {PaddleOCREngine.version()}"

    @staticmethod
    def languages(options):
        """Return the set of all languages supported by PaddleOCR."""
        # PaddleOCR supports many languages - return a comprehensive list
        return {
            'en', 'ch', 'chinese_cht', 'ta', 'te', 'ka', 'latin', 'ar', 'cy', 'da',
            'de', 'es', 'et', 'fr', 'ga', 'hi', 'it', 'ja', 'ko', 'la', 'nl', 'no',
            'oc', 'pt', 'ro', 'ru', 'sr', 'sv', 'tr', 'uk', 'vi',
            # Also include common Tesseract codes for compatibility
            'eng', 'chi_sim', 'chi_tra', 'deu', 'fra', 'spa', 'rus', 'jpn', 'kor'
        }

    @staticmethod
    def _get_paddle_lang(options):
        """Convert OCRmyPDF language to PaddleOCR language."""
        if not options.languages:
            return 'en'

        # Use first language
        lang = options.languages[0].lower()
        return PaddleOCREngine.LANGUAGE_MAP.get(lang, lang)

    @staticmethod
    def _get_paddle_ocr(options):
        """Create and configure PaddleOCR instance."""
        # OCRmyPDF's Tesseract plugin sets OMP_THREAD_LIMIT to limit Tesseract threading.
        # This affects all plugins in the process. PaddleOCR needs more threads to work properly.
        # Temporarily unset it before initializing PaddleOCR.
        import os
        saved_omp_limit = os.environ.get('OMP_THREAD_LIMIT')
        if saved_omp_limit:
            log.warning(f"Removing OMP_THREAD_LIMIT={saved_omp_limit} set by Tesseract plugin")
            del os.environ['OMP_THREAD_LIMIT']

        paddle_lang = PaddleOCREngine._get_paddle_lang(options)
        log.debug(f"Initializing PaddleOCR with language: {paddle_lang}")

        kwargs = {
            # Disable textline orientation - not needed for most documents
            'use_textline_orientation': False,
            'lang': paddle_lang,
            # Disable document unwarping - coordinates must match original image
            'use_doc_unwarping': False,
            # Disable orientation classification - OCRmyPDF handles page rotation
            'use_doc_orientation_classify': False,
        }

        # Set device for GPU/CPU
        if getattr(options, 'paddle_use_gpu', False):
            kwargs['device'] = 'gpu'
        else:
            kwargs['device'] = 'cpu'

        # Add model directories if specified
        if hasattr(options, 'paddle_det_model_dir') and options.paddle_det_model_dir:
            kwargs['text_detection_model_dir'] = options.paddle_det_model_dir
        if hasattr(options, 'paddle_rec_model_dir') and options.paddle_rec_model_dir:
            kwargs['text_recognition_model_dir'] = options.paddle_rec_model_dir
        if hasattr(options, 'paddle_cls_model_dir') and options.paddle_cls_model_dir:
            kwargs['textline_orientation_model_dir'] = options.paddle_cls_model_dir

        log.debug(f"Creating PaddleOCR with kwargs: {kwargs}")
        return PaddleOCR(**kwargs)

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        """Get page orientation."""
        # PaddleOCR handles orientation internally if use_angle_cls=True
        # Since we enable angle classification by default, we return neutral values
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        """Get deskew angle."""
        # PaddleOCR doesn't provide deskew information
        return 0.0

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options):
        """Generate hOCR output for an image."""
        log.debug(f"Running PaddleOCR on {input_file}")

        # Initialize PaddleOCR
        paddle_ocr = PaddleOCREngine._get_paddle_ocr(options)

        # Get image dimensions and DPI info
        with Image.open(input_file) as img:
            width, height = img.size
            dpi = img.info.get('dpi', (300, 300))
            log.debug(f"Input image: {width}x{height}, DPI: {dpi}")

        # Run OCR - use predict() instead of deprecated ocr()
        # Enable return_word_box=True for native word-level bounding boxes
        result = paddle_ocr.predict(str(input_file), return_word_box=True)

        # Calculate scaling factors from preprocessed image
        scale_x = 1.0
        scale_y = 1.0
        if result and len(result) > 0:
            ocr_result = result[0]

            # Check if there's a preprocessed image in the result
            if hasattr(ocr_result, 'get'):
                # Look for doc_preprocessor_res which contains the unwarped image
                doc_prep_res = ocr_result.get('doc_preprocessor_res')
                if doc_prep_res:
                    if hasattr(doc_prep_res, 'get'):
                        # The preprocessed image is in 'output_img' field
                        preprocessed_img = doc_prep_res.get('output_img')
                        if preprocessed_img is not None:
                            import numpy as np
                            if isinstance(preprocessed_img, np.ndarray):
                                prep_height, prep_width = preprocessed_img.shape[:2]
                                scale_x = width / prep_width
                                scale_y = height / prep_height
                                log.debug(f"Preprocessed image: {prep_width}x{prep_height}, "
                                         f"scaling factors: x={scale_x:.4f}, y={scale_y:.4f}")

        # Get language for hOCR
        lang = PaddleOCREngine._get_paddle_lang(options)
        # Map back to Tesseract-style language codes for compatibility
        lang_map_reverse = {v: k for k, v in PaddleOCREngine.LANGUAGE_MAP.items()}
        hocr_lang = lang_map_reverse.get(lang, 'eng')

        # Convert PaddleOCR 3.x output to hOCR
        hocr_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"',
            '    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
            '<head>',
            '<title></title>',
            '<meta http-equiv="content-type" content="text/html; charset=utf-8" />',
            '<meta name="ocr-system" content="PaddleOCR via ocrmypdf-paddleocr" />',
            '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />',
            '</head>',
            '<body>',
            f'<div class="ocr_page" id="page_1" title="bbox 0 0 {width} {height}">',
        ]

        # Collect all text for output_text
        all_text = []

        # PaddleOCR 3.x returns a list of OCRResult objects
        if result and len(result) > 0:
            ocr_result = result[0]  # Get first page result

            # OCRResult is a dict-like object with keys: rec_texts, rec_scores, rec_polys
            # With return_word_box=True, also: text_word, text_word_region
            texts = ocr_result.get('rec_texts', [])
            scores = ocr_result.get('rec_scores', [])
            polys = ocr_result.get('rec_polys', [])

            # Word-level data (from return_word_box=True)
            text_words = ocr_result.get('text_word', [])
            text_word_regions = ocr_result.get('text_word_region', [])

            has_word_boxes = bool(text_words and text_word_regions)
            log.debug(f"PaddleOCR found {len(texts)} text regions, word boxes: {has_word_boxes}")

            word_id = 1
            carea_id = 1
            par_id = 1

            for line_id, (text, score, poly) in enumerate(zip(texts, scores, polys), 1):
                if not text:
                    continue

                all_text.append(text)

                # poly is a numpy array of shape (N, 2) with polygon points
                # Convert to bounding box and apply scaling to map back to original image
                import numpy as np
                if isinstance(poly, np.ndarray):
                    # Apply scaling to map back to original image
                    poly_scaled = poly * [scale_x, scale_y]

                    # For horizontal bounds, use min/max
                    x_min = int(poly_scaled[:, 0].min())
                    x_max = int(poly_scaled[:, 0].max())

                    # For vertical bounds, use polygon edges for tighter fit
                    # For 4-point polygons: points 0-1 are top edge, points 2-3 are bottom edge
                    if len(poly_scaled) == 4:
                        y_min = int((poly_scaled[0][1] + poly_scaled[1][1]) / 2)
                        y_max = int((poly_scaled[2][1] + poly_scaled[3][1]) / 2)
                    else:
                        # Fallback to min/max for non-standard polygons
                        y_min = int(poly_scaled[:, 1].min())
                        y_max = int(poly_scaled[:, 1].max())
                else:
                    # Fallback if not numpy array
                    xs = [int(point[0] * scale_x) for point in poly]
                    ys = [int(point[1] * scale_y) for point in poly]
                    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

                conf_pct = int(score * 100)

                # Create a carea and par for each line (simple structure)
                hocr_lines.append(
                    f'<div class="ocr_carea" id="carea_{carea_id}" title="bbox {x_min} {y_min} {x_max} {y_max}">'
                )
                hocr_lines.append(
                    f'<p class="ocr_par" id="par_{par_id}" lang="{hocr_lang}" title="bbox {x_min} {y_min} {x_max} {y_max}">'
                )

                # Start the line span with baseline info
                hocr_lines.append(
                    f'<span class="ocr_line" id="line_{line_id}" '
                    f'title="bbox {x_min} {y_min} {x_max} {y_max}; baseline 0 0; x_wconf {conf_pct}">'
                )

                # Process word-level bounding boxes
                # Use native word boxes from PaddleOCR when available
                line_idx = line_id - 1  # 0-indexed for accessing word data

                if (has_word_boxes and
                    line_idx < len(text_words) and
                    line_idx < len(text_word_regions) and
                    text_words[line_idx] and
                    text_word_regions[line_idx]):
                    # Use native word boxes from PaddleOCR
                    line_word_tokens = text_words[line_idx]
                    line_word_boxes = text_word_regions[line_idx]

                    # Merge tokens that were split unexpectedly (punctuation, umlauts)
                    # Group non-whitespace tokens and compute union of their boxes
                    merged_words = []
                    current_word = []
                    current_boxes = []

                    for token, box in zip(line_word_tokens, line_word_boxes):
                        token_str = str(token).strip()
                        if not token_str or token_str.isspace():
                            # Whitespace token - finalize current word
                            if current_word:
                                merged_words.append((''.join(current_word), current_boxes))
                                current_word = []
                                current_boxes = []
                        else:
                            # Non-whitespace token - accumulate
                            current_word.append(token_str)
                            current_boxes.append(box)

                    # Don't forget last word
                    if current_word:
                        merged_words.append((''.join(current_word), current_boxes))

                    # Output merged words with their bounding boxes
                    for i, (word, boxes) in enumerate(merged_words):
                        if not word:
                            continue

                        # Compute union box of all sub-token boxes
                        import numpy as np
                        all_xs = []
                        all_ys_top = []
                        all_ys_bottom = []

                        for box in boxes:
                            if isinstance(box, np.ndarray):
                                box_scaled = box * [scale_x, scale_y]
                                all_xs.extend(box_scaled[:, 0])
                                # Use polygon edge method for vertical bounds
                                if len(box_scaled) == 4:
                                    all_ys_top.append((box_scaled[0][1] + box_scaled[1][1]) / 2)
                                    all_ys_bottom.append((box_scaled[2][1] + box_scaled[3][1]) / 2)
                                else:
                                    all_ys_top.append(box_scaled[:, 1].min())
                                    all_ys_bottom.append(box_scaled[:, 1].max())
                            else:
                                # Fallback for non-numpy boxes
                                for point in box:
                                    all_xs.append(point[0] * scale_x)
                                    all_ys_top.append(point[1] * scale_y)
                                    all_ys_bottom.append(point[1] * scale_y)

                        word_x_min = int(min(all_xs))
                        word_x_max = int(max(all_xs))
                        word_y_min = int(min(all_ys_top))
                        word_y_max = int(max(all_ys_bottom))

                        # Escape HTML entities in word
                        word_escaped = (word.replace('&', '&amp;')
                                           .replace('<', '&lt;')
                                           .replace('>', '&gt;'))

                        hocr_lines.append(
                            f'<span class="ocrx_word" id="word_{word_id}" '
                            f'title="bbox {word_x_min} {word_y_min} {word_x_max} {word_y_max}; '
                            f'x_wconf {conf_pct}">{word_escaped}</span>'
                        )

                        # Add space after word (except for last word)
                        if i < len(merged_words) - 1:
                            hocr_lines.append(' ')

                        word_id += 1
                else:
                    # Fallback: estimate word boxes from line box
                    # Split text into words and estimate bounding boxes
                    words = text.split()
                    if words:
                        line_width = x_max - x_min
                        # Calculate width available for words (excluding spaces)
                        total_chars = sum(len(w) for w in words)
                        num_spaces = len(words) - 1
                        # Allocate space for inter-word spaces
                        if total_chars + num_spaces > 0:
                            total_space_width = line_width - total_chars * (line_width / (total_chars + num_spaces))
                            space_width = int(total_space_width / num_spaces) if num_spaces > 0 else 0
                        else:
                            space_width = 0
                        # Width available for actual word characters
                        word_area_width = line_width - (space_width * num_spaces)

                        current_x = x_min
                        for i, word in enumerate(words):
                            # Estimate word width based on character proportion
                            if total_chars > 0:
                                word_width = int(word_area_width * len(word) / total_chars)
                            else:
                                word_width = line_width // len(words)

                            # For the last word, extend to line end to avoid rounding errors
                            if i == len(words) - 1:
                                word_x_max = x_max
                            else:
                                word_x_max = current_x + word_width

                            # Escape HTML entities in word
                            word_escaped = (word.replace('&', '&amp;')
                                               .replace('<', '&lt;')
                                               .replace('>', '&gt;'))

                            hocr_lines.append(
                                f'<span class="ocrx_word" id="word_{word_id}" '
                                f'title="bbox {current_x} {y_min} {word_x_max} {y_max}; '
                                f'x_wconf {conf_pct}">{word_escaped}</span>'
                            )

                            # Add space after word (except for last word)
                            if i < len(words) - 1:
                                hocr_lines.append(' ')

                            current_x = word_x_max + space_width
                            word_id += 1

                # Close the line span
                hocr_lines.append('</span>')
                # Close par and carea for this line
                hocr_lines.append('</p>')
                hocr_lines.append('</div>')

                carea_id += 1
                par_id += 1

        hocr_lines.extend([
            '</div>',  # ocr_page
            '</body>',
            '</html>',
        ])

        # Write hOCR output
        output_hocr.write_text('\n'.join(hocr_lines), encoding='utf-8')

        # Write text output
        text_content = '\n'.join(all_text)
        output_text.write_text(text_content, encoding='utf-8')

        log.debug(f"Generated hOCR with {len(all_text)} text regions")

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options):
        """Generate a text-only PDF from an image.

        PaddleOCR doesn't have native PDF generation, so we use hOCR as intermediate
        and convert it to PDF using OCRmyPDF's HocrTransform.
        """
        log.debug(f"Generating PDF from {input_file}")

        # Create a temporary hOCR file
        output_hocr = output_pdf.with_suffix('.hocr')

        # Generate hOCR
        PaddleOCREngine.generate_hocr(input_file, output_hocr, output_text, options)

        # Convert hOCR to PDF using OCRmyPDF's hocrtransform
        from ocrmypdf.hocrtransform import HocrTransform

        # Get DPI from image
        from PIL import Image
        with Image.open(input_file) as img:
            dpi = img.info.get('dpi', (300, 300))[0]  # Default to 300 DPI if not set

        hocr_transform = HocrTransform(
            hocr_filename=output_hocr,
            dpi=dpi
        )
        hocr_transform.to_pdf(
            out_filename=output_pdf,
            image_filename=input_file,
            invisible_text=True  # Text should be invisible since it's an overlay
        )


@hookimpl
def get_ocr_engine():
    """Register PaddleOCR as an OCR engine."""
    return PaddleOCREngine()
