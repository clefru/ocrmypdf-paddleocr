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

try:
    from paddleocr import PaddleOCRVL
except ImportError:
    PaddleOCRVL = None

log = logging.getLogger(__name__)


@hookimpl
def add_options(parser):
    """Add PaddleOCR-specific options to the argument parser."""
    paddle = parser.add_argument_group(
        "PaddleOCR",
        "Options for PaddleOCR engine"
    )
    paddle.add_argument(
        '--paddle-engine',
        choices=['classic', 'vl'],
        default='classic',
        dest='paddle_engine',
        help=(
            'OCR engine variant: "classic" (default, fast CNN-based pipeline) or '
            '"vl" (PaddleOCR-VL-1.5, a 0.9B VLM with superior accuracy but slower)'
        ),
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
    engine = getattr(options, 'paddle_engine', 'classic')
    if engine == 'vl':
        if PaddleOCRVL is None:
            from ocrmypdf.exceptions import MissingDependencyError
            raise MissingDependencyError(
                "PaddleOCRVL is not available. "
                "Install it with: pip install 'paddleocr[doc-parser]'"
            )
    else:
        if PaddleOCR is None:
            from ocrmypdf.exceptions import MissingDependencyError
            raise MissingDependencyError(
                "PaddleOCR is not installed. "
                "Install it with: pip install paddlepaddle paddleocr"
            )


def _poly_to_bbox(poly):
    """Convert a 4-point polygon (TL,TR,BR,BL tuples) to a tight bbox.

    Uses polygon-edge averaging for vertical bounds (tighter than min/max).
    """
    xs = [p[0] for p in poly]
    x_min = int(min(xs))
    x_max = int(max(xs))
    # Average top edge (TL+TR) and bottom edge (BR+BL) for tighter vertical fit
    if len(poly) == 4:
        y_min = int((poly[0][1] + poly[1][1]) / 2)
        y_max = int((poly[2][1] + poly[3][1]) / 2)
    else:
        ys = [p[1] for p in poly]
        y_min = int(min(ys))
        y_max = int(max(ys))
    return x_min, y_min, x_max, y_max


def _group_words_into_lines(word_boxes):
    """Group word polygons into text lines by vertical proximity.

    Args:
        word_boxes: list of (text, poly) where poly is a list of 4 (x,y) tuples
                    in TL, TR, BR, BL order.

    Returns:
        list of lines, each line is a list of (text, poly) sorted left-to-right.
    """
    if not word_boxes:
        return []

    def poly_y_center(poly):
        return sum(p[1] for p in poly) / len(poly)

    def poly_height(poly):
        y_top = (poly[0][1] + poly[1][1]) / 2   # average of TL, TR
        y_bot = (poly[2][1] + poly[3][1]) / 2   # average of BR, BL
        return abs(y_bot - y_top)

    # Sort words by Y-center for top-to-bottom processing
    sorted_words = sorted(word_boxes, key=lambda wb: poly_y_center(wb[1]))

    # Estimate grouping threshold from median word height
    heights = sorted([poly_height(wb[1]) for wb in sorted_words])
    median_height = heights[len(heights) // 2] if heights else 20
    threshold = median_height * 0.6

    lines = []
    current_line = [sorted_words[0]]
    current_y = poly_y_center(sorted_words[0][1])

    for word, poly in sorted_words[1:]:
        y_center = poly_y_center(poly)
        if abs(y_center - current_y) <= threshold:
            current_line.append((word, poly))
        else:
            lines.append(current_line)
            current_line = [(word, poly)]
            current_y = y_center

    lines.append(current_line)

    # Sort each line left-to-right by x_min of its polygon
    for i, line in enumerate(lines):
        lines[i] = sorted(line, key=lambda wb: min(p[0] for p in wb[1]))

    return lines


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
        engine = getattr(options, 'paddle_engine', 'classic')
        if engine == 'vl':
            return f"PaddleOCR-VL-1.5 {PaddleOCREngine.version()}"
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
    def _get_paddle_vl(options):
        """Create and configure PaddleOCRVL instance.

        Supports both paddleocr 3.3.x (PaddleOCR-VL v1) and newer versions
        that expose pipeline_version='v1.5' with native spotting mode.
        """
        import inspect
        sig = inspect.signature(PaddleOCRVL.__init__)

        kwargs = {
            # OCRmyPDF handles page rotation and unwarping upstream
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            # Disable layout detection: treat the whole page as one block
            'use_layout_detection': False,
        }

        # pipeline_version only available in paddleocr >= 3.4.x
        if 'pipeline_version' in sig.parameters:
            kwargs['pipeline_version'] = 'v1.5'
            log.debug("PaddleOCRVL: using pipeline_version=v1.5 (spotting mode)")
        else:
            log.debug("PaddleOCRVL: pipeline_version not available, using v1 (ocr mode)")

        if getattr(options, 'paddle_use_gpu', False):
            kwargs['device'] = 'gpu'
        else:
            kwargs['device'] = 'cpu'

        log.debug(f"Creating PaddleOCRVL with kwargs: {kwargs}")
        return PaddleOCRVL(**kwargs)

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
        engine = getattr(options, 'paddle_engine', 'classic')
        if engine == 'vl':
            PaddleOCREngine._generate_hocr_vl(input_file, output_hocr, output_text, options)
        else:
            PaddleOCREngine._generate_hocr_classic(input_file, output_hocr, output_text, options)

    @staticmethod
    def _generate_hocr_vl(input_file: Path, output_hocr: Path, output_text: Path, options):
        """Generate hOCR using PaddleOCR-VL in full-page mode.

        Uses spotting mode (native word boxes) when available (paddleocr >= 3.4.x,
        pipeline_version='v1.5'), otherwise falls back to OCR mode which returns
        text blocks whose word positions are estimated proportionally.
        """
        log.debug(f"Running PaddleOCR-VL on {input_file}")

        vl_pipeline = PaddleOCREngine._get_paddle_vl(options)

        with Image.open(input_file) as img:
            width, height = img.size

        # Detect which mode is available based on pipeline version
        has_spotting = hasattr(vl_pipeline, 'pipeline_version') and \
                       getattr(vl_pipeline, 'pipeline_version', None) == 'v1.5'

        predict_kwargs = {
            'use_layout_detection': False,
            'use_queues': False,  # get direct exceptions, not wrapped RuntimeError
        }
        if has_spotting:
            log.debug("Using spotting mode (native word-level boxes)")
            predict_kwargs['prompt_label'] = 'spotting'
        else:
            log.debug("Using OCR mode (block-level boxes, word positions estimated)")
            predict_kwargs['prompt_label'] = 'ocr'

        result = vl_pipeline.predict(str(input_file), **predict_kwargs)

        # Get language for hOCR
        lang = PaddleOCREngine._get_paddle_lang(options)
        lang_map_reverse = {v: k for k, v in PaddleOCREngine.LANGUAGE_MAP.items()}
        hocr_lang = lang_map_reverse.get(lang, 'eng')

        hocr_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"',
            '    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
            '<head>',
            '<title></title>',
            '<meta http-equiv="content-type" content="text/html; charset=utf-8" />',
            '<meta name="ocr-system" content="PaddleOCR-VL via ocrmypdf-paddleocr" />',
            '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />',
            '</head>',
            '<body>',
            f'<div class="ocr_page" id="page_1" title="bbox 0 0 {width} {height}">',
        ]

        all_text = []
        word_id = 1
        line_id = 1
        carea_id = 1
        par_id = 1

        if result and len(result) > 0:
            page_res = result[0]
            spotting = page_res.get('spotting_res') if hasattr(page_res, 'get') else None

            if spotting and spotting.get('rec_polys') and spotting.get('rec_texts'):
                # ── Spotting path: native word-level polygon boxes ──────────────
                rec_polys = spotting['rec_polys']
                rec_texts = spotting['rec_texts']

                word_boxes = [
                    (txt, poly)
                    for txt, poly in zip(rec_texts, rec_polys)
                    if txt and txt.strip()
                ]

                lines = _group_words_into_lines(word_boxes)
                log.debug(f"PaddleOCR-VL spotting: {len(word_boxes)} words, {len(lines)} lines")

                for line_words in lines:
                    if not line_words:
                        continue

                    all_x0, all_y0, all_x1, all_y1 = [], [], [], []
                    for _txt, poly in line_words:
                        wx0, wy0, wx1, wy1 = _poly_to_bbox(poly)
                        all_x0.append(wx0); all_y0.append(wy0)
                        all_x1.append(wx1); all_y1.append(wy1)

                    lx0, ly0, lx1, ly1 = min(all_x0), min(all_y0), max(all_x1), max(all_y1)
                    all_text.append(' '.join(t for t, _ in line_words))

                    hocr_lines.append(
                        f'<div class="ocr_carea" id="carea_{carea_id}" '
                        f'title="bbox {lx0} {ly0} {lx1} {ly1}">'
                    )
                    hocr_lines.append(
                        f'<p class="ocr_par" id="par_{par_id}" lang="{hocr_lang}" '
                        f'title="bbox {lx0} {ly0} {lx1} {ly1}">'
                    )
                    hocr_lines.append(
                        f'<span class="ocr_line" id="line_{line_id}" '
                        f'title="bbox {lx0} {ly0} {lx1} {ly1}; baseline 0 0; x_wconf 95">'
                    )
                    for i, (word, poly) in enumerate(line_words):
                        wx0, wy0, wx1, wy1 = _poly_to_bbox(poly)
                        we = word.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        hocr_lines.append(
                            f'<span class="ocrx_word" id="word_{word_id}" '
                            f'title="bbox {wx0} {wy0} {wx1} {wy1}; x_wconf 95">{we}</span>'
                        )
                        if i < len(line_words) - 1:
                            hocr_lines.append(' ')
                        word_id += 1

                    hocr_lines.extend(['</span>', '</p>', '</div>'])
                    line_id += 1; carea_id += 1; par_id += 1

            else:
                # ── OCR/parsing_res_list path: block-level boxes ───────────────
                parsing_res = page_res.get('parsing_res_list') \
                    if hasattr(page_res, 'get') else []
                if parsing_res is None:
                    parsing_res = []

                log.debug(f"PaddleOCR-VL ocr: {len(parsing_res)} blocks")

                # Labels that carry readable text content
                TEXT_LABELS = {
                    'text', 'content', 'paragraph_title', 'doc_title',
                    'abstract_title', 'reference_title', 'content_title',
                    'table_title', 'figure_title', 'chart_title',
                    'abstract', 'reference', 'reference_content',
                    'algorithm', 'number', 'footnote', 'header', 'footer',
                    'aside_text', 'vertical_text', 'vision_footnote', 'ocr',
                }

                for block in parsing_res:
                    label = getattr(block, 'label', None) or block.get('block_label', '')
                    content = getattr(block, 'content', None) or block.get('block_content', '')
                    bbox = getattr(block, 'bbox', None) or block.get('block_bbox', None)

                    if not content or not content.strip():
                        continue
                    if label not in TEXT_LABELS:
                        continue
                    if bbox is None or len(bbox) < 4:
                        continue

                    bx0, by0, bx1, by1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                    # Split block content into lines; estimate vertical position per line
                    block_lines = [l for l in content.split('\n') if l.strip()]
                    if not block_lines:
                        continue

                    block_h = max(by1 - by0, 1)
                    line_h = block_h // len(block_lines)

                    hocr_lines.append(
                        f'<div class="ocr_carea" id="carea_{carea_id}" '
                        f'title="bbox {bx0} {by0} {bx1} {by1}">'
                    )
                    hocr_lines.append(
                        f'<p class="ocr_par" id="par_{par_id}" lang="{hocr_lang}" '
                        f'title="bbox {bx0} {by0} {bx1} {by1}">'
                    )

                    for li, line_text in enumerate(block_lines):
                        words = line_text.split()
                        if not words:
                            continue

                        ly0_est = by0 + li * line_h
                        ly1_est = min(by0 + (li + 1) * line_h, by1)
                        all_text.append(line_text)

                        hocr_lines.append(
                            f'<span class="ocr_line" id="line_{line_id}" '
                            f'title="bbox {bx0} {ly0_est} {bx1} {ly1_est}; '
                            f'baseline 0 0; x_wconf 95">'
                        )

                        # Estimate word widths proportionally by character count
                        line_w = bx1 - bx0
                        total_chars = sum(len(w) for w in words)
                        num_spaces = len(words) - 1
                        if total_chars + num_spaces > 0:
                            space_w = int((line_w * num_spaces) / (total_chars + num_spaces))
                        else:
                            space_w = 0
                        word_area_w = line_w - space_w * num_spaces
                        cur_x = bx0

                        for wi, word in enumerate(words):
                            if total_chars > 0:
                                ww = int(word_area_w * len(word) / total_chars)
                            else:
                                ww = line_w // len(words)
                            wx1_est = bx1 if wi == len(words) - 1 else cur_x + ww
                            we = word.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                            hocr_lines.append(
                                f'<span class="ocrx_word" id="word_{word_id}" '
                                f'title="bbox {cur_x} {ly0_est} {wx1_est} {ly1_est}; '
                                f'x_wconf 95">{we}</span>'
                            )
                            if wi < len(words) - 1:
                                hocr_lines.append(' ')
                            cur_x = wx1_est + space_w
                            word_id += 1

                        hocr_lines.append('</span>')  # ocr_line
                        line_id += 1

                    hocr_lines.extend(['</p>', '</div>'])
                    carea_id += 1; par_id += 1

        hocr_lines.extend(['</div>', '</body>', '</html>'])

        output_hocr.write_text('\n'.join(hocr_lines), encoding='utf-8')
        output_text.write_text('\n'.join(all_text), encoding='utf-8')
        log.debug(f"Generated hOCR (VL) with {line_id - 1} lines and {word_id - 1} words")

    @staticmethod
    def _generate_hocr_classic(input_file: Path, output_hocr: Path, output_text: Path, options):
        """Generate hOCR output for an image using classic PaddleOCR pipeline."""
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
