# OCRmyPDF-PaddleOCR Bounding Box Fixes

This document describes the bounding box improvements implemented in the PaddleOCR plugin for OCRmyPDF.

## Overview

Three major improvements to text bounding boxes in the hOCR output:

1. **Native Word Boxes**: Use PaddleOCR 3.x's `return_word_box=True` for accurate word-level detection
2. **Horizontal Fallback**: Improved word width estimation when native boxes unavailable
3. **Vertical Issue**: Tighter line bounding boxes using polygon edge averaging

## Fix 1: Native Word-Level Bounding Boxes

### Implementation

PaddleOCR 3.x supports native word-level bounding boxes via the `return_word_box=True` parameter. The plugin now uses this feature:

```python
result = paddle_ocr.predict(str(input_file), return_word_box=True)

# Extract word-level data
text_words = ocr_result.get('text_word', [])
text_word_regions = ocr_result.get('text_word_region', [])
```

### Token Merging

PaddleOCR may split words unexpectedly (German umlauts, punctuation, emails). The plugin merges adjacent non-whitespace tokens:

```python
# Merge tokens that were split unexpectedly
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
```

The bounding box for merged words is computed as the union of all sub-token boxes, using the polygon-edge method for vertical bounds.

**Location**: `src/ocrmypdf_paddleocr/plugin.py:319-404`

### Results

- Pixel-accurate word boundaries from PaddleOCR's detection
- Proper handling of split tokens (umlauts, punctuation)
- Automatic fallback to estimation when word boxes unavailable

## Fix 2: Word Width Calculation (Fallback)

### Problem

When selecting text in the PDF, word bounding boxes were significantly too short. For example, on a 934-pixel-wide line, words only covered 784 pixels, leaving a ~150px gap at the end.

### Root Cause

The bug was in the word width estimation algorithm (lines 312-342 of `plugin.py`):

```python
# BEFORE (buggy):
total_chars = sum(len(w) for w in words) + len(words) - 1  # Included spaces
word_width = int(line_width * len(word) / total_chars)     # Divided by total with spaces
space_width = int(avg_char_width * 0.3)                     # Only allocated 30% back
```

The algorithm included spaces in `total_chars` when calculating word widths, but only allocated 30% of average character width back for spaces. This caused approximately 14.46px per space to be "lost", accumulating to ~145px over 10 spaces.

### Solution

Modified the algorithm to properly separate word character allocation from space allocation:

```python
# AFTER (fixed):
total_chars = sum(len(w) for w in words)  # Exclude spaces from character count
# Calculate space width separately
total_space_width = line_width - total_chars * (line_width / (total_chars + num_spaces))
space_width = int(total_space_width / num_spaces) if num_spaces > 0 else 0
# Width available for actual word characters
word_area_width = line_width - (space_width * num_spaces)
# Calculate word width using only the word area
word_width = int(word_area_width * len(word) / total_chars)
# Make last word extend to line end to avoid rounding errors
if i == len(words) - 1:
    word_x_max = x_max
```

**Location**: `src/ocrmypdf_paddleocr/plugin.py:306-343`

### Results

- Word bounding boxes now extend to the full line width
- Zero gap at line ends
- Proper proportional allocation for both words and spaces

## Fix 3: Vertical Bounds Using Polygon Edges

### Problem

Line bounding boxes had excess vertical padding, with the highlight extending too far above and below the actual text.

### Analysis

PaddleOCR returns 4-point polygons for each text line. The original code used min/max of all Y-coordinates to calculate bounding boxes:

```python
# BEFORE:
ys = poly[:, 1]
y_min = int(min(ys))
y_max = int(max(ys))
```

This approach includes any slight rotation or skew in the polygon corners, adding 1-3px of padding on each side.

### Solution

For horizontal text, PaddleOCR's 4-point polygons have a specific structure:
- Points 0-1 define the **top edge**
- Points 2-3 define the **bottom edge**

By averaging the Y-coordinates of each edge, we get tighter bounds that follow the actual text baseline and top:

```python
# AFTER (polygon edge method):
if isinstance(poly, np.ndarray):
    poly_scaled = poly * [scale_x, scale_y]

    # For horizontal bounds, use min/max
    x_min = int(poly_scaled[:, 0].min())
    x_max = int(poly_scaled[:, 0].max())

    # For vertical bounds, use polygon edges for tighter fit
    if len(poly_scaled) == 4:
        y_min = int((poly_scaled[0][1] + poly_scaled[1][1]) / 2)
        y_max = int((poly_scaled[2][1] + poly_scaled[3][1]) / 2)
    else:
        # Fallback to min/max for non-standard polygons
        y_min = int(poly_scaled[:, 1].min())
        y_max = int(poly_scaled[:, 1].max())
```

**Location**: `src/ocrmypdf_paddleocr/plugin.py:277-301`

### Results

Analysis of actual text from test documents:

```
Text: 'Ein- und Zweiobjekthaus i.S.d. § 1 Abs 2 Z 5 MRG'
  OLD (min/max):     height=53px
  NEW (poly edges):  height=51px
  ✓ Reduced by 2px (3.8%)

Text: 'Vermieter:'
  OLD (min/max):     height=49px
  NEW (poly edges):  height=46px
  ✓ Reduced by 3px (6.1%)

Text: 'MA Theresa Rosna'
  OLD (min/max):     height=49px
  NEW (poly edges):  height=47px
  ✓ Reduced by 2px (4.1%)
```

Typical reduction: **2-3px (3-6% of height)**

### Why This Approach

This is a **data-driven approach** that uses PaddleOCR's actual polygon geometry rather than guessing percentage-based padding values. It provides the tightest possible bounds without clipping actual text.

## Testing

Test script to verify the improvements:

```bash
# Generate test PDF with polygon edge method
nix-shell --run 'ocrmypdf -l deu --plugin ocrmypdf_paddleocr /tmp/test_page.png /tmp/test_output.pdf'

# Compare bounding box methods
nix-shell --run 'python3 /tmp/test_bbox_comparison.py'
```

## Technical Notes

### PaddleOCR Polygon Structure

PaddleOCR returns polygons in this format for horizontal text:

```
Point 0 (top-left) -------- Point 1 (top-right)
       |                           |
       |        TEXT HERE          |
       |                           |
Point 3 (bottom-left) ---- Point 2 (bottom-right)
```

The polygon may have slight rotation/skew, which is why averaging the edges gives better results than min/max.

### Coordinate System

- Origin (0,0) is at top-left of image
- X increases rightward
- Y increases downward
- All coordinates are scaled from PaddleOCR's resized image back to original resolution

### Word-Level Detection

The plugin uses PaddleOCR 3.x's native `return_word_box=True` parameter for accurate word-level bounding boxes. When native word boxes are available:
1. Extract `text_word` and `text_word_region` from OCR results
2. Merge split tokens (handles umlauts, punctuation)
3. Compute union bounding box using polygon-edge vertical bounds

When native word boxes are unavailable (blank pages, older PaddleOCR versions), the plugin falls back to estimation:
1. Splitting text on whitespace
2. Allocating horizontal space proportionally by character count
3. Using the same vertical bounds for all words in a line

## Files Modified

- `src/ocrmypdf_paddleocr/plugin.py` - Lines 207, 267-272, 319-460
  - Native word boxes: lines 319-404
  - Fallback word estimation: lines 405-453
  - Vertical bounds (polygon edges): lines 284-296, 371-376
- `shell.nix` - Lines 57-61
  - Added paddlex-with-ocr and paddleocr-with-ocr to environment
  - Added missing OCR dependencies (python-bidi, sentencepiece)

## Related Issues

- Word bounding boxes were too short (fixed with native boxes)
- Line bounding boxes had excess vertical padding (improved by 3-6%)
- PaddleX OCR dependencies missing in Nix environment (fixed)

## Future Improvements

1. **Character-level boxes**: PaddleOCR can provide character-level boxes for even more precise selection
2. **Baseline detection**: Use PaddleOCR's text direction/angle detection to handle rotated text better
3. **Per-word confidence scores**: Extract individual word confidence from PaddleOCR when available

## References

- PaddleOCR documentation: https://github.com/PaddlePaddle/PaddleOCR
- hOCR specification: https://github.com/kba/hocr-spec
- OCRmyPDF plugin API: https://ocrmypdf.readthedocs.io/
