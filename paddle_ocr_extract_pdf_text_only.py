from paddleocr import PaddleOCR
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import cv2
import pymupdf

# 0 -> inf scale, quality of the images fed into the model
DPI_QUALITY = 200

# 0.0 -> 1.0 scale, for what percent of overlap means two text boxes are on the same line
HEIGHT_RATIO = 0.5


def cap_long_side(img, max_side=3000):
    h, w = img.shape[:2]          # numpy arrays: (height, width, channels)
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def group_into_lines(items, height_ratio):
    """Group OCR text boxes into lines based on vertical overlap.

    items: iterable of (text, (xmin, ymin, xmax, ymax)) sorted by ymin.
    Returns a list of {"ymin", "ymax", "items": [(text, xmin), ...]}.
    """
    lines = []

    for text, box in items:
        if len(text) <= 2: continue
        xmin, ymin, xmax, ymax = box
        box_h = ymax - ymin

        matched = None
        for line in lines:
            overlap = min(ymax, line["ymax"]) - max(ymin, line["ymin"])
            if overlap <= 0:
                continue
            if overlap / min(box_h, line["ymax"] - line["ymin"]) >= height_ratio:
                matched = line
                break

        if matched:
            matched["items"].append((text, xmin))
            matched["ymin"] = min(matched["ymin"], ymin)
            matched["ymax"] = max(matched["ymax"], ymax)
        else:
            lines.append({"ymin": ymin, "ymax": ymax, "items": [(text, xmin)]})

    return lines


def extract_page(page, ocr, txt_out, label):
    """Run OCR directly on one rendered page (no layout detection) and append its text to txt_out."""
    page = cap_long_side(page)

    try:
        ocr_out = ocr.predict(page)
    except RuntimeError as e:
        print(f"OCR failed on {label}: {e}")
        return

    if not ocr_out:
        return

    res = ocr_out[0]

    # Save to image for testing purposes
    res.save_to_img(save_path="imgs")

    res_texts = res.json["res"]["rec_texts"]
    res_boxes = res.json["res"]["rec_boxes"]

    if not res_texts:
        return

    # Sort items by top coordinate (ymin)
    items = sorted(zip(res_texts, res_boxes), key=lambda x: x[1][1])
    lines = group_into_lines(items, HEIGHT_RATIO)

    # Sort each line left-to-right and print
    for line in lines:
        line["items"].sort(key=lambda x: x[1])
        txt_out.write("  |  ".join(t for t, _ in line["items"]) + "\n")

    # Blank separation after each page.
    txt_out.write("\n\n")


def main():
    if len(sys.argv) < 3:
        print("[!] Usage: python paddle_ocr_extract_pdf.py <in_pdf_folder> <new_out_folder>")
        sys.exit(1)

    in_folder = Path(sys.argv[1])
    out_folder = Path(sys.argv[2])

    pdf_files = sorted(f for f in in_folder.iterdir() if f.is_file() and f.suffix.lower() == ".pdf")
    if not pdf_files:
        print(f"[!] No PDFs found in {in_folder}")
        sys.exit(1)

    out_folder.mkdir(parents=True, exist_ok=False)

    # Text-only pipeline: no layout model, no doc-orientation/unwarping/textline-orientation
    # classifiers (they add inference cost and aren't needed for these scans). Using
    # PP-OCRv6_medium (released June 2026) — it beats the older PP-OCRv5_server on both
    # detection (+4.6% Hmean) and recognition (+5.1% accuracy) while also being faster on
    # GPU, so there's no accuracy/speed tradeoff versus v5. Detection input size is bumped
    # up since full pages (rather than pre-cropped layout regions) can contain small text
    # that a smaller det limit would miss.
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        text_det_limit_side_len=4000,
        text_det_limit_type="max",
        text_det_thresh=0.3,
        text_det_box_thresh=0.7,
        text_det_unclip_ratio=1.5,
        text_rec_score_thresh=0.5,
        lang="en",
    )

    for pdf_path in tqdm(pdf_files, desc="PDFs", unit="pdf"):
        doc = pymupdf.open(pdf_path)

        # One txt per PDF, named after the PDF; all pages appended into it.
        out_file_path = out_folder / f"{pdf_path.stem}.txt"
        with open(out_file_path, "w", encoding="utf-8") as txt_out:
            for page_num in tqdm(range(len(doc)), desc=pdf_path.name, unit="page", leave=False):
                page = np.array(doc[page_num].get_pixmap(dpi=DPI_QUALITY).pil_image())
                extract_page(page, ocr, txt_out, label=f"{pdf_path.name} page {page_num}")

        doc.close()


if __name__ == "__main__":
    main()