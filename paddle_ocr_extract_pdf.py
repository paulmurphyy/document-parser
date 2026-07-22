from paddlex import create_model
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import sys
import cv2

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


def main():
    if len(sys.argv) < 3:
        print("[!] Usage: python paddle_ocr_extract_pdf.py <in_pdf> <new_out_folder>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    out_folder = Path(sys.argv[2])

    out_folder.mkdir(parents=True, exist_ok=False)

    model_name = "PP-DocLayout_plus-L"
    model = create_model(model_name=model_name)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en"
    )

    pages = [np.array(Image.open(file.as_posix()).convert("RGB")) for file in sorted(file_path.iterdir())]

    for count, page in enumerate(tqdm(pages, desc="Pages", unit="page")):
        page = cap_long_side(page)
        output = model.predict(page, batch_size=1, layout_nms=True)
        res = next(output, None)
        res_boxes = res.json["res"]["boxes"]

        # Save to image for testing purposes
        # res.save_to_img(save_path="imgs")

        out_file_path = out_folder / f"{count}.txt"

        with open(out_file_path, "a", encoding="utf-8") as txt_out:
            for box in res_boxes:
                page_crop = page[round(box["coordinate"][1]):round(box["coordinate"][3]), round(box["coordinate"][0]):round(box["coordinate"][2])]

                try:
                    crop_out = ocr.predict(page_crop)
                except RuntimeError as e:
                    print(f"OCR failed on page {count}: {e}")
                    continue
                crop_res = crop_out[0]

                crop_res_texts = crop_res.json["res"]["rec_texts"]
                crop_res_boxes = crop_res.json["res"]["rec_boxes"]

                # Sort items by top coordinate (ymin)
                items = sorted(zip(crop_res_texts, crop_res_boxes), key=lambda x: x[1][1])
                lines = group_into_lines(items, HEIGHT_RATIO)

                # Sort each line left-to-right and print
                for line in lines:
                    line["items"].sort(key=lambda x: x[1])
                    txt_out.write("  |  ".join(t for t, _ in line["items"]) + "\n")

                txt_out.write("\n\n")

if __name__ == "__main__":
    main()