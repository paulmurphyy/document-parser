import subprocess, sys

OCR_SCRIPT = "paddle_ocr_extract_pdf.py"
MODEL_SCRIPT = "working_intel_model.py"

def main():
    if len(sys.argv) < 4: 
        print("[!] Usage: run_pipeline.py <in_pdf_path> <out_ocr_folder> <out_json_folder>")
        sys.exit(1)

    subprocess.run([sys.executable, OCR_SCRIPT, sys.argv[1], sys.argv[2]], check = True)
    subprocess.run([sys.executable, MODEL_SCRIPT, sys.argv[2], sys.argv[3]], check = True)

    print(f"JSON output for each page can be found in {sys.argv[3]}, numbered by page")

if __name__ == "__main__":
    main()