import subprocess, sys
 
OCR_SCRIPT = "paddle_ocr_extract_pdf.py"
TITLE_SCRIPT = "deepseek_extract_title.py"
 
def main():
    if len(sys.argv) < 3:
        print("[!] Usage: run_pipeline.py <in_pdf_folder> <out_ocr_folder>")
        sys.exit(1)
    subprocess.run([sys.executable, OCR_SCRIPT, sys.argv[1], sys.argv[2]], check=True)
    subprocess.run([sys.executable, TITLE_SCRIPT, sys.argv[2], sys.argv[1]], check=True)
    print(f"Titled PDFs are in {sys.argv[1]}; see rename_log.csv there for details")
 
if __name__ == "__main__":
    main()
