"""
Stage 2: Turn structured OCR text into a strict CSV schema using the
DeepSeek API.

Usage:
    python working_intel_model.py <in_folder> <out_folder>
"""
import os
import sys
import csv
import io
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

EXPECTED_HEADER = "Year,Population,Total Land,Residential,Commercial,Industrial,Public,Streets,Water,Vacant"
EXPECTED_ROW_COUNT = 10
EXPECTED_COLUMN_COUNT = 10


def build_prompt(text: str) -> str:
    return f"""Clean one OCR land-use table into CSV. Output CSV only: no markdown, no prose, no blank lines.

First, emit these three identifier labels, one per line, using values found in the table (blank if absent):
# Development Area: <value>
# Study Area: <value>
# Census Tract: <value>

Then the header (exact):
Year,Population,Total Land,Residential,Commercial,Industrial,Public,Streets,Water,Vacant

Then exactly 10 rows, Year order: 1970,1975,1980,1985,1990,1995,2000,2020,2000,2020. Each row: exactly 10 values, no empty fields.

Rules:
1. The OCR may capture the same table more than once. If the input repeats (e.g. the header/rows appear again), treat it as one table: use a single copy and ignore the duplicates. Never output more than 10 rows.
2. Cells are usually split by "|", but a "|" may be missing between two numbers. If a cell holds two run-together values, split them by the column's known scale (e.g. "407 407" -> 407, 407).
3. In numbers, "," and "." are formatting, not decimals: 6,300 and 6.300 both = 6300.
4. A stray 1-9 at a row's start is a footnote marker: drop it, shift that row's remaining values left.
5. Total Land and Water are constant across all rows.
6. Total Land = Residential+Commercial+Industrial+Public+Streets+Water+Vacant (exclude Year, Population). If a row's sum is off by >1.0, prepend a "# ..." line at the top but still output all rows.
7. Decimals: only treat a column as decimal if its values are genuinely decimal (consistent fractional parts). A single decimal in a column that is otherwise whole numbers is OCR noise — clean it to a whole number. In a real decimal column, give every value one decimal place. Never add decimals to whole-number columns.

Example input (columns abbreviated with ...):
Series "E"
Development Area | 6,229
Year | 1970 | 1975 | ...
Population | 1 | 6,300 | 6,300 | ...
Total Land | 6,229 | 6,229 | ...
Residential | 2 | 407 407 | ...
Public | 5,822 | 5,822 | ...

Example output:
# Development Area: 6229
# Study Area:
# Census Tract:
Year,Population,Total Land,Residential,Commercial,Industrial,Public,Streets,Water,Vacant
1970,6300,6229,407,0,0,5822,0,0,0
1975,6300,6229,407,0,0,5822,0,0,0

Now clean this table:
{text}
"""


def validate_csv(csv_output: str) -> list[str]:
    """Return a list of problems found with the model's CSV output (empty if it looks clean)."""
    problems = []
    lines = [line for line in csv_output.splitlines() if line.strip()]
    data_lines = [line for line in lines if not line.startswith("#")]

    if EXPECTED_HEADER not in lines:
        problems.append("expected header row not found")

    header_idx = next((i for i, line in enumerate(data_lines) if line.strip() == EXPECTED_HEADER), None)
    if header_idx is None:
        problems.append("could not locate header among non-comment lines")
        return problems

    row_lines = data_lines[header_idx + 1:]
    if len(row_lines) != EXPECTED_ROW_COUNT:
        problems.append(f"expected {EXPECTED_ROW_COUNT} data rows, found {len(row_lines)}")

    for row_num, row in enumerate(row_lines, start=1):
        fields = next(csv.reader(io.StringIO(row)))
        if len(fields) != EXPECTED_COLUMN_COUNT:
            problems.append(f"row {row_num} has {len(fields)} columns, expected {EXPECTED_COLUMN_COUNT}")
            continue
        for field in fields:
            try:
                float(field)
            except ValueError:
                problems.append(f"row {row_num} has non-numeric value: {field!r}")

    return problems


def main():
    if len(sys.argv) < 3:
        print("[!] Usage: python working_intel_model.py <in_folder> <new_out_folder>")
        sys.exit(1)
    in_folder, out_folder = Path(sys.argv[1]), Path(sys.argv[2])
    out_folder.mkdir(parents=True, exist_ok=False)
    files = [f for f in in_folder.iterdir() if f.is_file() and f.suffix == ".txt"]

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("[!] Set the DEEPSEEK_API_KEY environment variable before running this script.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    for file in tqdm(files, desc="Prompts", unit="prompt"):
        raw_text = file.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = raw_text.encode("ascii", "ignore").decode("ascii")
        prompt = build_prompt(cleaned_text)

        response = None
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": "You are a precise data cleaning assistant."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}}
            )
            csv_output = response.choices[0].message.content

            problems = validate_csv(csv_output)
            if problems:
                print(f"\n[!] {file.name} produced suspect CSV:")
                for problem in problems:
                    print(f"    - {problem}")

            out_path = out_folder / f"{file.name.replace('.', '_')}.csv"
            out_path.write_text(csv_output, encoding="utf-8")
        except Exception as e:
            print(f"\n[!] {file.name} failed: {e}")

if __name__ == "__main__":
    main()