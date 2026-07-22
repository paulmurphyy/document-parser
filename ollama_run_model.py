"""
Stage 2: Turn structured OCR text into strict JSON via Ollama.
Usage: python ollama_run_model.py <in_folder> <new_out_folder>
"""
import sys, json, re, ollama
from pathlib import Path
from tqdm import tqdm

# Model name for the ollama model used for text extraction, see available models on ollama.com
# and locally installed ones with ollama list in cmd, download a new model with ollama pull [model_name] 
MODEL_NAME = "granite4.1:3b"

# 0 -> inf scale, length of text model allow to generate (keep under 8192 to be safe)
NUM_PREDICT = 8192

# 0 -> inf scale, maximum length of all text model can look at (keep under 32768 to be safe)
NUM_CTX = 32768

# 0.0 -> 1.0 scale, variability of the output model can generate (keep under 0.2 to be safe)
TEMPERATURE = 0.0

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": ["string", "null"]},
        "account_number": {"type": ["string", "null"]},
        "service_address": {"type": ["string", "null"]},
        "date_bill_mailed": {"type": ["string", "null"]},
        "service_description": {"type": ["string", "null"]},
        "meter_number": {"type": ["string", "null"]},
        "date_present_reading": {"type": ["string", "null"]},
        "date_previous_reading": {"type": ["string", "null"]},
        "number_present_reading": {"type": ["integer", "null"]},
        "number_previous_reading": {"type": ["integer", "null"]},
        "total_kwh_used": {"type": ["integer", "null"]},
        "days_billed": {"type": ["integer", "null"]},
        "kwh_used_one_year_ago": {"type": ["integer", "null"]},
    },
    "required": list([
        "city","account_number","service_address","date_bill_mailed",
        "service_description","meter_number","date_present_reading",
        "date_previous_reading","number_present_reading","number_previous_reading",
        "total_kwh_used","days_billed","kwh_used_one_year_ago",
    ]),
}


prompt = f"""Clean this OCR-extracted land use table into a CSV.
Header: Year,Population,Total Land,Residential,Commercial,Industrial,Public,Streets,Water,Vacant
INPUT: Semicolon-separated. Strip comma/dot formatting from numbers when parsing.

IMPORTANT: Total Land = Residential + Commercial + Industrial + Public + Streets + Water + Vacant.
Year and Population are not included in the sum. If a row's sum differs from Total Land by more than 
1.0, flag it. Minor rounding differences are expected.

RULES (apply in order):
1. VERTICAL SPLITS: If a cell is truncated or ends with a dot, check if the same column in the
next row completes it. If concatenation yields a plausible value:
a. Write the merged value into the first cell
b. Delete the second cell from its row
c. Shift every value below it in that column up by exactly one row — each row takes the
    population value from the row below it, all the way to the last data row
d. The very last row in that column will now be empty — leave it for step 3 to fill
Do not leave any intermediate cells empty after shifting. The shift is mandatory.

2. FOOTNOTE ARTIFACTS: Discard any value that is a small integer (1–9) inconsistent with its
column's scale. May apply to multiple rows. Shift column up after discarding. When in doubt,
keep and flag rather than silently discard. After shifting, the last cell in that column will 
be empty — fill it only in step 3.

3. EXTRA ROWS (empty Year): Each non-empty value maps to the column it occupies. The table
contains two duplicate year groups (rows 1–5: 1970–2000, rows 6–10: 1970–2020). Check
plausibility against the same group's neighboring rows, not across groups. Merge into the
row above if plausible, including empty cells left by step 1. Discard only if implausible
in context. Multiple extra rows may each contribute to the same target row.

4. STRUCTURE: Exactly 10 rows: 1970,1975,1980,1985,1990,1995,2000,2020,2000,2020.
Each row must have exactly 10 values. Total Land and Water are constant across all rows.
Within each column, values almost always increase from top to bottom within each year group.

5. DECIMALS: Columns with any decimal value must have every value to exactly one decimal place.
Never add decimals to whole-number columns.

6. OCR CLEANUP: Fix only stray punctuation and misplaced decimals. If a value is an order of
magnitude off from its column's scale and no decimal fix resolves it, flag it as an error.

7. On any flagged error, output CSV anyway with a leading comment row (# ...) per issue.

OUTPUT: Raw CSV only. No markdown, no explanation, no blank lines.
Table:
{table_str}"""


def build_prompt(text: str) -> str:
    return f"""Extract these fields from the electric utility bill below and return one JSON object.
Copy values as they appear (don't reformat dates). Integers must have no commas or units.
Return only the JSON object.

Example:
Input: "Account: 1000000001 | MAIN ST PUMP | Meter 5551234 | Mailed 01/10/2025 | Present 01/05/2025 1000 | Previous 12/05/2024 900 | kWh 100 | Days 31 | Last yr 88 | City of DEMO, AR"
Output: {{"city":"DEMO","account_number":"1000000001","service_address":"MAIN ST PUMP","date_bill_mailed":"01/10/2025","service_description":"MAIN ST PUMP","meter_number":"5551234","date_present_reading":"01/05/2025","date_previous_reading":"12/05/2024","number_present_reading":1000,"number_previous_reading":900,"total_kwh_used":100,"days_billed":31,"kwh_used_one_year_ago":88}}

Bill:
{text}
"""

def strip_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def main():
    if len(sys.argv) < 3:
        print("[!] Usage: python ollama_run_model.py <in_folder> <new_out_folder>")
        sys.exit(1)
    in_folder, out_folder = Path(sys.argv[1]), Path(sys.argv[2])
    out_folder.mkdir(parents=True, exist_ok=False)
    files = [f for f in in_folder.iterdir() if f.is_file() and f.suffix == ".txt"]

    for file in tqdm(files, desc="Prompts", unit="prompt"):
        raw_text = file.read_text(encoding="utf-8", errors="ignore")
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(raw_text.encode("ascii", "ignore").decode("ascii"))}],
            format=SCHEMA,
            options={"temperature": 0.0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
        )
        msg = response["message"]
        result = msg.get("content") or ""
        # Reasoning models sometimes put everything in 'thinking'; use it if content is empty.
        if not result.strip():
            result = msg.get("thinking") or ""
        
        if response.get("done_reason") == "length":
            print(f"\n[!] {file.name}: hit token limit (raise NUM_PREDICT/NUM_CTX)")

        try:
            cleaned = re.search(r"(\{.*\})", strip_think_tags(result), re.DOTALL)
            if not cleaned:
                raise ValueError("[!] No JSON found")
            parsed = json.loads(cleaned.group(1))
            out_path = out_folder / f"{file.name.replace('.', '_')}.json"
            out_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        except Exception:
            print(f"\n[!] {file.name} failed:\n{result}")

if __name__ == "__main__":
    main()