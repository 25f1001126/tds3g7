import os
import re
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from dateutil import parser as dateparser
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    document_id: str
    text: str
    schema_: Optional[dict] = Field(default=None, alias="schema")


EXPECTED_KEYS = [
    "vendor", "currency", "total_amount", "invoice_date", "due_in_days",
    "is_paid", "priority", "contact_email", "line_items", "item_count"
]

# ---------------------------------------------------------------------------
# Deterministic helpers (no LLM — same input always gives same output)
# ---------------------------------------------------------------------------

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "a": 1, "an": 1
}
SCALE_WORDS = {"hundred": 100, "thousand": 1000, "million": 1000000}


def words_to_number(text: str) -> int:
    text = text.lower().replace("-", " ")
    tokens = [t for t in re.split(r"[\s,]+", text) if t and t != "and"]
    total, current = 0, 0
    matched_any = False
    for tok in tokens:
        if tok in NUM_WORDS:
            current += NUM_WORDS[tok]
            matched_any = True
        elif tok in SCALE_WORDS:
            scale = SCALE_WORDS[tok]
            matched_any = True
            if scale == 100:
                current = (current or 1) * scale
            else:
                total += (current or 1) * scale
                current = 0
    return (total + current) if matched_any else 0


CURRENCY_MAP = [
    (r"\bUSD\b|\$|\bdollars?\b", "USD"),
    (r"\bEUR\b|€|\beuros?\b", "EUR"),
    (r"\bGBP\b|£|\bpounds?\s*sterling\b|\bpounds?\b", "GBP"),
    (r"\bINR\b|₹|\brupees?\b", "INR"),
    (r"\bJPY\b|¥|\byen\b", "JPY"),
]


def extract_currency(text: str) -> str:
    for pattern, code in CURRENCY_MAP:
        if re.search(pattern, text, re.IGNORECASE):
            return code
    return "USD"


def extract_total_amount(text: str) -> int:
    label_pattern = re.compile(
        r"(total amount due|grand total|total due|amount due|total)\s*[:\-]?\s*([^\n\.]{0,80})",
        re.IGNORECASE
    )
    m = label_pattern.search(text)
    segment = m.group(2) if m else text

    # "12K" style shorthand
    k_match = re.search(r"([\d,]+(?:\.\d+)?)\s*[kK]\b", segment)
    if k_match:
        return int(round(float(k_match.group(1).replace(",", "")) * 1000))

    # digit-based amount, handles both Western (12,480) and Indian (1,24,800) grouping —
    # stripping commas/currency symbols works for both since digit order is unaffected
    digit_match = re.search(r"[\$€£¥₹]?\s*(\d[\d,]*)(?:\.\d+)?", segment)
    if digit_match:
        raw = digit_match.group(1).replace(",", "")
        if raw.isdigit():
            return int(raw)

    # spelled-out numbers
    words_num = words_to_number(segment)
    if words_num > 0:
        return words_num

    # fallback: scan whole document if the labeled segment had nothing
    if m:
        digit_match = re.search(r"[\$€£¥₹]?\s*(\d[\d,]*)(?:\.\d+)?", text)
        if digit_match:
            raw = digit_match.group(1).replace(",", "")
            if raw.isdigit():
                return int(raw)
        return words_to_number(text)

    return 0


def extract_invoice_date(text: str) -> str:
    label_match = re.search(
        r"invoice\s*date\s*[:\-]?\s*([A-Za-z0-9,\/\-\. ]{6,25})", text, re.IGNORECASE
    )
    candidates = [label_match.group(1)] if label_match else []

    generic_dates = re.findall(
        r"\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
        r"\d{4}-\d{2}-\d{2}|"
        r"[A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}|"
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
        text
    )
    candidates.extend(generic_dates)

    for cand in candidates:
        try:
            dt = dateparser.parse(cand, fuzzy=True, dayfirst=False)
            if dt:
                return dt.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            continue
    return ""


def extract_due_in_days(text: str) -> int:
    if re.search(r"due\s+on\s+receipt", text, re.IGNORECASE):
        return 0

    net_match = re.search(r"\bnet\s*(\d+)\b", text, re.IGNORECASE)
    if net_match:
        return int(net_match.group(1))

    unit_map = {"day": 1, "week": 7, "month": 30, "year": 365}
    phrase_match = re.search(
        r"(?:within|in|due in)\s+([a-zA-Z0-9\- ]+?)\s+(day|week|month|year)s?\b",
        text, re.IGNORECASE
    )
    if phrase_match:
        qty_text = phrase_match.group(1).strip()
        unit = phrase_match.group(2).lower()
        qty = int(qty_text) if qty_text.isdigit() else words_to_number(qty_text)
        qty = qty or 1
        return qty * unit_map[unit]

    return 0


def extract_is_paid(text: str) -> bool:
    if re.search(r"paid\s+in\s+full|payment\s+received|fully\s+paid|already\s+paid", text, re.IGNORECASE):
        return True
    if re.search(r"awaiting\s+payment|outstanding|unpaid|balance\s+due|payment\s+pending|not\s+yet\s+paid", text, re.IGNORECASE):
        return False
    return False


def extract_priority(text: str) -> str:
    if re.search(r"\burgent\b|immediate\s+attention|asap", text, re.IGNORECASE):
        return "urgent"
    if re.search(r"high\s+priority|past\s+due|overdue", text, re.IGNORECASE):
        return "high"
    if re.search(r"low\s+priority|no\s+rush|whenever\s+convenient", text, re.IGNORECASE):
        return "low"
    return "normal"


def extract_contact_email(text: str) -> str:
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return m.group(0).lower() if m else ""


# ---------------------------------------------------------------------------
# LLM used ONLY for vendor + line_items (the fields regex can't reliably parse)
# ---------------------------------------------------------------------------

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "emit_structural_fields",
        "description": "Emit only the vendor name and line items from the invoice text.",
        "parameters": {
            "type": "object",
            "properties": {
                "vendor": {
                    "type": "string",
                    "description": "The biller's proper name, exactly as written in the source text."
                },
                "line_items": {
                    "type": "array",
                    "description": "Line items in the exact order they appear in the source text.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "integer"}
                        },
                        "required": ["sku", "quantity", "unit_price"]
                    }
                }
            },
            "required": ["vendor", "line_items"]
        }
    }
}

SYSTEM_PROMPT = """You extract two fields from a messy invoice document by calling
emit_structural_fields exactly once:
- vendor: the biller's proper name, copied exactly as written (preserve capitalization/punctuation).
- line_items: array of {sku, quantity, unit_price} in the exact order they appear in the text.
  unit_price is an integer (no decimals/symbols).
Do not invent data. Call the tool with the complete answer."""


def extract_structural_fields(text: str) -> dict:
    last_error = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract vendor and line_items from this document:\n\n{text}"}
                ],
                tools=[TOOL_DEF],
                tool_choice={"type": "function", "function": {"name": "emit_structural_fields"}},
                temperature=0,
                seed=42,
                timeout=25,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                raise ValueError("Model did not return a tool call")
            raw = json.loads(tool_calls[0].function.arguments)
            if not isinstance(raw, dict):
                raise ValueError("Tool arguments were not a JSON object")
            return raw
        except Exception as e:
            last_error = e
            continue
    raise HTTPException(status_code=502, detail=f"Structural extraction failed: {last_error}")


def _safe_int(value, default=0):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(round(value))
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d\-]", "", value)
        try:
            return int(cleaned)
        except ValueError:
            return default
    return default


def extract_invoice(text: str) -> dict:
    structural = extract_structural_fields(text)

    raw_items = structural.get("line_items", [])
    if not isinstance(raw_items, list):
        raw_items = []
    line_items = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        line_items.append({
            "sku": str(item.get("sku", "")).strip(),
            "quantity": _safe_int(item.get("quantity", 0)),
            "unit_price": _safe_int(item.get("unit_price", 0)),
        })

    return {
        "vendor": str(structural.get("vendor", "")).strip(),
        "currency": extract_currency(text),
        "total_amount": extract_total_amount(text),
        "invoice_date": extract_invoice_date(text),
        "due_in_days": extract_due_in_days(text),
        "is_paid": extract_is_paid(text),
        "priority": extract_priority(text),
        "contact_email": extract_contact_email(text),
        "line_items": line_items,
        "item_count": len(line_items),
    }


@app.post("/")
async def extract(req: ExtractRequest):
    result = extract_invoice(req.text)
    return {k: result[k] for k in EXPECTED_KEYS}


@app.post("/extract")
async def extract_alias(req: ExtractRequest):
    return await extract(req)


@app.get("/")
async def health():
    return {"status": "ok"}
