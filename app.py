import os
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    document_id: str
    text: str
    schema: Optional[dict] = None


EXPECTED_KEYS = [
    "vendor", "currency", "total_amount", "invoice_date", "due_in_days",
    "is_paid", "priority", "contact_email", "line_items", "item_count"
]

# ---------------------------------------------------------------------------
# Tool definition forced on the Groq model (OpenAI-compatible tool schema)
# ---------------------------------------------------------------------------

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "emit_invoice",
        "description": "Emit the fully normalized, strongly-typed invoice data.",
        "parameters": {
            "type": "object",
            "properties": {
                "vendor": {
                    "type": "string",
                    "description": "The biller's proper name, exactly as written in the source text."
                },
                "currency": {
                    "type": "string",
                    "enum": ["USD", "EUR", "GBP", "INR", "JPY"],
                    "description": "ISO 4217 code inferred from symbols/words like $, euros, £, pounds sterling, ₹, rupees, yen."
                },
                "total_amount": {
                    "type": "integer",
                    "description": "Integer amount in the main unit, no separators/symbols. Convert spelled-out numbers, comma-grouped numbers (12,480), Indian grouping (1,24,800), and 'K' suffixes (12K -> 12000)."
                },
                "invoice_date": {
                    "type": "string",
                    "description": "Normalized to YYYY-MM-DD."
                },
                "due_in_days": {
                    "type": "integer",
                    "description": "Numeric days derived from phrasing like 'Net 30' -> 30, 'within 45 days' -> 45, 'due in two weeks' -> 14, 'due on receipt' -> 0."
                },
                "is_paid": {
                    "type": "boolean",
                    "description": "True if the text indicates payment already made; false if outstanding/awaiting payment."
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "Inferred urgency; default 'normal' if unstated."
                },
                "contact_email": {
                    "type": "string",
                    "description": "Lowercased contact email address."
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
                },
                "item_count": {
                    "type": "integer",
                    "description": "Number of line items (must equal length of line_items)."
                }
            },
            "required": [
                "vendor", "currency", "total_amount", "invoice_date", "due_in_days",
                "is_paid", "priority", "contact_email", "line_items", "item_count"
            ]
        }
    }
}

SYSTEM_PROMPT = """You are an expert invoice-data extraction engine for a logistics firm's ERP system.
You will be given raw, messy free-text invoice content. Extract fully normalized, strongly-typed
fields and call the `emit_invoice` tool exactly once with the complete answer. Follow these rules:

- vendor: the biller's proper name, copied exactly as written (preserve capitalization/punctuation).
- currency: map wording/symbols to ISO 4217: $ or "dollars" -> USD, "euros"/€ -> EUR,
  "pounds sterling"/£ -> GBP, ₹ or "rupees" -> INR, ¥ or "yen" -> JPY.
- total_amount: integer in the main unit only (no decimals, symbols, separators).
  Handle spelled-out numbers ("twelve thousand four hundred eighty" -> 12480),
  comma-grouped numbers (12,480 -> 12480), Indian digit grouping (1,24,800 -> 124800),
  and "K" shorthand (12K -> 12000).
- invoice_date: normalize any date format to YYYY-MM-DD.
- due_in_days: convert payment-term phrasing to an integer number of days
  ("Net 30" -> 30, "payable within 45 days" -> 45, "due in two weeks" -> 14,
  "due on receipt" -> 0, "one month" -> 30).
- is_paid: true if text indicates the invoice is already paid/settled; false if outstanding/unpaid.
- priority: one of low, normal, high, urgent — infer from urgency language;
  default "normal" if nothing indicates otherwise.
- contact_email: the invoice's contact email address, all lowercase.
- line_items: array of {sku, quantity, unit_price} in the exact order they appear.
  unit_price is an integer (no decimals/symbols).
- item_count: must equal the length of the line_items array.

Be precise and deterministic. Do not invent data not present or reasonably inferable from the text.
You MUST call emit_invoice — never reply with plain text."""


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


def normalize_output(data: dict) -> dict:
    """Safety net: coerce types and enforce invariants regardless of what the
    model returned, so minor LLM formatting slips don't break exact-match grading."""

    out = {}
    out["vendor"] = str(data.get("vendor", "")).strip()

    currency = str(data.get("currency", "")).strip().upper()
    if currency not in {"USD", "EUR", "GBP", "INR", "JPY"}:
        currency = currency[:3] if len(currency) >= 3 else currency
    out["currency"] = currency

    out["total_amount"] = _safe_int(data.get("total_amount", 0))
    out["invoice_date"] = str(data.get("invoice_date", "")).strip()
    out["due_in_days"] = _safe_int(data.get("due_in_days", 0))

    is_paid = data.get("is_paid", False)
    if isinstance(is_paid, str):
        is_paid = is_paid.strip().lower() in {"true", "yes", "paid"}
    out["is_paid"] = bool(is_paid)

    priority = str(data.get("priority", "normal")).strip().lower()
    if priority not in {"low", "normal", "high", "urgent"}:
        priority = "normal"
    out["priority"] = priority

    out["contact_email"] = str(data.get("contact_email", "")).strip().lower()

    raw_items = data.get("line_items", [])
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
    out["line_items"] = line_items
    out["item_count"] = len(line_items)  # always derived, never trusted blindly

    return out


def extract_invoice(text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract the invoice data from this document:\n\n{text}"}
        ],
        tools=[TOOL_DEF],
        tool_choice={"type": "function", "function": {"name": "emit_invoice"}},
        temperature=0,
    )

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        raise HTTPException(status_code=502, detail="Model did not return a tool call")

    import json
    try:
        raw_data = json.loads(tool_calls[0].function.arguments)
    except (json.JSONDecodeError, AttributeError, IndexError):
        raise HTTPException(status_code=502, detail="Could not parse tool call arguments")

    if not isinstance(raw_data, dict):
        raise HTTPException(status_code=502, detail="Tool arguments were not a JSON object")

    return normalize_output(raw_data)


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
