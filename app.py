import os
import re
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
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
# Tool schema — LLM does ALL semantic interpretation
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
                    "description": "Integer in the main unit, no separators/symbols/decimals. Convert spelled-out numbers ('twelve thousand four hundred eighty' -> 12480), comma-grouped (12,480), Indian grouping (1,24,800 -> 124800), and 'K' suffix (12K -> 12000)."
                },
                "invoice_date": {
                    "type": "string",
                    "description": "Normalized to YYYY-MM-DD."
                },
                "due_in_days": {
                    "type": "integer",
                    "description": "The number of days until payment is due, as a plain integer, however it is phrased in the text: 'Net 30' -> 30, 'payable within 45 days' -> 45, 'due in two weeks' -> 14, 'due on receipt' -> 0, '15-day terms' -> 15, 'payment terms: 15 days' -> 15, 'one month' -> 30, a fortnight -> 14. Read the whole document for any payment-terms phrasing, not just one fixed pattern."
                },
                "is_paid": {
                    "type": "boolean",
                    "description": "True if wording indicates payment already made ('paid in full', 'payment received'); false if outstanding/awaiting/unpaid."
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "Inferred urgency from tone/wording; default 'normal' if nothing indicates otherwise."
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
                    "description": "Number of line items."
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
You will be given raw, messy free-text invoice content. Read the ENTIRE document carefully — payment
terms, dates, and amounts can be phrased in many different ways and may not follow a fixed template.
Extract fully normalized, strongly-typed fields and call the `emit_invoice` tool exactly once with the
complete, final answer. Follow these rules:

- vendor: the biller's proper name, copied exactly as written (preserve capitalization/punctuation).
- currency: map any wording/symbol to ISO 4217: $ or "dollars" -> USD, "euros"/€ -> EUR,
  "pounds sterling"/£ -> GBP, ₹ or "rupees" -> INR, ¥ or "yen" -> JPY.
- total_amount: integer in the main unit only (no decimals, symbols, separators). Handle spelled-out
  numbers, comma-grouped numbers (12,480), Indian digit grouping (1,24,800 -> 124800), and "K" shorthand
  (12K -> 12000).
- invoice_date: normalize any date format to YYYY-MM-DD.
- due_in_days: find whatever phrase describes payment terms anywhere in the document (it may be
  "Net 30", "payable within 45 days", "due in two weeks", "due on receipt", "15-day terms",
  "payment terms: 15 days", spelled-out durations, etc.) and convert it to a plain integer number
  of days. Do not default to 0 unless the text truly says something like "due on receipt" or gives
  no payment-terms information at all.
- is_paid: true if text indicates the invoice is already paid/settled; false if outstanding/unpaid.
- priority: one of low, normal, high, urgent — infer from urgency language; default "normal" if
  nothing indicates otherwise.
- contact_email: the invoice's contact email address, all lowercase.
- line_items: array of {sku, quantity, unit_price} in the exact order they appear. unit_price is an
  integer (no decimals/symbols).
- item_count: must equal the length of the line_items array.

Be precise and deterministic — given the same document, always produce the same answer.
Do not invent data not present or reasonably inferable from the text.
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
    """Type-safety net only — never re-derives semantics, just casts/cleans
    whatever the LLM already correctly interpreted."""

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
    last_error = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract the invoice data from this document:\n\n{text}"}
                ],
                tools=[TOOL_DEF],
                tool_choice={"type": "function", "function": {"name": "emit_invoice"}},
                temperature=0,
                seed=42,
                timeout=25,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                raise ValueError("Model did not return a tool call")
            raw_data = json.loads(tool_calls[0].function.arguments)
            if not isinstance(raw_data, dict):
                raise ValueError("Tool arguments were not a JSON object")
            return normalize_output(raw_data)
        except Exception as e:
            last_error = e
            continue

    raise HTTPException(status_code=502, detail=f"Extraction failed after retries: {last_error}")


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
