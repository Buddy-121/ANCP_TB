"""AI Strategic Advisory engine (Anthropic Claude API).

Claude receives ONLY the Python-calculated evidence pack and must cite it. It is
instructed not to compute new financial figures, and every statement is tagged as
FACT (from the pack), FORECAST (model output in the pack) or OPINION (AI judgement).
"""
from __future__ import annotations

import json
import re

from .config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

AREAS = ["Pricing strategy", "Procurement strategy", "Supply chain optimisation", "Raw material cost management",
         "Foreign exchange risk", "Profitability improvement", "Working capital management", "Capital allocation",
         "Competitive positioning"]

SYSTEM_PROMPT = """You are a corporate strategy adviser and equity research analyst preparing recommendations
for the management of Asian Paints Limited (India's largest decorative paints company).

STRICT EVIDENCE RULES
1. Use ONLY numbers present in the EVIDENCE PACK. Never invent, estimate or recompute financial figures.
   If a figure is needed but absent, say "not available in the dataset".
2. Tag every piece of supporting data with its type:
   FACT = calculated historical value from the pack; FORECAST = statistical forecast from the pack;
   SCENARIO = assumption-based scenario output from the pack.
   Your own reasoning and recommendations are OPINION.
3. Correlation is not causation. Small samples (n < 20) and p-values > 0.05 must lower confidence.
4. Ratios in the pack are decimals (0.21 = 21%). Amounts are ₹ Crore.
5. Confidence: High = multiple consistent, statistically supported facts; Medium = consistent but limited
   data; Low = weak/indirect evidence or mostly judgement.
6. Expected financial impact: only quantify when it can be derived directly from a SCENARIO or FACT in the
   pack (cite it); otherwise state "Not quantifiable from available data".

OUTPUT: return ONLY valid JSON (no markdown fences) with this structure:
{
  "executive_summary": "3-5 sentences (OPINION grounded in facts)",
  "key_findings": [{"finding": "...", "evidence_type": "FACT|FORECAST|SCENARIO", "source": "pack path e.g. financials.history"}],
  "recommendations": [
    {"area": "one of the requested areas",
     "title": "short action title",
     "business_problem": "...",
     "supporting_data": [{"metric": "...", "value": "...", "evidence_type": "FACT|FORECAST|SCENARIO", "source": "pack path"}],
     "analytical_interpretation": "...",
     "recommended_action": "...",
     "expected_financial_impact": "...",
     "implementation_timeframe": "e.g. 0-3 months / 3-12 months / 12-36 months",
     "risks_and_limitations": "...",
     "confidence": "High|Medium|Low",
     "confidence_rationale": "..."}
  ],
  "data_gaps": ["data that would materially improve the analysis"]
}"""


def get_key(override: str | None = None) -> str:
    return (override or ANTHROPIC_API_KEY or "").strip()


def _extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def generate_advisory(context: dict, areas: list[str], api_key: str, model: str | None = None,
                      extra_instructions: str = "", max_tokens: int = 12000) -> dict:
    import anthropic

    model = model or ANTHROPIC_MODEL
    client = anthropic.Anthropic(api_key=api_key)
    user = (f"Requested recommendation areas: {', '.join(areas)}.\n"
            f"Produce one recommendation per area (more only if strongly supported).\n"
            f"{('Additional management focus: ' + extra_instructions) if extra_instructions else ''}\n\n"
            f"EVIDENCE PACK (JSON):\n{json.dumps(context, ensure_ascii=False, default=str)}")
    resp = client.messages.create(model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
                                  messages=[{"role": "user", "content": user}])
    text = "".join(getattr(b, "text", "") for b in resp.content)
    try:
        data = _extract_json(text)
    except json.JSONDecodeError:
        data = {"executive_summary": "", "recommendations": [], "parse_error": True, "raw_text": text}
    data["_meta"] = {"model": model, "stop_reason": getattr(resp, "stop_reason", None),
                     "input_tokens": getattr(resp.usage, "input_tokens", None),
                     "output_tokens": getattr(resp.usage, "output_tokens", None)}
    return data


def ask_followup(context: dict, advisory: dict, question: str, api_key: str, model: str | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    sys = SYSTEM_PROMPT.split("OUTPUT:")[0] + "\nAnswer in concise prose. Label numbers as FACT/FORECAST/SCENARIO and your judgement as OPINION."
    msg = (f"EVIDENCE PACK:\n{json.dumps(context, default=str)}\n\nPREVIOUS ADVISORY:\n"
           f"{json.dumps(advisory, default=str)}\n\nQUESTION: {question}")
    resp = client.messages.create(model=model or ANTHROPIC_MODEL, max_tokens=3000, system=sys,
                                  messages=[{"role": "user", "content": msg}])
    return "".join(getattr(b, "text", "") for b in resp.content)
