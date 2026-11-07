"""Input schemas, validation rules, and validators for structured AI output.

Validation is implemented with the standard library (no Pydantic dependency) so it
runs identically in tests and in the app. Every validator either returns a clean
dictionary with guaranteed keys or raises :class:`SchemaError`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

from utils import clean_text

# --------------------------------------------------------------------------- features
FEATURES: tuple[str, ...] = (
    "total_funding_usd", "funding_rounds", "relationships", "milestones",
    "has_vc", "has_angel", "has_series_a", "has_series_b",
)
FEATURE_LABELS = {
    "total_funding_usd": "Total funding (USD)",
    "funding_rounds": "Funding rounds",
    "relationships": "Relationships",
    "milestones": "Milestones",
    "has_vc": "Venture capital backing",
    "has_angel": "Angel investor backing",
    "has_series_a": "Series A raised",
    "has_series_b": "Series B raised",
}
BINARY_FEATURES = ("has_vc", "has_angel", "has_series_a", "has_series_b")
COUNT_LIMITS = {"funding_rounds": 50, "relationships": 10_000, "milestones": 1_000}
MAX_FUNDING_USD = 100_000_000_000.0
UNIT_MULTIPLIERS = {"USD": 1.0, "USD thousands": 1e3, "USD millions": 1e6}
TARGET_COLUMN = "success"

STAGES = ("Idea", "Prototype", "Early launch", "Early revenue")

LABELS = ("source_supported", "ai_interpretation", "assumption", "unknown", "rule_based")
LABEL_DISPLAY = {
    "source_supported": "Source-supported",
    "ai_interpretation": "AI interpretation",
    "assumption": "Assumption",
    "unknown": "Unknown",
    "rule_based": "Rule-based (offline)",
}
LABEL_SYMBOL = {"source_supported": "◆", "ai_interpretation": "◇", "assumption": "△",
                "unknown": "?", "rule_based": "▢"}


class SchemaError(ValueError):
    """Raised when data does not satisfy a schema."""


# --------------------------------------------------------------------------- helpers
def _to_number(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("_", "")
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "y", "true", "1"):
            return True
        if s in ("no", "n", "false", "0"):
            return False
    return None


@dataclass
class ValidationResult:
    value: Any = None
    errors: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.value is not None


# --------------------------------------------------------------------------- predict input
@dataclass(frozen=True)
class PredictInput:
    total_funding_usd: float
    funding_rounds: int
    relationships: int
    milestones: int
    has_vc: bool
    has_angel: bool
    has_series_a: bool
    has_series_b: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_row(self) -> list[float]:
        """Values in the exact order the model expects."""
        return [float(getattr(self, f)) for f in FEATURES]


def validate_predict_input(raw: Mapping[str, Any], funding_unit: str = "USD") -> ValidationResult:
    errors: dict[str, str] = {}
    warnings: list[str] = []
    mult = UNIT_MULTIPLIERS.get(funding_unit)
    if mult is None:
        errors["total_funding_usd"] = f"Unknown funding unit '{funding_unit}'."
        mult = 1.0

    funding = _to_number(raw.get("total_funding_usd"))
    if funding is None:
        errors.setdefault("total_funding_usd", "Enter total funding as a number.")
    else:
        funding *= mult
        if funding < 0:
            errors["total_funding_usd"] = "Funding cannot be negative."
        elif funding > MAX_FUNDING_USD:
            errors["total_funding_usd"] = "Funding exceeds the supported maximum of USD 100 billion."

    counts: dict[str, int] = {}
    for name, hi in COUNT_LIMITS.items():
        v = _to_number(raw.get(name))
        if v is None or not float(v).is_integer():
            errors[name] = "Enter a whole number."
        elif v < 0:
            errors[name] = "Cannot be negative."
        elif v > hi:
            errors[name] = f"Must be at most {hi:,}."
        else:
            counts[name] = int(v)

    flags: dict[str, bool] = {}
    for name in BINARY_FEATURES:
        b = _to_bool(raw.get(name))
        if b is None:
            errors[name] = "Choose yes or no."
        else:
            flags[name] = b

    if not errors:
        rounds = counts["funding_rounds"]
        if funding and funding > 0 and rounds == 0:
            errors["funding_rounds"] = "Funding is recorded but funding rounds is 0. Enter at least 1 round."
        if flags["has_series_b"] and not flags["has_series_a"]:
            errors["has_series_b"] = "Series B requires Series A. Mark Series A as raised or clear Series B."
        if flags["has_series_a"] and rounds < 1:
            errors["has_series_a"] = "Series A implies at least 1 funding round."
        if flags["has_series_a"] and flags["has_series_b"] and rounds < 2:
            errors["funding_rounds"] = "Series A and Series B together imply at least 2 funding rounds."
        if (flags["has_vc"] or flags["has_angel"]) and rounds == 0:
            errors["funding_rounds"] = "Investor backing implies at least 1 funding round."
        if not errors:
            if rounds > 0 and funding == 0:
                warnings.append("Funding rounds are recorded but total funding is 0; confirm the amount is undisclosed rather than zero.")
            if flags["has_series_a"] and not flags["has_vc"]:
                warnings.append("Series A without venture-capital backing is uncommon; check the inputs.")

    if errors:
        return ValidationResult(None, errors, warnings)
    value = PredictInput(float(funding), counts["funding_rounds"], counts["relationships"],
                         counts["milestones"], **flags)
    return ValidationResult(value, {}, warnings)


# --------------------------------------------------------------------------- explore input
@dataclass(frozen=True)
class ExploreInput:
    idea: str
    industry: str
    target_customer: str
    geography: str
    business_model: str
    problem: str
    stage: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


_EXPLORE_RULES = {
    "idea": (30, 3000, "Describe the idea in at least 30 characters."),
    "industry": (2, 120, "Enter an industry or category."),
    "target_customer": (3, 300, "Describe who the customer is."),
    "geography": (2, 200, "Enter a target city, country or region."),
    "business_model": (2, 200, "Enter a business model, e.g. B2B subscription."),
    "problem": (20, 2000, "Describe the problem in at least 20 characters."),
}


def validate_explore_input(raw: Mapping[str, Any]) -> ValidationResult:
    errors: dict[str, str] = {}
    vals: dict[str, str] = {}
    for name, (lo, hi, msg) in _EXPLORE_RULES.items():
        v = raw.get(name)
        s = clean_text(v) if isinstance(v, str) else ""
        if len(s) < lo:
            errors[name] = msg
        elif len(s) > hi:
            errors[name] = f"Too long ({len(s):,} characters); the limit is {hi:,}."
        else:
            vals[name] = s
    stage = raw.get("stage")
    if stage not in STAGES:
        errors["stage"] = "Choose a stage from the list."
    if errors:
        return ValidationResult(None, errors, [])
    return ValidationResult(ExploreInput(stage=stage, **vals), {}, [])


# --------------------------------------------------------------------------- AI output validation
@dataclass
class ClaimContext:
    """Collects validation notes while normalising AI output."""
    valid_ids: set[str]
    notes: list[str] = field(default_factory=list)


def _text(v: Any, name: str, max_len: int = 1500, required: bool = False) -> str:
    if v is None:
        v = ""
    if not isinstance(v, (str, int, float)) or isinstance(v, bool):
        raise SchemaError(f"'{name}' must be text.")
    s = clean_text(str(v))[:max_len]
    if required and not s:
        raise SchemaError(f"'{name}' is required and was empty.")
    return s


def _label(v: Any) -> str:
    s = str(v or "ai_interpretation").strip().lower().replace("-", "_").replace(" ", "_")
    # The model may not claim to be offline rule-based output.
    return s if s in LABELS and s != "rule_based" else "ai_interpretation"


def _ids(item: dict, name: str, ctx: ClaimContext) -> list[str]:
    ids = item.get("source_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    if not isinstance(ids, list):
        raise SchemaError(f"'{name}.source_ids' must be a list.")
    ids = [str(i).strip() for i in ids]
    bad = [i for i in ids if i not in ctx.valid_ids]
    if bad:
        ctx.notes.append(f"Removed unknown source id(s) {bad} from {name}.")
    return [i for i in ids if i in ctx.valid_ids]


def norm_claim(item: Any, name: str, ctx: ClaimContext) -> dict[str, Any]:
    if isinstance(item, str):
        item = {"text": item}
    if not isinstance(item, dict):
        raise SchemaError(f"'{name}' items must be text or objects.")
    text = _text(item.get("text") or item.get("claim"), f"{name}.text", required=True)
    label = _label(item.get("label"))
    ids = _ids(item, name, ctx)
    downgraded = False
    if label == "source_supported" and not ids:
        label, downgraded = "ai_interpretation", True
        ctx.notes.append(f"Claim in {name} was marked source-supported without a valid source; shown as AI interpretation.")
    return {"text": text, "label": label, "source_ids": ids, "downgraded": downgraded}


def norm_claims(items: Any, name: str, ctx: ClaimContext, required: bool = False,
                max_items: int = 12) -> list[dict[str, Any]]:
    if items is None:
        items = []
    if not isinstance(items, list):
        raise SchemaError(f"'{name}' must be a list.")
    out = [norm_claim(i, f"{name}[{n}]", ctx) for n, i in enumerate(items[:max_items])]
    if required and not out:
        raise SchemaError(f"'{name}' must contain at least one item.")
    return out


def _str_list(items: Any, name: str, max_items: int = 10, max_len: int = 400) -> list[str]:
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        raise SchemaError(f"'{name}' must be a list of text.")
    return [s for s in (_text(i, name, max_len) for i in items[:max_items]) if s]


def _swot(data: Any, ctx: ClaimContext) -> dict[str, list[dict]]:
    if not isinstance(data, dict):
        raise SchemaError("'swot' must be an object with strengths, weaknesses, opportunities, threats.")
    return {k: norm_claims(data.get(k), f"swot.{k}", ctx, required=True, max_items=6)
            for k in ("strengths", "weaknesses", "opportunities", "threats")}


def _require_dict(data: Any) -> dict:
    if not isinstance(data, dict):
        raise SchemaError("Top-level JSON value must be an object.")
    return data


def validate_predict_report(data: Any, valid_ids: set[str]) -> dict[str, Any]:
    data = _require_dict(data)
    ctx = ClaimContext(set(valid_ids))
    recs_raw = data.get("recommendations")
    if not isinstance(recs_raw, list) or not recs_raw:
        raise SchemaError("'recommendations' must be a non-empty list.")
    recs = []
    for n, r in enumerate(recs_raw[:8]):
        if not isinstance(r, dict):
            raise SchemaError(f"recommendations[{n}] must be an object.")
        pr = _to_number(r.get("priority"))
        recs.append({
            "priority": int(min(5, max(1, pr))) if pr is not None else 3,
            "title": _text(r.get("title"), f"recommendations[{n}].title", 200, True),
            "rationale": _text(r.get("rationale"), f"recommendations[{n}].rationale", 800),
            "actions": _str_list(r.get("actions"), f"recommendations[{n}].actions", 6),
            "label": _label(r.get("label")),
            "source_ids": _ids(r, f"recommendations[{n}]", ctx),
        })
        if recs[-1]["label"] == "source_supported" and not recs[-1]["source_ids"]:
            recs[-1]["label"] = "ai_interpretation"
            ctx.notes.append(f"Recommendation {n + 1} lacked a valid source; shown as AI interpretation.")
    recs.sort(key=lambda r: r["priority"])
    return {
        "summary": _text(data.get("summary"), "summary", 1200, True),
        "swot": _swot(data.get("swot"), ctx),
        "recommendations": recs,
        "unknowns": _str_list(data.get("unknowns"), "unknowns"),
        "validation_notes": ctx.notes,
    }


def _figure(v: Any) -> float | None:
    n = _to_number(v)
    return n if n is not None and n >= 0 else None


def _sizing_level(level: str, item: Any, ctx: ClaimContext) -> dict[str, Any]:
    if not isinstance(item, dict):
        ctx.notes.append(f"{level} was not provided by the model; marked unknown.")
        item = {}
    lo, hi = _figure(item.get("value_low")), _figure(item.get("value_high"))
    method = _text(item.get("method"), f"{level}.method", 800)
    label = _label(item.get("label"))
    ids = _ids(item, level, ctx)
    if lo is None and hi is not None:
        lo = hi
    if hi is None and lo is not None:
        hi = lo
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    if lo is not None and not method:
        lo = hi = None
        ctx.notes.append(f"{level}: dropped a numeric figure because no calculation method was given.")
    if lo is not None:
        if label == "source_supported" and not ids:
            ctx.notes.append(f"{level}: figure was marked source-supported without a valid source; shown as an assumption.")
        if label != "source_supported" or not ids:
            label = "assumption"
    elif label == "source_supported":
        label = "unknown"
    return {
        "level": level,
        "definition": _text(item.get("definition"), f"{level}.definition", 500),
        "value_low": lo, "value_high": hi,
        "currency": _text(item.get("currency"), f"{level}.currency", 12) or None,
        "unit": _text(item.get("unit"), f"{level}.unit", 60) or None,
        "method": method,
        "assumptions": _str_list(item.get("assumptions"), f"{level}.assumptions", 6),
        "label": label if lo is not None else (label if label in ("unknown", "assumption") else "unknown"),
        "source_ids": ids,
    }


def validate_explore_overview(data: Any, valid_ids: set[str]) -> dict[str, Any]:
    data = _require_dict(data)
    ctx = ClaimContext(set(valid_ids))
    ps = data.get("problem_solution")
    if not isinstance(ps, dict):
        raise SchemaError("'problem_solution' must be an object.")
    customers = data.get("target_customers")
    if not isinstance(customers, list) or not customers:
        raise SchemaError("'target_customers' must be a non-empty list.")
    cust = []
    for n, c in enumerate(customers[:5]):
        if not isinstance(c, dict):
            raise SchemaError(f"target_customers[{n}] must be an object.")
        cust.append({
            "segment": _text(c.get("segment"), "segment", 200, True),
            "needs": _text(c.get("needs"), "needs", 600),
            "buyer_persona": _text(c.get("buyer_persona"), "buyer_persona", 600),
            "label": _label(c.get("label")),
            "source_ids": _ids(c, f"target_customers[{n}]", ctx),
        })
    sizing_raw = data.get("market_sizing")
    if not isinstance(sizing_raw, dict):
        raise SchemaError("'market_sizing' must be an object with TAM, SAM and SOM.")
    sizing = [_sizing_level(lv, sizing_raw.get(lv) or sizing_raw.get(lv.lower()), ctx)
              for lv in ("TAM", "SAM", "SOM")]
    comps = []
    comp_raw = data.get("competitors") or []
    if not isinstance(comp_raw, list):
        raise SchemaError("'competitors' must be a list.")
    for n, c in enumerate(comp_raw[:8]):
        if not isinstance(c, dict):
            raise SchemaError(f"competitors[{n}] must be an object.")
        ctype = str(c.get("type", "potential")).strip().lower()
        ctype = ctype if ctype in ("confirmed", "potential", "illustrative") else "potential"
        ids = _ids(c, f"competitors[{n}]", ctx)
        if ctype == "confirmed" and not ids:
            ctype = "potential"
            ctx.notes.append(f"Competitor '{c.get('name')}' cannot be confirmed without retrieved evidence; shown as potential, unverified.")
        comps.append({"name": _text(c.get("name"), "competitor.name", 120, True), "type": ctype,
                      "note": _text(c.get("note"), "competitor.note", 500), "source_ids": ids})
    return {
        "executive_summary": _text(data.get("executive_summary"), "executive_summary", 1500, True),
        "problem_solution": {k: _text(ps.get(k), f"problem_solution.{k}", 800, k != "value_proposition")
                             for k in ("problem", "solution", "value_proposition")},
        "target_customers": cust,
        "market_sizing": sizing,
        "competitors": comps,
        "differentiation": norm_claims(data.get("differentiation"), "differentiation", ctx, required=True, max_items=6),
        "unknowns": _str_list(data.get("unknowns"), "unknowns"),
        "validation_notes": ctx.notes,
    }


RISK_CATEGORIES = ("market", "product", "financial", "operational", "regulatory")


def validate_explore_strategy(data: Any, valid_ids: set[str]) -> dict[str, Any]:
    data = _require_dict(data)
    ctx = ClaimContext(set(valid_ids))
    bm = data.get("business_model")
    if not isinstance(bm, dict):
        raise SchemaError("'business_model' must be an object.")
    risks = data.get("risks")
    if not isinstance(risks, dict):
        raise SchemaError("'risks' must be an object keyed by category.")
    risk_out = {k: norm_claims(risks.get(k), f"risks.{k}", ctx, max_items=4) for k in RISK_CATEGORIES}
    if sum(len(v) for v in risk_out.values()) < 3:
        raise SchemaError("'risks' must include at least 3 items across categories.")
    road = data.get("roadmap")
    if not isinstance(road, list) or not road:
        raise SchemaError("'roadmap' must be a non-empty list.")
    roadmap = []
    for n, r in enumerate(road[:6]):
        if not isinstance(r, dict):
            raise SchemaError(f"roadmap[{n}] must be an object.")
        acts = _str_list(r.get("actions"), f"roadmap[{n}].actions", 6)
        if not acts:
            raise SchemaError(f"roadmap[{n}].actions must not be empty.")
        roadmap.append({"phase": _text(r.get("phase"), "roadmap.phase", 80, True), "actions": acts})
    exps_raw = data.get("validation_experiments")
    if not isinstance(exps_raw, list) or not exps_raw:
        raise SchemaError("'validation_experiments' must be a non-empty list.")
    exps = []
    for n, e in enumerate(exps_raw[:6]):
        if not isinstance(e, dict):
            raise SchemaError(f"validation_experiments[{n}] must be an object.")
        exps.append({k: _text(e.get(k), f"validation_experiments.{k}", 500, True)
                     for k in ("hypothesis", "experiment", "success_metric")})
    fr = data.get("funding_readiness")
    if not isinstance(fr, dict):
        raise SchemaError("'funding_readiness' must be an object.")
    level = str(fr.get("level", "early")).strip().lower()
    return {
        "business_model": {
            "revenue_streams": norm_claims(bm.get("revenue_streams"), "revenue_streams", ctx, required=True, max_items=5),
            "pricing_hypotheses": norm_claims(bm.get("pricing_hypotheses"), "pricing_hypotheses", ctx, max_items=5),
            "cost_drivers": norm_claims(bm.get("cost_drivers"), "cost_drivers", ctx, max_items=5),
        },
        "swot": _swot(data.get("swot"), ctx),
        "go_to_market": norm_claims(data.get("go_to_market"), "go_to_market", ctx, required=True, max_items=6),
        "risks": risk_out,
        "roadmap": roadmap,
        "validation_experiments": exps,
        "funding_readiness": {
            "level": level if level in ("early", "developing", "approaching") else "early",
            "indicators": norm_claims(fr.get("indicators"), "funding_readiness.indicators", ctx, max_items=6),
            "gaps": _str_list(fr.get("gaps"), "funding_readiness.gaps", 8),
            "milestones": _str_list(fr.get("milestones"), "funding_readiness.milestones", 8),
        },
        "unknowns": _str_list(data.get("unknowns"), "unknowns"),
        "validation_notes": ctx.notes,
    }
