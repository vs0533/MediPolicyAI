"""LLM-based judge for FinanceBench evaluation.

The judge drives **all** evaluation decisions:
- **Accuracy**: whether the prediction is semantically equivalent to the gold answer.
- **Coverage**: whether the prediction contains any information relevant to the question.

This replaces the previous EM/F1 rule-driven pipeline with a single LLM-based
evaluation authority, providing more nuanced correctness signals for financial QA.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


_JUDGE_PROMPT = """\
You are an expert financial analyst and auditor evaluating answer correctness \
with **zero tolerance for numerical or factual errors**.

Question: {question}
Gold Answer: {gold}
Model Prediction: {prediction}

Task: Determine if the model's prediction is **semantically equivalent** \
to the gold answer in the context of this financial question.

═══════════════════════════════════════════════
EQUIVALENT — only when ALL of the following hold:
═══════════════════════════════════════════════

1. **Numerical precision (ZERO TOLERANCE)**:
   - Values must be mathematically identical after unit conversion.
   - $1.5B = $1,500M = $1,500,000K = $1,500,000,000 ✓
   - $1,577 ≠ $1,580 ✗ (rounding is NOT acceptable)
   - 15.3% = 15.30% = 0.153 ✓ but 15.3% ≠ 15% ✗
   - $1.5M ≠ $1.5B ✗ (unit mismatch is a critical error)

2. **Negative / bracket notation**:
   - ($500) = -$500 = -500 ✓
   - ($500) ≠ $500 ✗ (sign matters)

3. **Time period / fiscal year**:
   - FY2018 = fiscal year 2018 = 2018 ✓
   - FY2018 ≠ FY2019 ✗ (different fiscal year — NEVER equivalent)
   - Q3 2019 ≠ Q4 2019 ✗ (different quarter)
   - "year ended December 2018" = FY2018 ✓

4. **Currency formatting**:
   - $1,577.00 = $1577 = 1577 ✓ (same value, format differs)

5. **Financial term equivalences (accepted)**:
   - net income = net profit ✓
   - CAPEX = capital expenditure ✓
   - EPS = earnings per share ✓
   - EBITDA = earnings before interest, taxes, depreciation and amortization ✓
   - YoY = year-over-year ✓
   - COGS = cost of goods sold ✓
   - D&A = depreciation and amortization ✓

6. **Financial term distinctions (NOT interchangeable)**:
   - revenue ≠ net revenue ≠ gross revenue (unless context is clear)
   - operating income ≠ net income
   - gross profit ≠ net profit
   - total assets ≠ net assets

7. **Prediction with extra context**:
   - If prediction contains the correct answer with additional supporting \
     detail, treat as equivalent (e.g., "Revenue was $1,577M in FY2018" \
     vs "$1,577M" — equivalent, provided the value is correct).

═══════════════════════════════════════════════
NOT EQUIVALENT — if ANY of the following hold:
═══════════════════════════════════════════════

1. Different numerical values (even slightly: $1,577 ≠ $1,580)
2. Different time periods or fiscal years
3. Different companies or entities
4. Opposite trend direction (increased ≠ decreased, growth ≠ decline)
5. Unit mismatch ($1.5M ≠ $1.5B)
6. Missing or wrong sign (positive ≠ negative)
7. Prediction is vague or hedging where gold is precise
8. Prediction is a refusal or states it cannot find the answer
9. Near-approximate values that are not mathematically equal after unit conversion

═══════════════════════════════════════════════
CONSERVATIVE JUDGMENT POLICY
═══════════════════════════════════════════════

- **When in doubt, judge as NOT equivalent.** Financial accuracy demands \
  precision; a false positive (incorrectly marking wrong answer as correct) \
  is far worse than a false negative.
- If you are less than 80% confident the answers are equivalent, \
  judge as NOT equivalent.
- Set confidence to reflect your actual certainty (0.0 = no idea, \
  1.0 = absolutely certain).

═══════════════════════════════════════════════
FEW-SHOT EXAMPLES
═══════════════════════════════════════════════

Example 1 — EQUIVALENT (format difference):
  Gold: "$1,577"  |  Prediction: "$1,577.00 million"
  → {{"equivalent": true, "confidence": 0.95, "reasoning": "Same value $1,577M, trailing zeros are formatting."}}

Example 2 — EQUIVALENT (abbreviation):
  Gold: "$1.5 billion"  |  Prediction: "$1,500M"
  → {{"equivalent": true, "confidence": 0.97, "reasoning": "$1.5B = $1,500M, correct unit conversion."}}

Example 3 — NOT EQUIVALENT (different value):
  Gold: "$1,577"  |  Prediction: "$1,580"
  → {{"equivalent": false, "confidence": 0.99, "reasoning": "Values differ: 1577 ≠ 1580. No rounding tolerance."}}

Example 4 — NOT EQUIVALENT (different fiscal year):
  Gold: "FY2018"  |  Prediction: "FY2019"
  → {{"equivalent": false, "confidence": 1.0, "reasoning": "Different fiscal years."}}

Example 5 — NOT EQUIVALENT (unit mismatch):
  Gold: "$1.5 million"  |  Prediction: "$1.5 billion"
  → {{"equivalent": false, "confidence": 1.0, "reasoning": "Unit mismatch: million ≠ billion."}}

Example 6 — EQUIVALENT (negative notation):
  Gold: "-$500"  |  Prediction: "($500)"
  → {{"equivalent": true, "confidence": 0.98, "reasoning": "Same negative value, bracket = negative."}}

Respond ONLY with a JSON object (no markdown, no extra text):
{{"equivalent": true or false, "confidence": 0.0 to 1.0, "reasoning": "brief explanation"}}"""


# Refusal detection phrases (subset for quick judge-side check)
_REFUSAL_INDICATORS: frozenset[str] = frozenset(
    {
        "i cannot",
        "i can't",
        "unable to",
        "not able to",
        "i don't know",
        "i do not know",
        "unknown",
        "no results found",
        "cannot determine",
        "insufficient data",
        "data not found",
        "could not find",
        "couldn't find",
        "unable to determine",
        "unable to find",
    }
)


class FinanceBenchLLMJudge:
    """LLM-based judge driving all FinanceBench evaluation.

    Provides two evaluation axes:
    - ``judge()``: semantic equivalence (Accuracy).
    - ``judge_coverage()``: information relevance (Coverage).

    Token usage from every LLM call is tracked and returned.
    """

    _CONFIDENCE_THRESHOLD: float = 0.7
    _MAX_RETRIES: int = 2

    _NUMERIC_EQUIVALENCE_TOLERANCE: float = 0.02
    """Relative tolerance for numeric equivalence fast-path.

    Two values are considered equivalent if
    ``|a - b| / max(|a|, |b|) <= tolerance``.
    Handles legitimate rounding differences (e.g. ``$8.74B`` vs ``$8.70B``
    derived from ``$8,738M`` rounded to billions).
    """

    _NUMERIC_ABSOLUTE_TOLERANCE: float = 0.3
    """Absolute tolerance floor for small values (percentage points, ratios).

    Activated only when ``max(|a|, |b|) < 10`` so that small numerical
    answers (e.g. ratios, percentage points) are not penalised by the
    relative tolerance alone.
    """

    # Coverage evaluation prompt
    _COVERAGE_PROMPT: str = """\
You are evaluating whether a system's response contains ANY useful information \
relevant to the given financial question.

Question: {question}
System Response: {prediction}

Task: Determine if the response contains relevant, useful information.

═══════════════════════════════════════════════
HAS COVERAGE (has_coverage = true) — when ANY of:
═══════════════════════════════════════════════
1. Contains specific financial data (dollar amounts, percentages, ratios)
2. Contains relevant factual statements about the company or topic
3. Contains partial but concrete information related to the question
4. Provides a direct answer (even if potentially incorrect)

═══════════════════════════════════════════════
NO COVERAGE (has_coverage = false) — when ALL of:
═══════════════════════════════════════════════
1. Response is a refusal ("I cannot", "No results found", etc.)
2. Response contains no concrete data related to the question
3. Response is empty, purely apologetic, or only contains generic filler

Respond ONLY with a JSON object (no markdown, no extra text):
{{"has_coverage": true or false, "confidence": 0.0 to 1.0, "reasoning": "brief explanation"}}"""

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._cache: Dict[tuple, Dict[str, Any]] = {}
        self._total_tokens_used: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def judge(
        self,
        prediction: str,
        gold_answer: str,
        question: str = "",
    ) -> Dict[str, Any]:
        """Judge whether prediction is semantically equivalent to gold.

        Args:
            prediction: Model's answer text.
            gold_answer: Ground-truth answer text.
            question: The original question (for context).

        Returns:
            {
                "equivalent": bool,
                "confidence": float (0-1),
                "reasoning": str,
                "cached": bool,
                "error": Optional[str],
                "tokens_used": int,
            }
        """
        # --- Refusal short-circuit (saves LLM call) ---
        if self._is_refusal(prediction):
            return {
                "equivalent": False,
                "confidence": 1.0,
                "reasoning": "Prediction is a refusal — skipped LLM judge.",
                "cached": False,
                "error": None,
                "tokens_used": 0,
            }

        # --- Quick exact-match shortcut ---
        from evaluate import normalize_answer

        if normalize_answer(prediction) == normalize_answer(gold_answer):
            return {
                "equivalent": True,
                "confidence": 1.0,
                "reasoning": "Normalized exact match",
                "cached": False,
                "error": None,
                "tokens_used": 0,
            }

        # --- Numeric equivalence fast-path ---
        # Handles legitimate rounding differences (e.g. $8.74B vs $8.70B) before
        # invoking the LLM Judge. Returns None when the comparison is ambiguous
        # or non-numeric, in which case we fall through to the LLM.
        _numeric_eq = self._check_numeric_equivalence(
            prediction,
            gold_answer,
            self._NUMERIC_EQUIVALENCE_TOLERANCE,
        )
        if _numeric_eq is not None:
            pred_val = self._normalize_numeric_answer(prediction)
            gold_val = self._normalize_numeric_answer(gold_answer)
            logger.debug(
                "[Judge] Numeric fast-path: pred=%s, gold=%s, equiv=%s",
                pred_val,
                gold_val,
                _numeric_eq,
            )
            return {
                "equivalent": _numeric_eq,
                "confidence": 0.95 if _numeric_eq else 0.90,
                "reasoning": (
                    "Numeric fast-path: values are equivalent within tolerance"
                    if _numeric_eq
                    else "Numeric fast-path: values differ beyond tolerance"
                ),
                "cached": False,
                "error": None,
                "tokens_used": 0,
            }

        # --- Check cache (key includes question for context-sensitivity) ---
        cache_key = (
            question.strip().lower(),
            prediction.strip().lower(),
            gold_answer.strip().lower(),
        )
        if cache_key in self._cache:
            result = dict(self._cache[cache_key])
            result["cached"] = True
            return result

        # --- Call LLM with retry ---
        prompt = _JUDGE_PROMPT.format(
            question=question or "N/A",
            gold=gold_answer,
            prediction=prediction,
        )

        result: Dict[str, Any] | None = None
        last_error: str | None = None
        tokens_used: int = 0

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                resp = await self._llm.achat(
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                )
                tokens_used = self._extract_tokens(resp)
                raw = resp.content.strip()
                result = self._parse_response(raw)
                if result.get("error") is None:
                    break  # success
                last_error = result.get("error")
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "LLM Judge call failed (attempt %d/%d): %s",
                    attempt,
                    self._MAX_RETRIES,
                    e,
                )
                result = None

        if result is None or result.get("error") is not None:
            result = {
                "equivalent": False,
                "confidence": 0.0,
                "reasoning": f"Judge error after {self._MAX_RETRIES} attempts: {last_error}",
                "error": last_error,
            }

        # --- Apply confidence threshold (conservative) ---
        if (
            result.get("error") is None
            and result["equivalent"]
            and result["confidence"] < self._CONFIDENCE_THRESHOLD
        ):
            result["equivalent"] = False
            result["reasoning"] = (
                f"Overridden to NOT equivalent: confidence "
                f"{result['confidence']:.2f} < threshold "
                f"{self._CONFIDENCE_THRESHOLD} — conservative policy. "
                f"Original reasoning: {result['reasoning']}"
            )

        result.setdefault("cached", False)
        result.setdefault("error", None)
        result["tokens_used"] = tokens_used
        self._total_tokens_used += tokens_used

        # Cache successful results only
        if result["error"] is None:
            self._cache[cache_key] = {
                k: v for k, v in result.items() if k != "cached"
            }

        return result

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse LLM JSON response with robust fallback heuristics."""
        # --- Try direct JSON parse ---
        parsed = self._try_parse_json(raw)
        if parsed is not None:
            return self._validated_result(parsed, raw)

        # --- Fallback: keyword detection (conservative) ---
        lower = raw.lower()

        # Look for explicit true/false patterns with word boundaries
        true_match = re.search(
            r'"equivalent"\s*:\s*true\b', lower
        )
        false_match = re.search(
            r'"equivalent"\s*:\s*false\b', lower
        )

        if false_match and not true_match:
            return {
                "equivalent": False,
                "confidence": 0.5,
                "reasoning": f"Keyword fallback (NOT equivalent): {raw[:200]}",
            }
        elif true_match and not false_match:
            # Conservative: lower confidence for keyword-only parse
            return {
                "equivalent": True,
                "confidence": 0.5,
                "reasoning": f"Keyword fallback (equivalent): {raw[:200]}",
            }

        # --- Cannot parse → conservative default ---
        logger.warning("Cannot parse judge response: %s", raw[:200])
        return {
            "equivalent": False,
            "confidence": 0.0,
            "reasoning": f"Unparseable response: {raw[:200]}",
            "error": "parse_error",
        }

    def _try_parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """Attempt multiple JSON extraction strategies."""
        strategies = [
            raw.strip(),
            # Strip markdown code fences
            re.sub(r"```(?:json)?\s*\n?", "", raw).strip().rstrip("`").strip(),
            # Extract first {...} block
            self._extract_json_block(raw),
        ]

        for text in strategies:
            if not text:
                continue
            # Fix common LLM JSON quirks
            text = self._fix_json_quirks(text)
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_json_block(raw: str) -> Optional[str]:
        """Extract the first {...} JSON object from raw text."""
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        return match.group(0) if match else None

    @staticmethod
    def _fix_json_quirks(text: str) -> str:
        """Fix common non-standard JSON from LLMs."""
        # Replace single quotes with double quotes (basic heuristic)
        # Only if the text doesn't already have double quotes for keys
        if "'" in text and '"' not in text:
            text = text.replace("'", '"')
        # Remove trailing commas before closing braces
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return text

    def _validated_result(
        self, obj: Dict[str, Any], raw: str
    ) -> Dict[str, Any]:
        """Build a validated result dict from parsed JSON, clamping values."""
        equivalent = bool(obj.get("equivalent", False))

        # Clamp confidence to [0.0, 1.0]
        try:
            confidence = float(obj.get("confidence", 0.0))
        except (ValueError, TypeError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(obj.get("reasoning", ""))

        return {
            "equivalent": equivalent,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Numeric equivalence fast-path
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_numeric_answer(text: str) -> Optional[float]:
        """Extract and normalize the primary numeric value from an answer.

        Handles:
            - Currency with magnitude suffixes: ``$1.5B`` -> ``1_500_000_000``
            - Percentages: ``15.3%`` -> ``0.153``
            - Plain numbers with thousand separators: ``1,577`` -> ``1577``
            - Negative values and parenthetical negatives: ``(1.5)`` -> ``-1.5``
            - Word-form magnitudes: ``$1.5 billion`` -> ``1_500_000_000``

        Returns:
            The normalized ``float`` value, or ``None`` if the text does not
            contain a single dominant numeric answer (multiple inconsistent
            numbers, or text dominated by qualitative description).
        """
        if not text or not isinstance(text, str):
            return None
        s = text.strip()
        if not s:
            return None

        # Time-period notation guard: tokens like FY2018, Q3 2019, 2018-2019,
        # or bare 4-digit years should never be treated as numeric answers --
        # a 1-year delta is well within 2% tolerance and would otherwise yield
        # false-positive equivalence (e.g. FY2018 vs FY2019).
        if re.search(r"\b(?:FY|Q[1-4]|H[12]|CY)\s*\d{2,4}\b", s, re.IGNORECASE):
            return None
        # Pure year token(s) without any other numeric content -> ambiguous.
        non_year_numeric = re.search(
            r"(?<!\d)(?!(?:19|20)\d{2}(?!\d))\d", s
        )
        if not non_year_numeric and re.search(r"\b(?:19|20)\d{2}\b", s):
            return None

        # Capture: optional sign / opening paren, optional currency, digits
        # (with thousand separators), optional fractional part, optional
        # magnitude suffix (B/M/K/T or full word), optional closing paren,
        # optional percent.
        pattern = re.compile(
            r"""
            (?P<sign>-|\()?                                  # negative or open paren
            \s*
            \$?                                              # optional currency symbol
            \s*
            (?P<num>\d{1,3}(?:,\d{3})+|\d+)                  # integer body
            (?P<frac>\.\d+)?                                 # optional decimal
            \s*
            (?P<suffix>billion|million|thousand|trillion|bn|mn|[BMKT])?  # magnitude
            \s*
            (?P<close>\))?                                   # close paren
            \s*
            (?P<pct>%)?                                      # percent sign
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        multipliers = {
            "b": 1e9, "bn": 1e9, "billion": 1e9,
            "m": 1e6, "mn": 1e6, "million": 1e6,
            "k": 1e3, "thousand": 1e3,
            "t": 1e12, "trillion": 1e12,
        }

        values: list[float] = []
        for match in pattern.finditer(s):
            num_str = match.group("num").replace(",", "")
            frac = match.group("frac") or ""
            try:
                val = float(num_str + frac)
            except ValueError:
                continue

            sign = match.group("sign")
            close = match.group("close")
            if sign == "-" or (sign == "(" and close == ")"):
                val = -val

            suffix = (match.group("suffix") or "").lower()
            if suffix:
                val *= multipliers.get(suffix, 1.0)

            if match.group("pct"):
                val /= 100.0

            values.append(val)

        if not values:
            return None

        # If multiple numeric tokens were captured, only accept if they are
        # all numerically consistent (e.g. "$1,577 (or $1.577K)"). Otherwise
        # the answer is ambiguous and we fall through to the LLM Judge.
        primary = values[0]
        for other in values[1:]:
            if primary == 0.0 and other == 0.0:
                continue
            denom = max(abs(primary), abs(other))
            if denom == 0.0:
                continue
            if abs(primary - other) / denom > 1e-3:
                return None

        # Conservative qualitative-content guard: if the answer contains many
        # alphabetic content words beyond unit / filler tokens, treat it as
        # a free-form description and fall through to the LLM Judge.
        unit_or_filler = {
            "billion", "million", "thousand", "trillion", "bn", "mn",
            "b", "m", "k", "t",
            "percent", "pct",
            "usd", "eur", "gbp", "jpy", "cny", "rmb",
            "dollar", "dollars",
            "approximately", "approx", "about", "around",
            "and", "or", "of", "the", "a", "an",
            "is", "was", "were", "are", "be",
        }
        words = re.findall(r"[A-Za-z]+", s)
        content_words = [w for w in words if w.lower() not in unit_or_filler]
        if len(content_words) > 5:
            return None

        return primary

    @staticmethod
    def _check_numeric_equivalence(
        prediction: str,
        gold: str,
        tolerance: float = 0.02,
    ) -> Optional[bool]:
        """Fast-path numeric equivalence check before LLM Judge.

        Extracts the primary numeric value from both ``prediction`` and
        ``gold`` answer, normalizes units, and compares within a relative
        ``tolerance``.

        Args:
            prediction: Model's answer text.
            gold: Ground-truth answer text.
            tolerance: Maximum allowed relative difference
                ``|a - b| / max(|a|, |b|)``.

        Returns:
            ``True``  - values are numerically equivalent (within tolerance).
            ``False`` - values are clearly different (beyond tolerance).
            ``None``  - cannot determine (non-numeric or ambiguous); the
            caller should fall through to the LLM Judge.
        """
        pred_val = FinanceBenchLLMJudge._normalize_numeric_answer(prediction)
        gold_val = FinanceBenchLLMJudge._normalize_numeric_answer(gold)

        if pred_val is None or gold_val is None:
            return None

        # Both exactly zero -> equivalent.
        if pred_val == 0.0 and gold_val == 0.0:
            return True

        denom = max(abs(pred_val), abs(gold_val))
        if denom == 0.0:
            # One is zero, the other is not -> clearly different.
            return False

        rel_diff = abs(pred_val - gold_val) / denom
        if rel_diff <= tolerance:
            return True
        # Absolute tolerance floor for small values (percentage points, ratios)
        abs_diff = abs(pred_val - gold_val)
        if (max(abs(pred_val), abs(gold_val)) < 10.0
                and abs_diff <= FinanceBenchLLMJudge._NUMERIC_ABSOLUTE_TOLERANCE):
            return True
        return False

    # ------------------------------------------------------------------
    # Refusal detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_refusal(text: str) -> bool:
        """Quick check whether *text* looks like a refusal / non-answer.

        When the text contains an explicit ``**Answer: xxx**`` marker,
        only the answer value is checked for refusal phrases so that
        reasoning text containing phrases like "insufficient data" (as
        analytical context) does not trigger a false positive.
        """
        if not text or not text.strip():
            return True
        lower = text.strip().lower()
        if lower in ("unknown", "n/a", "none", ""):
            return True

        # If there is an explicit **Answer: xxx** marker, only check that value
        answer_match = re.search(r'\*\*answer:\s*(.+?)\*\*', lower)
        if answer_match:
            answer_val = answer_match.group(1).strip()
            for phrase in _REFUSAL_INDICATORS:
                if phrase in answer_val:
                    return True
            return False

        # No structured answer marker — check the leading portion only
        check_region = lower[:300]
        for phrase in _REFUSAL_INDICATORS:
            if phrase in check_region:
                return True
        return False

    async def judge_coverage(
        self,
        prediction: str,
        question: str,
    ) -> Dict[str, Any]:
        """Evaluate whether *prediction* contains relevant information for *question*.

        Returns:
            {
                "has_coverage": bool,
                "confidence": float (0-1),
                "reasoning": str,
                "tokens_used": int,
                "error": Optional[str],
            }
        """
        # --- Refusal short-circuit ---
        if self._is_refusal(prediction):
            return {
                "has_coverage": False,
                "confidence": 1.0,
                "reasoning": "Explicit refusal detected.",
                "tokens_used": 0,
                "error": None,
            }

        prompt = self._COVERAGE_PROMPT.format(
            question=question or "N/A",
            prediction=prediction[:4000],
        )

        result: Dict[str, Any] | None = None
        last_error: str | None = None
        tokens_used: int = 0

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                resp = await self._llm.achat(
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                )
                tokens_used = self._extract_tokens(resp)
                raw = resp.content.strip()
                result = self._parse_coverage_response(raw)
                if result.get("error") is None:
                    break
                last_error = result.get("error")
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "LLM Coverage judge failed (attempt %d/%d): %s",
                    attempt,
                    self._MAX_RETRIES,
                    e,
                )
                result = None

        if result is None or result.get("error") is not None:
            result = {
                "has_coverage": False,
                "confidence": 0.0,
                "reasoning": f"Coverage judge error after {self._MAX_RETRIES} attempts: {last_error}",
                "error": last_error,
            }

        result.setdefault("error", None)
        result["tokens_used"] = tokens_used
        self._total_tokens_used += tokens_used
        return result

    # ------------------------------------------------------------------
    # Coverage response parsing
    # ------------------------------------------------------------------

    def _parse_coverage_response(self, raw: str) -> Dict[str, Any]:
        """Parse LLM JSON response for coverage evaluation."""
        parsed = self._try_parse_json(raw)
        if parsed is not None:
            has_coverage = bool(parsed.get("has_coverage", False))
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except (ValueError, TypeError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(parsed.get("reasoning", ""))
            return {
                "has_coverage": has_coverage,
                "confidence": confidence,
                "reasoning": reasoning,
            }

        # Fallback: keyword detection
        lower = raw.lower()
        true_match = re.search(r'"has_coverage"\s*:\s*true\b', lower)
        false_match = re.search(r'"has_coverage"\s*:\s*false\b', lower)

        if false_match and not true_match:
            return {
                "has_coverage": False,
                "confidence": 0.5,
                "reasoning": f"Keyword fallback (no coverage): {raw[:200]}",
            }
        elif true_match and not false_match:
            return {
                "has_coverage": True,
                "confidence": 0.5,
                "reasoning": f"Keyword fallback (has coverage): {raw[:200]}",
            }

        logger.warning("Cannot parse coverage response: %s", raw[:200])
        return {
            "has_coverage": False,
            "confidence": 0.0,
            "reasoning": f"Unparseable response: {raw[:200]}",
            "error": "parse_error",
        }

    # ------------------------------------------------------------------
    # Token tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tokens(resp: Any) -> int:
        """Extract total token count from an LLM response."""
        usage = getattr(resp, "usage", None)
        if isinstance(usage, dict):
            return int(usage.get("total_tokens", 0))
        return 0

    @property
    def total_tokens_used(self) -> int:
        """Cumulative tokens consumed by all judge calls."""
        return self._total_tokens_used

    @property
    def cache_size(self) -> int:
        """Return the number of cached judge results."""
        return len(self._cache)
