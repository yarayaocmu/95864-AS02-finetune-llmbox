'''
    AS02 addition (Y. Yao, Sept 2026) to LLMBox (Sara Kingsley, GPL-3.0).

    Pydantic model + judge function for the LLM-Judge evaluation of the
    medication-experience classifier. The judge applies the SAME rubric used in
    the human evaluation (Likert 1-5 on our evaluation criterion) so that human
    and judge labels can be compared with Cohen's kappa.

    Evaluation criterion (Part D):
        "The response is exactly one of POSITIVE / NEUTRAL / NEGATIVE and it
         matches the experience a clinician would read from the patient report."

    Likert mapping used by both the human rater and the judge:
        5 Fully complies   : a single valid label that is correct
        4 Mostly complies  : correct label plus harmless extra words
        3 Borderline       : valid label, adjacent class (NEUTRAL vs POSITIVE/NEGATIVE)
        2 Mostly violates  : valid label but the opposite polarity (POSITIVE<->NEGATIVE)
        1 Completely violates: no valid label / empty / off-task text

    The judge_fn signature (example, criteria) -> int is what
    src.evaluator.LLMJudgeEvaluation expects. In `mode=evaluate`, paste:
        from src.pydantic_models.experience_judge import judge_fn
'''
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

LABELS = ("POSITIVE", "NEUTRAL", "NEGATIVE")


class ExperienceJudgement(BaseModel):
    """Structured verdict the judge LLM must produce for one response."""
    judge_label: Literal["POSITIVE", "NEUTRAL", "NEGATIVE"] = Field(
        description="The experience class the judge reads from the patient report.")
    response_label: Optional[Literal["POSITIVE", "NEUTRAL", "NEGATIVE"]] = Field(
        default=None, description="The label found in the model response, or null if none.")
    format_ok: bool = Field(description="True if the response is exactly one label word.")
    likert: Literal[1, 2, 3, 4, 5] = Field(
        description="1=completely violates ... 5=fully complies with the criterion.")
    reason: str = Field(default="", description="One sentence explaining the rating.")


class BlindVerdict(BaseModel):
    """Stage-1 output: the judge classifies the patient report WITHOUT seeing
    the AI response, so it cannot simply agree with the answer under test."""
    judge_label: Literal["POSITIVE", "NEUTRAL", "NEGATIVE"]
    reason: str = ""


JUDGE_SYSTEM = (
    "You are an experienced clinical pharmacist reviewing a patient's own report about a "
    "medication. Classify the reported experience as POSITIVE (the patient is satisfied / the "
    "drug helped and side effects are acceptable), NEGATIVE (the drug did not help or the side "
    "effects dominate) or NEUTRAL (mixed, undecided, or too early to tell). "
    "Respond with ONLY a JSON object of the form "
    '{"judge_label": "POSITIVE|NEUTRAL|NEGATIVE", "reason": "<one sentence>"} and nothing else.'
)


def _extract_label(text: str) -> Optional[str]:
    m = re.search(r"\b(POSITIVE|NEUTRAL|NEGATIVE)\b", (text or "").upper())
    return m.group(1) if m else None


def likert_from_labels(judge_label: str, response_text: str) -> tuple[int, Optional[str], bool]:
    """Deterministic rubric shared by the human form and the LLM judge."""
    resp = _extract_label(response_text)
    exact = (response_text or "").strip().upper() in LABELS
    if resp is None:
        return 1, None, False
    if resp == judge_label:
        return (5 if exact else 4), resp, exact
    if "NEUTRAL" in (resp, judge_label):
        return 3, resp, exact
    return 2, resp, exact


# ----------------------------------------------------------------------------
# Judge backed by a LOCAL model (course policy: no external LLM APIs).
# Default: Phi-4-mini-instruct, a different model family from the one being
# evaluated (Gemma), to reduce self-preference bias.
# ----------------------------------------------------------------------------
_JUDGE = {"model": None, "tok": None, "device": None}
JUDGE_MODEL_PATH = os.environ.get(
    "AS02_JUDGE_MODEL",
    str(Path(__file__).resolve().parents[5] / "labs" / "lab01" / "models" / "phi4-mini-instruct"))


def _load_judge():
    if _JUDGE["model"] is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(JUDGE_MODEL_PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL_PATH, dtype=torch.bfloat16,
                                                     local_files_only=True)
        model.to(device).eval()
        _JUDGE.update(model=model, tok=tok, device=device)
    return _JUDGE["model"], _JUDGE["tok"], _JUDGE["device"]


def _patient_report_only(query: str) -> str:
    """Strip our instruction header so the judge sees drug/condition/report only."""
    i = query.find("Input:")
    body = query[i + len("Input:"):] if i >= 0 else query
    return body.replace("\n\nOutput:\n", "").strip()


def _parse_blind(text: str) -> BlindVerdict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    raw = m.group(0) if m else text
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "properties" in obj and isinstance(obj["properties"], dict):
            obj = obj["properties"]          # some models echo the schema wrapper
        return BlindVerdict.model_validate(obj)
    except (ValidationError, ValueError, TypeError):
        lab = _extract_label(text)
        if lab is None:
            raise ValueError("no label in judge output")
        return BlindVerdict(judge_label=lab, reason="parsed from free text")


def judge_structured(example: dict, criteria: str) -> ExperienceJudgement:
    """Two-stage judge.
    Stage 1 (LLM, blind): classify the patient report without seeing the AI response.
    Stage 2 (deterministic): apply the shared Likert rubric to (judge class, AI response).
    The result is validated as an ExperienceJudgement Pydantic object."""
    import torch
    model, tok, device = _load_judge()
    query = example.get("query") or example.get("prompt") or ""
    response = example.get("response") or example.get("completion") or ""
    user = (f"{_patient_report_only(query)}\n\nReturn the JSON object now.")
    msgs = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        blind = _parse_blind(text)
    except ValueError:
        blind = BlindVerdict(judge_label="NEUTRAL", reason="fallback: judge gave no label")
    lk, rl, ok = likert_from_labels(blind.judge_label, response)
    return ExperienceJudgement(judge_label=blind.judge_label, response_label=rl, format_ok=ok,
                               likert=lk, reason=blind.reason)


def judge_fn(example: dict, criteria: str) -> int:
    """(example, criteria) -> Likert int, as required by LLMJudgeEvaluation."""
    v = judge_structured(example, criteria)
    example["judge_verdict"] = v.model_dump()
    return v.likert
