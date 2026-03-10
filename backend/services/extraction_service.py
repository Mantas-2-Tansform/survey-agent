"""
services/extraction_service.py

Dynamically extract structured survey answers from a call transcript.
The extraction schema is built at runtime from the campaign's questions —
no hardcoded fields, no hardcoded column names.

Extraction principles:
  • temperature=0 — deterministic, no creative filling.
  • Only extract answers that were actually spoken.
  • Missing / unclear / refused answers → "No Response".
  • Returns a dict keyed by question_id (UUID str).
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

import vertexai
from vertexai.preview.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

_CANDIDATE_MODELS: List[str] = [
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash",
]

_NO_RESPONSE = "No Response"

_INVALID_PLACEHOLDERS = {
    "", "none", "n/a", "na", "not mentioned", "not stated", "unknown",
    "unclear", "-", "–", "—", "null", "not provided", "not given",
    "not available", "no answer", "didn't say",
    # Regional language equivalents that may leak through if translation fails
    "पता नहीं", "नहीं पता", "मालूम नहीं",   # Hindi: don't know
    "தெரியாது", "இல்லை",                      # Tamil: don't know / no
    "తెలియదు",                                  # Telugu: don't know
    "ಗೊತ್ತಿಲ್ಲ",                               # Kannada: don't know
    "അറിയില്ല",                                # Malayalam: don't know
    "জানি না",                                  # Bengali: don't know
    "ખબર નથી",                                 # Gujarati: don't know
    "माहित नाही",                               # Marathi: don't know
}


# ---------------------------------------------------------------------------
# Vertex AI helpers
# ---------------------------------------------------------------------------

def _get_model(project: str, location: str) -> GenerativeModel:
    vertexai.init(project=project, location=location)
    for model_id in _CANDIDATE_MODELS:
        try:
            m = GenerativeModel(model_id)
            logger.info("ExtractionService: using model %s", model_id)
            return m
        except Exception as e:
            logger.warning("Model %s unavailable: %s", model_id, e)
    raise RuntimeError("No Vertex AI model available for answer extraction")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_extraction_prompt(questions: List[Dict[str, Any]], transcript: str) -> str:
    """
    Build an LLM prompt that instructs the model to extract answers
    for every question, keyed by question_id.

    questions: list of dicts {"id", "question_order", "question_text", "question_type", "options"}
    """
    q_schema_lines = []
    for q in sorted(questions, key=lambda x: x["question_order"]):
        qid = str(q["id"])
        q_text = q["question_text"]
        q_type = q.get("question_type", "open_text")

        # Build type hint — handle N=Label scale options properly
        if q_type in ("scale", "scale_1_5"):
            opts = q.get("options", [])
            if opts and "=" in str(opts[0]):
                # N=Label format: e.g. ["1=Daily","2=Sometimes","4=Never","8=No response"]
                # Extract the numeric values the respondent should give
                labels = []
                nums = []
                for o in opts:
                    parts = str(o).split("=", 1)
                    if len(parts) == 2:
                        labels.append(f"{parts[0].strip()} ({parts[1].strip()})")
                        nums.append(parts[0].strip())
                type_hint = f"one of these integers: {', '.join(nums)} — or 'No Response'. Valid options: {'; '.join(labels)}"
            elif len(opts) >= 2 and "=" not in str(opts[0]):
                type_hint = f"integer {opts[0]}-{opts[1]} or 'No Response'"
            else:
                type_hint = "an integer or 'No Response'"
        else:
            type_hint = {
                "yes_no":          "'Yes' or 'No' or 'No Response'",
                "numeric":         "a number or 'No Response'",
                "multiple_choice": f"one of {q.get('options', [])} or 'No Response'",
                "open_text":       "a short text summary or 'No Response'",
            }.get(q_type, "a short text summary or 'No Response'")

        q_schema_lines.append(f'  "{qid}": <{type_hint}>  // Q{q["question_order"]}: {q_text}')

    schema_block = "\n".join(q_schema_lines)

    return f"""You are a survey data extractor. The transcript may contain speech in multiple Indian languages
(Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Urdu) as well as Indian English.
Extract answers from the call transcript below.

Return a single JSON object exactly matching this schema:
{{
{schema_block}
}}

STRICT RULES:
1. Only extract answers that were clearly stated by the RESPONDENT (not the interviewer/Sneha).
2. If a question was not reached, was skipped, or the respondent refused, use "No Response".
3. Do NOT infer, guess, or hallucinate any answer.
4. Do NOT copy the question text as an answer.
5. For scale / scale_1_5: return only the number (e.g. 4), not a label.
6. For yes_no: return only "Yes" or "No" in English — translate हाँ/हाँ/ஆம்/అవును etc. to "Yes"; नहीं/இல்லை/కాదు etc. to "No".
7. For multiple_choice: return only one of the listed options exactly as written in English.
8. LANGUAGE TRANSLATION — ALL answers MUST be in English:
   - Translate any answer given in Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali,
     Gujarati, Marathi, or Urdu into English before storing.
   - For names of people, places, or brands spoken in a regional language,
     transliterate them to Roman script (English letters).
   - For numeric values spoken as words in any language, convert to digits (e.g. "पाँच" → 5).
9. Return ONLY the JSON object. No markdown, no preamble, no explanation.

---
TRANSCRIPT:
{transcript}
""".strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_answers(
    transcript: str,
    questions: List[Dict[str, Any]],
    vertex_project: str,
    vertex_location: str,
    detected_gender: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract structured answers from a transcript.

    Args:
        transcript:      Full call transcript text.
        questions:       List of campaign question dicts:
                         {"id", "question_order", "question_text",
                          "question_type", "options", "required"}
        vertex_project:  GCP project for Vertex AI.
        vertex_location: GCP region for Vertex AI.
        detected_gender: Optional voice-detected gender ("M"/"F"/"Unknown").
                         Stored alongside answers but not mapped to a question.

    Returns:
        Dict keyed by question_id (str) → extracted answer.
        Includes special key "_meta" with call-level metadata.
    """
    empty_result: Dict[str, Any] = {
        str(q["id"]): _NO_RESPONSE for q in questions
    }
    empty_result["_meta"] = {
        "detected_gender": detected_gender or "Unknown",
        "participated": False,
    }

    if not transcript or not transcript.strip():
        return empty_result

    try:
        model = _get_model(vertex_project, vertex_location)
        prompt = _build_extraction_prompt(questions, transcript)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        text = response.candidates[0].content.parts[0].text.strip()

        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

        raw: Dict[str, Any] = json.loads(text)

        # Sanitise — replace empty / invalid placeholders with "No Response"
        result: Dict[str, Any] = {}
        for q in questions:
            qid = str(q["id"])
            val = raw.get(qid, _NO_RESPONSE)
            if val is None or str(val).strip().lower() in _INVALID_PLACEHOLDERS:
                val = _NO_RESPONSE
            result[qid] = val

        # Determine whether the respondent actually participated
        participated = any(v != _NO_RESPONSE for v in result.values())

        result["_meta"] = {
            "detected_gender": detected_gender if (detected_gender and participated) else "Unknown",
            "participated": participated,
        }

        logger.info(
            "Extraction complete: %d questions, participated=%s",
            len(questions), participated
        )
        return result

    except Exception as e:
        logger.exception("Answer extraction failed: %s", e)
        return empty_result