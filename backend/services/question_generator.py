"""
services/question_generator.py

Generate survey questions from a plain-English campaign description using Vertex AI.

This replaces the document-upload pipeline when the admin doesn't have a pre-written
questionnaire — they describe what they want and the AI designs the questions.

Returns the same structure as document_parser.parse_document() so the same
downstream save flow (POST /admin/campaign/{id}/questions) works unchanged.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

import vertexai
from vertexai.preview.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

# Standard text/generation models only.
# Live models (gemini-*-live-*) are voice streaming models — they only work
# with the BidiGenerateContent WebSocket API and will error if used here.
# This list mirrors exactly what main.py uses for the /api/ai analysis endpoint.
_TEXT_MODELS: List[str] = [
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash",
]


def _get_model(project: str, location: str) -> GenerativeModel:
    """
    Return the best available standard text model.

    Reads VERTEX_MODEL_ID from config first — if it is a live/streaming model
    it is skipped automatically since live model IDs contain "live".
    Falls back through _TEXT_MODELS in order.
    """
    vertexai.init(project=project, location=location)

    # Prefer whatever model the operator has explicitly configured, but only if
    # it is a standard text model (not a live/streaming model).
    try:
        import config as _c
        configured = getattr(_c, "VERTEX_MODEL_ID", "").strip()
        if configured and "live" not in configured.lower():
            m = GenerativeModel(configured)
            logger.info("QuestionGenerator: using configured model %s", configured)
            return m
    except Exception as e:
        logger.warning("Configured model unavailable, falling back: %s", e)

    for model_id in _TEXT_MODELS:
        try:
            m = GenerativeModel(model_id)
            logger.info("QuestionGenerator: using model %s", model_id)
            return m
        except Exception as e:
            logger.warning("Model %s unavailable: %s", model_id, e)
    raise RuntimeError("No Vertex AI text model available for question generation")


_GENERATION_SYSTEM_PROMPT = """
You are an expert survey designer. Your task is to create a well-structured,
professional survey based on the campaign description provided.

QUESTION TYPES — choose the most appropriate for each question:
  "scale"           - Rating/Likert/frequency scale with labelled integer values.
                      Use options in "N=Label" format.
                      Example: ["1=Very Dissatisfied","2=Dissatisfied","3=Neutral","4=Satisfied","5=Very Satisfied"]
  "multiple_choice" - Discrete named options, respondent picks one.
                      Options are plain label strings.
                      Example: ["Yes, definitely","Possibly","No","Don't know"]
  "yes_no"          - Strictly binary Yes/No. options=[].
  "numeric"         - Pure number (age, count, etc.). options=[].
  "open_text"       - Free-form text answer. options=[].

SURVEY DESIGN PRINCIPLES:
1. Start with easy, non-sensitive questions to build rapport.
2. Group related topics together.
3. Use scale questions for opinions/satisfaction/frequency measurements.
4. Use multiple_choice when there are 3–6 distinct named options.
5. Use yes_no only for truly binary questions.
6. Use numeric for demographic facts (age, income bracket as number, etc.).
7. Use open_text sparingly — max 1–2 per survey, for qualitative insights.
8. Questions must be clear, unambiguous, and free from leading language.
9. Include a "No Response" / "Don't know" option in scale and multiple_choice
   questions where it's natural for a respondent to genuinely not know.
10. For scale questions, always include "8=No response" or similar as last option.

LOGIC/BRANCHING HINTS:
If the campaign description mentions conditional flows (e.g. "if they say Yes to X,
ask Y"), include a "logic" field on the question:
  "logic": [
    {"condition": "Yes", "next_order": 3},
    {"condition": "No",  "next_order": 5}
  ]
For questions with no branching, omit the "logic" field entirely.

OUTPUT FORMAT:
Return ONLY a valid JSON array, no markdown fences, no explanation.
Each element MUST have:
{
  "question_number": <integer, sequential from 1>,
  "question_text":   <complete standalone question text>,
  "question_type":   <scale | multiple_choice | yes_no | numeric | open_text>,
  "options":         <array of strings, [] when not applicable>,
  "required":        <true | false>,
  "logic":           <array of condition objects, omit if no branching>
}

RULES:
1. Produce EXACTLY the number of questions requested.
2. question_number must be sequential integers starting at 1.
3. Every question must stand alone — no references like "as mentioned above".
4. Match question types to the nature of each question (do not use open_text for everything).
5. Align closely with the campaign description's research objectives.
6. Never fabricate demographic questions not implied by the description.
""".strip()


def generate_questions(
    description: str,
    num_questions: int,
    vertex_project: str,
    vertex_location: str,
    extra_instructions: str = "",
) -> List[Dict[str, Any]]:
    """
    Generate survey questions from a campaign description.

    Args:
        description:        Detailed plain-English description of the survey campaign,
                            its objectives, target audience, and topics to cover.
        num_questions:      Exact number of questions to generate (1–50).
        vertex_project:     GCP project ID for Vertex AI.
        vertex_location:    GCP region (e.g. "us-central1").
        extra_instructions: Optional additional guidance for the AI (e.g. language,
                            tone, specific topics to emphasise or exclude).

    Returns:
        List of question dicts compatible with the admin QuestionCreate schema:
        [
            {
                "question_number": 1,
                "question_text":   "...",
                "question_type":   "scale",
                "options":         ["1=Very Dissatisfied", ..., "8=No response"],
                "required":        True,
                "logic":           [...]   # only if branching
            },
            ...
        ]

    Raises:
        ValueError  if inputs are invalid.
        RuntimeError if no Vertex AI model is available.
    """
    if not description or not description.strip():
        raise ValueError("Campaign description must not be empty.")

    if not (1 <= num_questions <= 50):
        raise ValueError("num_questions must be between 1 and 50.")

    model = _get_model(vertex_project, vertex_location)

    hint_block = (
        f"\n\nADDITIONAL INSTRUCTIONS FROM ADMIN:\n{extra_instructions.strip()}"
        if extra_instructions.strip()
        else ""
    )

    prompt = (
        f"{_GENERATION_SYSTEM_PROMPT}{hint_block}\n\n"
        f"---\n"
        f"CAMPAIGN DESCRIPTION:\n{description.strip()}\n\n"
        f"Generate exactly {num_questions} survey questions for this campaign.\n"
        f"Return ONLY the JSON array with {num_questions} elements."
    )

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.3},  # slight creativity for question wording
    )
    text = response.candidates[0].content.parts[0].text.strip()

    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"AI returned unexpected structure: {type(data)}")

    # Enforce sequential numbering and defaults
    result = []
    for i, q in enumerate(data, start=1):
        q["question_number"] = i
        q.setdefault("required", True)
        q.setdefault("options", [])
        q.setdefault("logic", None)

        # Remove logic key if it's None/empty (keep it clean)
        if not q.get("logic"):
            q.pop("logic", None)

        result.append(q)

    logger.info(
        "Generated %d questions for description (%.60s...)",
        len(result),
        description,
    )
    return result