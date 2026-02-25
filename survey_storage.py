"""
Extract survey answers from call transcript and append to Google Sheet.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

import google.auth
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from googleapiclient.discovery import build

from config import (
    EXTRACT_SURVEY_PROMPT,
    GOOGLE_SHEET_ID,
    VERTEX_LOCATION,
    VERTEX_MODEL_ID,
    VERTEX_PROJECT_ID,
)

logger = logging.getLogger(__name__)

# Sheet columns in order (must match user requirement)
SHEET_HEADERS = [
    "Call_id",
    "Age",
    "Gender",
    "State",
    "Q1 - (How Satisfied are you with the Government)",
    "Q2 - (How do you think the Government has done on tacking Price rise)",
    "Q3 - (Will you vote for the Government)",
    "Q4 - (Who do you think is India’s greatest sportsman)",
    "Concern1",
    "Concern2",
    "Concern3",
    "Transcript",
]


def extract_answers_from_transcript(transcript: str, detected_gender: str = None) -> Dict[str, Any]:
    """
    Use Vertex AI to extract structured survey answers from call transcript.
    If detected_gender is supplied (from voice analysis), it overrides whatever
    the LLM extracts from the text (since gender is no longer asked in the survey).
    """
    if not transcript or not transcript.strip():
        # No transcript = call not started, all fields No Response including gender
        return {k: "No Response" for k in ["age", "gender", "state", "q1_satisfaction",
                                            "q2_price_rise", "q3_vote", "q4_greatest_sportsman",
                                            "concern1", "concern2", "concern3"]}

    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        # Try model versions in order — availability varies by project/region
        # The Live audio model (gemini-live-*) does NOT support generateContent
        candidate_models = [
            "gemini-2.0-flash-001",
            "gemini-2.0-flash",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash",
        ]
        model = None
        last_error = None
        for model_id in candidate_models:
            try:
                model = GenerativeModel(model_id)
                # Quick probe — if it raises on init, try next
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_id} unavailable: {e}")
                model = None
        if model is None:
            raise Exception(f"No extraction model available. Last error: {last_error}")
        response = model.generate_content(
            f"{EXTRACT_SURVEY_PROMPT}\n\n---\nTranscript:\n{transcript}"
        )
        text = response.candidates[0].content.parts[0].text.strip()
        # Strip markdown code block if present
        if "```" in text:
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        data = json.loads(text)
        result = {
            "age": data.get("age", "No Response"),
            "gender": data.get("gender", "No Response"),
            "state": data.get("state", "No Response"),
            "q1_satisfaction": data.get("q1_satisfaction", "No Response"),
            "q2_price_rise": data.get("q2_price_rise", "No Response"),
            "q3_vote": data.get("q3_vote", "No Response"),
            "q4_greatest_sportsman": data.get("q4_greatest_sportsman", "No Response"),
        }
        # Split Q5 concerns — LLM returns "concern1","concern2","concern3" or "q5_top3_concerns"
        raw_q5 = data.get("q5_top3_concerns", "")
        c1 = data.get("concern1", "")
        c2 = data.get("concern2", "")
        c3 = data.get("concern3", "")
        if not (c1 or c2 or c3) and raw_q5 and raw_q5 != "No Response":
            parts = [p.strip() for p in raw_q5.split(",")]
            c1 = parts[0] if len(parts) > 0 else "No Response"
            c2 = parts[1] if len(parts) > 1 else "No Response"
            c3 = parts[2] if len(parts) > 2 else "No Response"
        result["concern1"] = c1 or "No Response"
        result["concern2"] = c2 or "No Response"
        result["concern3"] = c3 or "No Response"
        # Always override gender with voice-detected value if available
        if detected_gender:
            result["gender"] = detected_gender
        return result
    except Exception as e:
        logger.exception("Failed to extract survey answers from transcript: %s", e)
        result = {
            "age": "No Response",
            "gender": "No Response",
            "state": "No Response",
            "q1_satisfaction": "No Response",
            "q2_price_rise": "No Response",
            "q3_vote": "No Response",
            "q4_greatest_sportsman": "No Response",
            "concern1": "No Response",
            "concern2": "No Response",
            "concern3": "No Response",
        }
        if detected_gender:
            result["gender"] = detected_gender
        return result


def append_survey_to_sheet(
    call_id: str,
    answers: Dict[str, Any],
    transcript: str,
) -> bool:
    """
    Append one survey response as a row to the configured Google Sheet.
    Sheet must exist and have at least one row (header). Uses default credentials.
    """
    if not GOOGLE_SHEET_ID:
        logger.warning("GOOGLE_SHEET_ID not set; skipping Google Sheet append")
        return False

    row: List[Any] = [
        call_id,
        answers.get("age", ""),
        answers.get("gender", ""),
        answers.get("state", ""),
        answers.get("q1_satisfaction", ""),
        answers.get("q2_price_rise", ""),
        answers.get("q3_vote", ""),
        answers.get("q4_greatest_sportsman", ""),
        answers.get("concern1", ""),
        answers.get("concern2", ""),
        answers.get("concern3", ""),
        transcript[:50000] if transcript else "",  # Sheets cell limit
    ]

    try:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=credentials)
        # Ensure header row exists
        header_range = "Sheet1!A1:L1"
        header_result = service.spreadsheets().values().get(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=header_range,
        ).execute()
        header_values = header_result.get("values", [])
        if not header_values or not any(header_values):
            service.spreadsheets().values().update(
                spreadsheetId=GOOGLE_SHEET_ID,
                range=header_range,
                valueInputOption="USER_ENTERED",
                body={"values": [SHEET_HEADERS]},
            ).execute()
            logger.info("Written header row to Google Sheet")
        body = {"values": [row]}
        service.spreadsheets().values().append(
            spreadsheetId=GOOGLE_SHEET_ID,
            range="Sheet1!A:L",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body,
        ).execute()
        logger.info("Appended survey row for call_id=%s to Google Sheet", call_id)
        return True
    except Exception as e:
        logger.exception("Failed to append to Google Sheet: %s", e)
        return False