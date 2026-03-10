"""
services/sheet_service.py

Google Sheets integration with per-campaign tab support.

Responsibilities:
  • Sanitise campaign names to valid sheet tab names.
  • Create a new tab (sheet) for each campaign inside the configured spreadsheet.
  • Write a dynamic header row based on campaign questions.
  • Append response rows to the correct campaign tab.
  • Read back all rows from a campaign tab (for the dashboard).
"""

import logging
import re
from typing import Any, Dict, List

import google.auth
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

# Max tab name length enforced by Google Sheets
_SHEET_TAB_MAX_LEN = 100

# Fixed columns that wrap every row
_COL_CALL_ID = "Call_id"
_COL_GENDER = "Gender (Voice Detected)"
_COL_TRANSCRIPT = "Transcript"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitise_tab_name(name: str) -> str:
    """
    Convert an arbitrary string to a valid Google Sheets tab name.
    Sheets disallows: / \\ ? * [ ]
    """
    clean = re.sub(r"[/\\?*\[\]]", "_", name)
    clean = clean.strip()

    # Collapse whitespace
    clean = re.sub(r"\s+", " ", clean)

    return clean[:_SHEET_TAB_MAX_LEN] or "Campaign"


def _sheet_range(tab_name: str, cells: str) -> str:
    """
    Build a safe Google Sheets A1 range string.
    Always quoting prevents parsing errors.
    """

    safe = tab_name.replace("'", "''")
    return f"'{safe}'!{cells}"


def _build_headers(questions: List[Dict[str, Any]]) -> List[str]:
    """
    Build the full header row for a campaign tab.
    """

    headers = [_COL_CALL_ID, _COL_GENDER]

    for q in sorted(questions, key=lambda x: x["question_order"]):
        text = q["question_text"][:80]
        headers.append(f"Q{q['question_order']} - {text}")

    headers.append(_COL_TRANSCRIPT)

    return headers


def _get_sheets_service(readonly: bool = False):
    """Build authenticated Google Sheets API service."""

    scope = (
        "https://www.googleapis.com/auth/spreadsheets.readonly"
        if readonly
        else "https://www.googleapis.com/auth/spreadsheets"
    )

    credentials, _ = google.auth.default(scopes=[scope])

    return build("sheets", "v4", credentials=credentials)


# ---------------------------------------------------------------------------
# Tab management
# ---------------------------------------------------------------------------

def create_campaign_tab(
    spreadsheet_id: str,
    tab_name: str,
    questions: List[Dict[str, Any]],
) -> str:
    """
    Create a new sheet tab in the spreadsheet for a campaign.
    Writes the dynamic header row.
    """

    service = _get_sheets_service()

    meta = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id
    ).execute()

    existing_titles = {
        s["properties"]["title"]
        for s in meta.get("sheets", [])
    }

    if tab_name not in existing_titles:

        body = {
            "requests": [
                {
                    "addSheet": {
                        "properties": {"title": tab_name}
                    }
                }
            ]
        }

        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body
        ).execute()

        logger.info("Created sheet tab: %s", tab_name)

    else:
        logger.info("Sheet tab already exists: %s", tab_name)

    headers = _build_headers(questions)

    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=_sheet_range(tab_name, "A1"),
        valueInputOption="USER_ENTERED",
        body={"values": [headers]},
    ).execute()

    logger.info("Written header row to tab '%s': %s", tab_name, headers)

    return tab_name


# ---------------------------------------------------------------------------
# Row append
# ---------------------------------------------------------------------------

def append_response_row(
    spreadsheet_id: str,
    tab_name: str,
    call_id: str,
    structured_answers: Dict[str, Any],
    questions: List[Dict[str, Any]],
    transcript: str,
    detected_gender: str = "Unknown",
) -> bool:

    try:
        service = _get_sheets_service()

        # -------------------------------------------------
        # Ensure tab exists (create if missing)
        # -------------------------------------------------
        meta = service.spreadsheets().get(
            spreadsheetId=spreadsheet_id
        ).execute()

        existing_tabs = {
            s["properties"]["title"]
            for s in meta.get("sheets", [])
        }

        if tab_name not in existing_tabs:
            logger.warning("Sheet tab '%s' not found. Creating it.", tab_name)
            create_campaign_tab(spreadsheet_id, tab_name, questions)

        # -------------------------------------------------
        # Build row
        # -------------------------------------------------

        row: List[Any] = [call_id, detected_gender]

        for q in sorted(questions, key=lambda x: x["question_order"]):
            qid = str(q["id"])
            row.append(structured_answers.get(qid, "No Response"))

        row.append(transcript[:50000] if transcript else "")

        body = {"values": [row]}

        service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=_sheet_range(tab_name, "A:A"),
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body,
        ).execute()

        logger.info(
            "Appended response row for call_id=%s to tab='%s'",
            call_id,
            tab_name,
        )

        return True

    except Exception as e:
        logger.exception("Failed to append row to sheet: %s", e)
        return False


# ---------------------------------------------------------------------------
# Read back rows
# ---------------------------------------------------------------------------

def read_campaign_responses(
    spreadsheet_id: str,
    tab_name: str,
) -> Dict[str, Any]:
    """
    Read all rows from a campaign tab.
    """

    try:
        service = _get_sheets_service(readonly=True)

        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=_sheet_range(tab_name, "A:ZZ"),
        ).execute()

        rows = result.get("values", [])

        if not rows:
            return {
                "headers": [],
                "data": [],
                "count": 0
            }

        headers = rows[0]

        data = []

        for row in rows[1:]:

            padded = row + [""] * (len(headers) - len(row))

            data.append(dict(zip(headers, padded)))

        return {
            "headers": headers,
            "data": data,
            "count": len(data)
        }

    except Exception as e:

        logger.exception(
            "Failed to read from sheet tab '%s': %s",
            tab_name,
            e
        )

        return {
            "headers": [],
            "data": [],
            "count": 0,
            "error": str(e)
        }