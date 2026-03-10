# telephony/vicidial_client.py
"""
VICIdial API Client
====================
Handles all POST-survey communication back to VICIdial:
  - Disposition updates  (mark lead as SALE/DNC/NI/CALLBK etc.)
  - Lead field updates   (write survey answers back to lead record)
  - Call hangup          (programmatically end active call)
  - Campaign management  (Phase 2 — upload leads, start/pause campaigns)

VICIdial exposes two API surfaces:
  1. Agent API     — /agc/api.php          (actions on behalf of a logged-in agent)
  2. Non-Agent API — /vicidial/non_agent_api.php  (admin actions, lead management)

Both use HTTP GET/POST with form-encoded params and return plain text or JSON.

NOTE: Exact endpoints, required fields, and auth params will be confirmed by
the VICIdial team. This client is structured around the questions in our email
and uses standard VICIdial API conventions. Update the constants once you get
their response.

Env vars:
  VICIDIAL_API_URL       — Base URL (e.g. https://dialer.example.com)
  VICIDIAL_API_USER      — API username
  VICIDIAL_API_PASS      — API password
  VICIDIAL_AGENT_USER    — Agent username our AI system is registered as
  VICIDIAL_AGENT_PASS    — Agent password
  VICIDIAL_SOURCE        — API source identifier (default: "survey_ai")
"""

import os
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION  (update once VICIdial team confirms)
# =============================================================================

VICIDIAL_API_URL = os.getenv("VICIDIAL_API_URL", "https://dialer.example.com")
VICIDIAL_API_USER = os.getenv("VICIDIAL_API_USER", "")
VICIDIAL_API_PASS = os.getenv("VICIDIAL_API_PASS", "")
VICIDIAL_AGENT_USER = os.getenv("VICIDIAL_AGENT_USER", "survey_ai_agent")
VICIDIAL_AGENT_PASS = os.getenv("VICIDIAL_AGENT_PASS", "")
VICIDIAL_SOURCE = os.getenv("VICIDIAL_SOURCE", "survey_ai")

# Standard VICIdial API paths (may differ per installation)
AGENT_API_PATH = "/agc/api.php"
NON_AGENT_API_PATH = "/vicidial/non_agent_api.php"


# =============================================================================
# DISPOSITION CODES  (map survey outcomes → VICIdial status codes)
# =============================================================================

class SurveyDisposition(str, Enum):
    """
    Survey outcome → VICIdial disposition code mapping.
    Update these once VICIdial team confirms their status codes.
    """
    COMPLETED = "SVYCMP"      # Survey fully completed
    PARTIAL = "SVYPAR"        # Partial completion (respondent hung up mid-survey)
    REFUSED = "SVYREF"        # Respondent declined to participate
    LANGUAGE_BARRIER = "SVYLB"  # Could not communicate
    NO_RESPONSE = "NI"        # No meaningful interaction
    CALLBACK = "CALLBK"       # Respondent asked to be called back
    DNC = "DNC"               # Do not call request
    WRONG_NUMBER = "WRONG"    # Wrong number
    VOICEMAIL = "AA"          # Answering machine / voicemail
    AGENT_ERROR = "SYSERR"    # System/agent error


# Maps our internal survey result → VICIdial disposition
def map_survey_to_disposition(answers: Dict[str, Any], transcript: str) -> str:
    """
    Determine VICIdial disposition code from survey answers.

    Logic:
    - If any substantive answer exists → COMPLETED
    - If transcript exists but all answers are "No Response" → REFUSED or PARTIAL
    - If no transcript → NO_RESPONSE
    """
    if not transcript or not transcript.strip():
        return SurveyDisposition.NO_RESPONSE.value

    substantive_fields = [
        "q1_satisfaction", "q2_price_rise", "q3_vote",
        "q4_greatest_sportsman", "age", "state",
    ]
    has_answer = any(
        answers.get(f) not in (None, "", "No Response")
        for f in substantive_fields
    )

    if has_answer:
        return SurveyDisposition.COMPLETED.value
    else:
        return SurveyDisposition.REFUSED.value


# =============================================================================
# API CLIENT
# =============================================================================

class VicidialClient:
    """
    HTTP client for VICIdial Agent API and Non-Agent API.
    All methods are async and fail gracefully (log + return error dict).
    """

    def __init__(
        self,
        api_url: str = None,
        api_user: str = None,
        api_pass: str = None,
        agent_user: str = None,
        agent_pass: str = None,
        source: str = None,
    ):
        self.api_url = (api_url or VICIDIAL_API_URL).rstrip("/")
        self.api_user = api_user or VICIDIAL_API_USER
        self.api_pass = api_pass or VICIDIAL_API_PASS
        self.agent_user = agent_user or VICIDIAL_AGENT_USER
        self.agent_pass = agent_pass or VICIDIAL_AGENT_PASS
        self.source = source or VICIDIAL_SOURCE

        if not self.api_user:
            logger.warning("⚠️ VICIDIAL_API_USER not set — VICIdial API calls will fail")

    # ── Shared request helpers ──────────────────────────────────────────────

    def _agent_api_url(self) -> str:
        return f"{self.api_url}{AGENT_API_PATH}"

    def _non_agent_api_url(self) -> str:
        return f"{self.api_url}{NON_AGENT_API_PATH}"

    def _base_agent_params(self) -> Dict[str, str]:
        """Common params for Agent API calls."""
        return {
            "source": self.source,
            "user": self.agent_user,
            "pass": self.agent_pass,
        }

    def _base_non_agent_params(self) -> Dict[str, str]:
        """Common params for Non-Agent API calls."""
        return {
            "source": self.source,
            "user": self.api_user,
            "pass": self.api_pass,
        }

    async def _post(self, url: str, params: Dict[str, str]) -> Dict[str, Any]:
        """Make an HTTP POST to VICIdial API and parse the response."""
        import httpx

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, data=params)
                response_text = response.text.strip()

                logger.info(
                    f"VICIdial API [{response.status_code}]: "
                    f"{params.get('function', '?')} → {response_text[:200]}"
                )

                # VICIdial returns plain text like "SUCCESS: ..." or "ERROR: ..."
                success = (
                    response.status_code == 200
                    and not response_text.upper().startswith("ERROR")
                )

                return {
                    "success": success,
                    "status_code": response.status_code,
                    "response": response_text,
                    "function": params.get("function", ""),
                }

        except Exception as e:
            logger.error(f"VICIdial API error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "function": params.get("function", ""),
            }

    # =========================================================================
    # 1. DISPOSITION UPDATE  (after survey completes)
    # =========================================================================

    async def update_disposition(
        self,
        lead_id: str,
        status: str,
        *,
        uniqueid: str = None,
        campaign_id: str = None,
        comments: str = None,
    ) -> Dict[str, Any]:
        """
        Update the disposition/status of a lead after the survey call.

        Uses Non-Agent API: function=update_lead

        Args:
            lead_id:     VICIdial lead ID (from SIP headers or call metadata)
            status:      Disposition code (e.g. "SVYCMP", "SVYREF", "DNC")
            uniqueid:    VICIdial call uniqueid (for linking to specific call)
            campaign_id: Campaign ID (if needed for the API)
            comments:    Free-text notes about the call
        """
        params = {
            **self._base_non_agent_params(),
            "function": "update_lead",
            "lead_id": lead_id,
            "status": status,
        }
        if uniqueid:
            params["uniqueid"] = uniqueid
        if campaign_id:
            params["campaign_id"] = campaign_id
        if comments:
            params["comments"] = comments[:255]  # VICIdial field limit

        logger.info(f"📋 Updating disposition: lead={lead_id} status={status}")
        return await self._post(self._non_agent_api_url(), params)

    # =========================================================================
    # 2. LEAD FIELD UPDATE  (write survey answers back to lead record)
    # =========================================================================

    async def update_lead_fields(
        self,
        lead_id: str,
        fields: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Update custom fields on a lead record with survey answers.

        VICIdial custom fields typically map to user-defined columns.
        The exact field names depend on your VICIdial list configuration.

        Args:
            lead_id: VICIdial lead ID
            fields:  Dict of field_name → value to update
                     e.g. {"age": "35", "state": "Maharashtra", "q1": "Satisfied"}

        Common VICIdial lead fields:
            first_name, last_name, phone_number, email, address1, city,
            state, postal_code, comments, rank, owner
        Plus any custom fields defined in the list.
        """
        params = {
            **self._base_non_agent_params(),
            "function": "update_lead",
            "lead_id": lead_id,
        }
        # Merge survey fields — VICIdial accepts them as flat params
        params.update(fields)

        logger.info(f"📝 Updating lead fields: lead={lead_id} fields={list(fields.keys())}")
        return await self._post(self._non_agent_api_url(), params)

    # =========================================================================
    # 3. CALL HANGUP  (programmatically end an active call)
    # =========================================================================

    async def hangup_call(
        self,
        *,
        lead_id: str = None,
        uniqueid: str = None,
        agent_user: str = None,
    ) -> Dict[str, Any]:
        """
        Programmatically hang up an active call.

        Uses Agent API: function=external_hangup

        The VICIdial team will confirm which identifier to use.
        Typically it's the agent_user (the AI agent's login name).

        Args:
            lead_id:    VICIdial lead ID (if hangup is by lead)
            uniqueid:   VICIdial call uniqueid (if hangup is by call ID)
            agent_user: Agent username to hangup (defaults to our AI agent)
        """
        params = {
            **self._base_agent_params(),
            "function": "external_hangup",
            "agent_user": agent_user or self.agent_user,
            "value": "1",  # 1 = hangup
        }
        if lead_id:
            params["lead_id"] = lead_id
        if uniqueid:
            params["uniqueid"] = uniqueid

        logger.info(f"📞 Hanging up call: lead={lead_id} uniqueid={uniqueid}")
        return await self._post(self._agent_api_url(), params)

    # =========================================================================
    # 4. CALL STATUS CHECK
    # =========================================================================

    async def get_agent_status(
        self,
        agent_user: str = None,
    ) -> Dict[str, Any]:
        """
        Check the current status of our AI agent in VICIdial.
        Useful for health checks and debugging.
        """
        params = {
            **self._base_agent_params(),
            "function": "external_status",
            "agent_user": agent_user or self.agent_user,
        }
        return await self._post(self._agent_api_url(), params)

    # =========================================================================
    # 5. CAMPAIGN MANAGEMENT  (Phase 2)
    # =========================================================================

    async def add_lead(
        self,
        phone_number: str,
        *,
        campaign_id: str = None,
        list_id: str = None,
        first_name: str = "",
        last_name: str = "",
        custom_fields: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a single lead to a VICIdial campaign list.

        Args:
            phone_number: Lead's phone number (will be normalized)
            campaign_id:  Target campaign ID
            list_id:      Target list ID within the campaign
            first_name:   Lead first name (optional)
            last_name:    Lead last name (optional)
            custom_fields: Additional custom fields
        """
        phone = self._normalize_number(phone_number)

        params = {
            **self._base_non_agent_params(),
            "function": "add_lead",
            "phone_number": phone,
            "phone_code": "91",  # India country code
        }
        if campaign_id:
            params["campaign_id"] = campaign_id
        if list_id:
            params["list_id"] = list_id
        if first_name:
            params["first_name"] = first_name
        if last_name:
            params["last_name"] = last_name
        if custom_fields:
            params.update(custom_fields)

        logger.info(f"➕ Adding lead: {phone} to campaign={campaign_id}")
        return await self._post(self._non_agent_api_url(), params)

    async def add_leads_batch(
        self,
        leads: List[Dict[str, str]],
        campaign_id: str = None,
        list_id: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Upload multiple leads. VICIdial processes them one at a time.

        Args:
            leads: List of dicts, each with at least "phone_number"
                   Optional: "first_name", "last_name", plus custom fields
            campaign_id: Target campaign ID
            list_id: Target list ID
        """
        import asyncio

        results = []
        for lead in leads:
            result = await self.add_lead(
                phone_number=lead["phone_number"],
                campaign_id=campaign_id,
                list_id=list_id,
                first_name=lead.get("first_name", ""),
                last_name=lead.get("last_name", ""),
                custom_fields={
                    k: v for k, v in lead.items()
                    if k not in ("phone_number", "first_name", "last_name")
                },
            )
            results.append(result)
            await asyncio.sleep(0.1)  # rate limit courtesy

        logger.info(f"📊 Batch upload: {len(results)}/{len(leads)} leads processed")
        return results

    async def campaign_control(
        self,
        campaign_id: str,
        action: str,
    ) -> Dict[str, Any]:
        """
        Start/pause/stop a VICIdial campaign. (Phase 2)

        Args:
            campaign_id: Campaign to control
            action: "start" | "pause" | "stop"

        NOTE: Exact API function TBD — depends on VICIdial version
        and whether they expose campaign control via API.
        Some installations require direct DB access for this.
        """
        # This is a placeholder — VICIdial's standard API doesn't
        # have a clean campaign start/stop endpoint. It's usually done
        # via the admin web UI or direct MySQL updates.
        # The VICIdial team will confirm if they have a custom endpoint.
        logger.warning(
            f"⚠️ campaign_control({campaign_id}, {action}) — "
            "endpoint TBD, awaiting VICIdial team confirmation"
        )
        return {
            "success": False,
            "error": "Campaign control endpoint not yet configured",
            "campaign_id": campaign_id,
            "action": action,
        }

    # =========================================================================
    # UTILITIES
    # =========================================================================

    @staticmethod
    def _normalize_number(number: str) -> str:
        """Strip to digits, remove country code prefix for VICIdial."""
        number = number.strip().replace(" ", "").replace("-", "").replace("+", "")
        # VICIdial typically stores 10-digit local numbers
        if number.startswith("91") and len(number) == 12:
            number = number[2:]
        if number.startswith("0") and len(number) == 11:
            number = number[1:]
        return number


# =============================================================================
# MODULE-LEVEL CONVENIENCE INSTANCE
# =============================================================================

# Create a default client that other modules can import directly:
#   from telephony.vicidial_client import vicidial
#   await vicidial.update_disposition(lead_id="123", status="SVYCMP")
vicidial = VicidialClient()
