"""
backend/routers/admin.py

Admin-only endpoints:
  User management  : POST /admin/users, GET /admin/users
  Campaign CRUD    : POST/GET/PATCH/DELETE /admin/campaign(s)
  Question CRUD    : POST/GET/PATCH/DELETE /admin/campaign/{id}/questions
  Doc upload       : POST /admin/campaign/{id}/upload-questions
  Responses        : GET  /admin/campaign/{id}/responses
  Sheet provision  : POST /admin/campaign/{id}/provision-sheet

Fixes from original:
  - db.flush() replaced with db.commit() + db.refresh() so data is
    actually persisted (flush only writes to the current transaction
    without committing, so a crash or rollback would lose the row).
  - UUID route params changed to str to match String(36) model IDs.
  - Sheet provisioning is non-fatal (warns, doesn't 500) when Google
    Sheets is not configured.
"""
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.db import get_db
from database.models import Campaign, Question, Response, User
from services.document_parser import parse_document
from services.question_generator import generate_questions
from services.sheet_service import (
    create_campaign_tab,
    read_campaign_responses,
    sanitise_tab_name,
)
from utils.security import hash_password, require_admin, require_interviewer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

def _cfg(env_var: str, config_attr: str, default: str = "") -> str:
    """
    Always read live — never cached at module level.
    Priority: env var -> config.py attribute -> config._get_secret() -> default.
    """
    val = os.environ.get(env_var, "").strip()
    if val:
        return val
    try:
        import config as _c
        # Try the module-level attribute first (already resolved via Secret Manager at startup)
        val = str(getattr(_c, config_attr, "")).strip()
        if val:
            return val
        # Retry Secret Manager directly in case startup resolution was empty
        val = _c._get_secret(env_var, default).strip()
        if val:
            return val
    except Exception:
        pass
    return default


# Lazy accessors — called at request time so Secret Manager values are never
# missed due to module import ordering.  Do NOT cache these as module constants.
def _vertex_project()  -> str: return _cfg("VERTEX_PROJECT_ID", "VERTEX_PROJECT_ID")
def _vertex_location() -> str: return _cfg("VERTEX_LOCATION",   "VERTEX_LOCATION", "us-central1")
def _sheet_id()        -> str: return _cfg("GOOGLE_SHEET_ID",   "GOOGLE_SHEET_ID")

ENABLE_SHEETS: bool = os.environ.get("ENABLE_SHEETS", "false").lower() == "true"


# ===========================================================================
# Pydantic schemas
# ===========================================================================

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "interviewer"


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    role: str

    class Config:
        from_attributes = True


class CampaignCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    description: Optional[str] = None


class CampaignUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None   # draft | active | closed


class CampaignOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    google_sheet_tab_name: Optional[str]
    status: str
    created_by: str

    class Config:
        from_attributes = True


class LogicRule(BaseModel):
    """
    A single branching rule evaluated after the respondent answers a question.
    condition : exact answer value (e.g. "Yes", "3") OR "default" (catch-all).
    next_order: question_order of the question to jump to.
    """
    condition:  str
    next_order: int


class QuestionCreate(BaseModel):
    question_order: int = Field(..., ge=1)
    question_text: str  = Field(..., min_length=5)
    question_type: str  = Field(
        default="open_text",
        pattern="^(scale|scale_1_5|multiple_choice|yes_no|numeric|open_text)$",
    )
    options: Optional[List[str]]               = None
    validation_rules: Optional[Dict[str, Any]] = None
    required: bool = True
    question_logic: Optional[List[LogicRule]]  = None

    def normalise(self) -> "QuestionCreate":
        """Normalise legacy scale_1_5 to scale with default range [1, 5]."""
        if self.question_type == "scale_1_5":
            object.__setattr__(self, "question_type", "scale")
            if not self.options:
                object.__setattr__(self, "options", ["1", "5"])
        return self


class QuestionUpdate(BaseModel):
    question_order:    Optional[int]               = None
    question_text:     Optional[str]               = None
    question_type:     Optional[str]               = None
    options:           Optional[List[str]]          = None
    validation_rules:  Optional[Dict[str, Any]]    = None
    required:          Optional[bool]              = None
    question_logic:    Optional[List[LogicRule]]   = None


class QuestionOut(BaseModel):
    id: str
    campaign_id: str
    question_order: int
    question_text: str
    question_type: str
    options: Optional[List[str]]
    validation_rules: Optional[Dict[str, Any]]
    required: bool
    question_logic: Optional[List[Dict[str, Any]]] = None

    class Config:
        from_attributes = True


class GenerateQuestionsRequest(BaseModel):
    """Request body for AI-based question generation from a description."""
    description: str  = Field(..., min_length=20)
    num_questions: int = Field(..., ge=1, le=50)
    extra_instructions: Optional[str] = None


# ===========================================================================
# User management
# ===========================================================================

@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Create a new user (admin or interviewer). Admin only."""
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    # FIX: commit() not flush() — flush() alone never persists the row
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/users", response_model=List[UserOut])
async def list_users(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """List all users. Admin only."""
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return result.scalars().all()


# ===========================================================================
# Campaign CRUD
# ===========================================================================

@router.post("/campaign", response_model=CampaignOut, status_code=status.HTTP_201_CREATED)
async def create_campaign(
    payload: CampaignCreate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Create a new campaign. Optionally provisions a Google Sheet tab."""
    tab_name = sanitise_tab_name(payload.name)

    campaign = Campaign(
        name=payload.name,
        description=payload.description,
        created_by=str(admin.id),
        google_sheet_tab_name=tab_name,
        status="draft",
    )
    db.add(campaign)
    # FIX: commit so the campaign is actually saved
    await db.commit()
    await db.refresh(campaign)

    # Provision Google Sheet tab (optional — non-fatal if not configured)
    if ENABLE_SHEETS and _sheet_id():
        try:
            create_campaign_tab(_sheet_id(), tab_name, questions=[])
        except Exception as e:
            logger.warning("Sheet tab creation failed (non-fatal): %s", e)

    return campaign


@router.get("/campaigns/active", response_model=List[CampaignOut])
async def list_active_campaigns(
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(require_interviewer),
):
    """
    Return active campaigns only.
    Accessible by both admins and interviewers — used by client.html
    to populate the campaign selector for the voice survey agent.
    """
    result = await db.execute(
        select(Campaign)
        .where(Campaign.status == "active")
        .order_by(Campaign.created_at.desc())
    )
    return result.scalars().all()


@router.get("/campaigns", response_model=List[CampaignOut])
async def list_campaigns(
    status_filter: Optional[str] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """List all campaigns, optionally filtered by status."""
    q = select(Campaign).order_by(Campaign.created_at.desc())
    if status_filter:
        q = q.where(Campaign.status == status_filter)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/campaign/{campaign_id}", response_model=CampaignOut)
async def get_campaign(
    campaign_id: str,   # FIX: str not UUID
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign


@router.patch("/campaign/{campaign_id}", response_model=CampaignOut)
async def update_campaign(
    campaign_id: str,
    payload: CampaignUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    if payload.name is not None:
        campaign.name = payload.name
        campaign.google_sheet_tab_name = sanitise_tab_name(payload.name)
    if payload.description is not None:
        campaign.description = payload.description
    if payload.status is not None:
        if payload.status not in ("draft", "active", "closed"):
            raise HTTPException(status_code=400, detail="status must be draft | active | closed")
        campaign.status = payload.status

    await db.commit()
    await db.refresh(campaign)
    return campaign


@router.delete("/campaign/{campaign_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_campaign(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    await db.delete(campaign)
    await db.commit()


# ===========================================================================
# Question management
# ===========================================================================

@router.post(
    "/campaign/{campaign_id}/questions",
    response_model=List[QuestionOut],
    status_code=status.HTTP_201_CREATED,
)
async def add_questions(
    campaign_id: str,
    questions: List[QuestionCreate],
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Manually add one or more questions to a campaign."""
    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    created = []
    for q_data in questions:
        q_data = q_data.normalise()   # scale_1_5 → scale with [1,5] range
        q = Question(
            campaign_id=campaign_id,
            question_order=q_data.question_order,
            question_text=q_data.question_text,
            question_type=q_data.question_type,
            options=q_data.options,
            validation_rules=q_data.validation_rules,
            required=q_data.required,
            question_logic=[r.model_dump() for r in q_data.question_logic] if q_data.question_logic else None,
        )
        db.add(q)
        created.append(q)

    # FIX: commit all questions at once
    await db.commit()
    for q in created:
        await db.refresh(q)

    # Re-provision sheet header (non-fatal)
    if ENABLE_SHEETS and _sheet_id() and campaign.google_sheet_tab_name:
        q_dicts = [
            {"question_order": q.question_order, "question_text": q.question_text}
            for q in created
        ]
        try:
            create_campaign_tab(_sheet_id(), campaign.google_sheet_tab_name, q_dicts)
        except Exception as e:
            logger.warning("Sheet header re-provision failed (non-fatal): %s", e)

    return created


@router.get("/campaign/{campaign_id}/questions", response_model=List[QuestionOut])
async def list_questions(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    result = await db.execute(
        select(Question)
        .where(Question.campaign_id == campaign_id)
        .order_by(Question.question_order)
    )
    return result.scalars().all()


@router.patch(
    "/campaign/{campaign_id}/questions/{question_id}",
    response_model=QuestionOut,
)
async def update_question(
    campaign_id: str,
    question_id: str,
    payload: QuestionUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    q = await db.get(Question, question_id)
    if not q or q.campaign_id != campaign_id:
        raise HTTPException(status_code=404, detail="Question not found")

    for field, value in payload.model_dump(exclude_none=True).items():
        setattr(q, field, value)

    await db.commit()
    await db.refresh(q)
    return q


@router.delete(
    "/campaign/{campaign_id}/questions/{question_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_question(
    campaign_id: str,
    question_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    q = await db.get(Question, question_id)
    if not q or q.campaign_id != campaign_id:
        raise HTTPException(status_code=404, detail="Question not found")
    await db.delete(q)
    await db.commit()


# ===========================================================================
# AI question generation from description
# ===========================================================================

@router.post("/campaign/{campaign_id}/generate-questions")
async def generate_questions_from_description(
    campaign_id: str,
    payload: GenerateQuestionsRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """
    Generate survey questions using AI from a plain-English campaign description.

    The admin provides:
      - description      : detailed text describing the survey purpose, target
                           audience, topics to cover, expected outcomes, etc.
      - num_questions    : exact number of questions to generate (1-50).
      - extra_instructions: optional guidance for tone, language, topics to avoid, etc.

    The AI designs well-typed, well-ordered questions and returns them for review.
    Questions are NOT saved automatically — call POST /admin/campaign/{id}/questions
    with the (optionally edited) list to persist them.

    Supports logic/branching: the AI will include branching rules if the description
    mentions conditional flows (e.g. "if they say Yes to Q1, skip to Q4").
    """
    _vp = _vertex_project()
    _vl = _vertex_location()
    logger.info("generate-questions: VERTEX_PROJECT_ID=%r", _vp)
    if not _vp:
        raise HTTPException(
            status_code=503,
            detail="Vertex AI is not configured. Set VERTEX_PROJECT_ID environment variable.",
        )

    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    try:
        generated = generate_questions(
            description=payload.description,
            num_questions=payload.num_questions,
            vertex_project=_vp,
            vertex_location=_vl,
            extra_instructions=payload.extra_instructions or "",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Question generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return {
        "campaign_id": campaign_id,
        "generated_count": len(generated),
        "questions": generated,
        "message": (
            "Review and edit these AI-generated questions, then POST to "
            "/admin/campaign/{id}/questions to save them."
        ),
    }


# ===========================================================================
# Question logic (branching rules) — update independently
# ===========================================================================

@router.patch("/campaign/{campaign_id}/questions/{question_id}/logic")
async def update_question_logic(
    campaign_id: str,
    question_id: str,
    rules: List[LogicRule],
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """
    Set or replace the branching logic rules for a specific question.

    POST an empty list [] to remove all logic and revert to sequential flow.

    Example rules body:
      [
        {"condition": "Yes", "next_order": 5},
        {"condition": "No",  "next_order": 3},
        {"condition": "default", "next_order": 4}
      ]

    "condition" values:
      - Exact answer text ("Yes", "No", "3", "Very Satisfied", etc.)
      - "default"  — fallback if no other condition matches

    "next_order" is the question_order (1-based integer) to jump to.
    To skip to survey end, use a very large number (e.g. 999).
    """
    q = await db.get(Question, question_id)
    if not q or q.campaign_id != campaign_id:
        raise HTTPException(status_code=404, detail="Question not found")

    q.question_logic = [r.model_dump() for r in rules] if rules else None
    await db.commit()
    await db.refresh(q)

    return {
        "question_id": question_id,
        "question_order": q.question_order,
        "question_logic": q.question_logic,
        "message": "Logic rules updated. The voice agent will follow these branches during calls.",
    }


@router.get("/campaign/{campaign_id}/questions/{question_id}/logic")
async def get_question_logic(
    campaign_id: str,
    question_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Return the current branching rules for a question."""
    q = await db.get(Question, question_id)
    if not q or q.campaign_id != campaign_id:
        raise HTTPException(status_code=404, detail="Question not found")
    return {
        "question_id": question_id,
        "question_order": q.question_order,
        "question_text": q.question_text,
        "question_logic": q.question_logic or [],
    }


# ===========================================================================
# Document upload -> question extraction
# ===========================================================================

@router.post("/campaign/{campaign_id}/upload-questions")
async def upload_questions_from_document(
    campaign_id: str,
    file: UploadFile = File(...),
    extraction_hint: str = Form(""),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """
    Upload a PDF / DOCX / CSV / XLSX / TXT file.
    LLM extracts questions -> returned for admin review.
    Optional extraction_hint: plain-English instructions for the LLM about
    what data to extract or how to interpret the document.
    Questions are NOT saved automatically.
    Call POST /admin/campaign/{id}/questions with the edited list to save them.
    """
    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    try:
        extracted = parse_document(
            file_content=content,
            filename=file.filename or "upload",
            vertex_project=_vertex_project(),
            vertex_location=_vertex_location(),
            extraction_hint=extraction_hint,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    return {
        "campaign_id": campaign_id,
        "filename": file.filename,
        "extracted_count": len(extracted),
        "questions": extracted,
        "message": (
            "Review and edit these questions, then POST to "
            "/admin/campaign/{id}/questions to save them."
        ),
    }


# ===========================================================================
# Campaign responses (DB read-back)
# ===========================================================================

@router.get("/campaign/{campaign_id}/responses")
async def get_campaign_responses(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Return all survey responses for a campaign from the database."""
    result = await db.execute(
        select(Response)
        .where(Response.campaign_id == campaign_id)
        .order_by(Response.created_at.desc())
    )
    responses = result.scalars().all()
    return {
        "campaign_id": campaign_id,
        "count": len(responses),
        "responses": [
            {
                "id": str(r.id),
                "call_id": r.call_id,
                "structured_answers": r.structured_answers,
                "detected_gender": r.detected_gender,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in responses
        ],
    }


@router.get("/campaign/{campaign_id}/sheet-data")
async def get_campaign_sheet_data(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """
    Return all rows for a campaign.
    Primary source: Google Sheet (when ENABLE_SHEETS=true and GOOGLE_SHEET_ID is set).
    Fallback: local database responses table (always available).
    """
    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # ── Primary: Google Sheets ──
    if ENABLE_SHEETS and _sheet_id() and campaign.google_sheet_tab_name:
        try:
            data = read_campaign_responses(_sheet_id(), campaign.google_sheet_tab_name)
            if data.get("data"):
                return data
            # Sheet exists but is empty — fall through to DB
        except Exception as e:
            logger.warning("Sheet read failed, falling back to DB: %s", e)

    # ── Fallback: local DB responses ──
    try:
        result = await db.execute(
            select(Response)
            .where(Response.campaign_id == campaign_id)
            .order_by(Response.created_at.desc())
        )
        responses = result.scalars().all()

        if not responses:
            return {"headers": [], "data": [], "count": 0}

        # Load questions for column headers
        q_result = await db.execute(
            select(Question)
            .where(Question.campaign_id == campaign_id)
            .order_by(Question.question_order)
        )
        questions = q_result.scalars().all()

        headers = ["Call_id", "Gender"] + [
            f"Q{q.question_order} - {q.question_text[:60]}" for q in questions
        ] + ["Transcript"]

        rows = []
        for r in responses:
            answers = r.structured_answers or {}
            row = {
                "Call_id": r.call_id,
                "Gender": r.detected_gender or "Unknown",
            }
            for q in questions:
                col = f"Q{q.question_order} - {q.question_text[:60]}"
                row[col] = answers.get(str(q.id), "No Response")
            row["Transcript"] = r.transcript or ""
            rows.append(row)

        return {"headers": headers, "data": rows, "count": len(rows)}
    except Exception as e:
        logger.exception("DB fallback failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Could not load campaign data: {e}")


# ===========================================================================
# Sheet tab provisioning (manual trigger)
# ===========================================================================

@router.post("/campaign/{campaign_id}/provision-sheet")
async def provision_sheet_tab(
    campaign_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Re-create / refresh the Google Sheet tab and header row for a campaign."""
    if not ENABLE_SHEETS or not _sheet_id():
        raise HTTPException(
            status_code=400,
            detail="Google Sheets is not enabled. Set ENABLE_SHEETS=true and GOOGLE_SHEET_ID.",
        )

    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    result = await db.execute(
        select(Question)
        .where(Question.campaign_id == campaign_id)
        .order_by(Question.question_order)
    )
    questions = result.scalars().all()
    q_dicts = [
        {"question_order": q.question_order, "question_text": q.question_text}
        for q in questions
    ]

    tab_name = campaign.google_sheet_tab_name or sanitise_tab_name(campaign.name)
    campaign.google_sheet_tab_name = tab_name

    try:
        create_campaign_tab(_sheet_id(), tab_name, q_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sheet provisioning failed: {e}")

    await db.commit()
    return {"tab_name": tab_name, "columns": len(q_dicts) + 3, "status": "ok"}