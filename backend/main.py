"""
backend/main.py

FastAPI application entry point for the Dynamic Survey Campaign Platform.

Run (from the backend/ directory):
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from database.db import init_db, AsyncSessionLocal
from database.models import User
from routers.admin import router as admin_router
from routers.auth import router as auth_router
from routers.voice import router as voice_router
from utils.security import hash_password

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_NAME = os.environ.get("PROJECT_NAME", "Dynamic Survey Campaign Platform")
VERSION = os.environ.get("APP_VERSION", "2.0.0")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

AUTO_MIGRATE = os.environ.get("AUTO_MIGRATE", "true").lower() == "true"

SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))

# Default admin account (created only if no users exist)
DEFAULT_ADMIN_EMAIL = os.environ.get("DEFAULT_ADMIN_EMAIL", "admin@example.com")
DEFAULT_ADMIN_PASSWORD = os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin123")
DEFAULT_ADMIN_NAME = os.environ.get("DEFAULT_ADMIN_NAME", "Admin")


# ---------------------------------------------------------------------------
# Create default admin if DB empty
# ---------------------------------------------------------------------------
async def seed_admin():
    from sqlalchemy.future import select

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).limit(1))
        existing = result.scalar_one_or_none()

        if existing:
            return

        admin = User(
            name=DEFAULT_ADMIN_NAME,
            email=DEFAULT_ADMIN_EMAIL,
            password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
            role="admin",
        )

        db.add(admin)
        await db.commit()

        logger.info(
            "Default admin created: %s / %s  <-- CHANGE THIS PASSWORD",
            DEFAULT_ADMIN_EMAIL,
            DEFAULT_ADMIN_PASSWORD,
        )


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting %s v%s", PROJECT_NAME, VERSION)

    if AUTO_MIGRATE:
        await init_db()
        logger.info("Database tables verified / created.")
        await seed_admin()

    yield

    logger.info("Application shutting down.")


# ---------------------------------------------------------------------------
# Create FastAPI App
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:

    app = FastAPI(
        title=PROJECT_NAME,
        version=VERSION,
        description="Dynamic Survey Campaign Platform",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # CORS
    # -----------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Static files
    # -----------------------------------------------------------------------
    base_dir = os.path.join(os.path.dirname(__file__), "..")

    static_dir = os.path.join(base_dir, "static")

    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # -----------------------------------------------------------------------
    # Routers
    # -----------------------------------------------------------------------
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(voice_router)

    # -----------------------------------------------------------------------
    # HTML helper
    # -----------------------------------------------------------------------
    def read_html(filename: str) -> str:
        path = os.path.join(base_dir, filename)

        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"<h1>{filename} not found</h1>"

    # -----------------------------------------------------------------------
    # UI Routes
    # -----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def root():
        return read_html("login.html")

    @app.get("/login", response_class=HTMLResponse)
    @app.get("/login.html", response_class=HTMLResponse)
    async def login():
        return read_html("login.html")

    @app.get("/admin", response_class=HTMLResponse)
    @app.get("/admin.html", response_class=HTMLResponse)
    async def admin_ui():
        return read_html("admin.html")

    @app.get("/client", response_class=HTMLResponse)
    @app.get("/client.html", response_class=HTMLResponse)
    async def client_ui():
        return read_html("client.html")

    @app.get("/dashboard", response_class=HTMLResponse)
    @app.get("/survey_dashboard.html", response_class=HTMLResponse)
    async def dashboard():
        return read_html("survey_dashboard.html")

    # -----------------------------------------------------------------------
    # Dashboard API endpoints  (/api/ai  and  /api/survey-data)
    # These were previously only in application.py; added here so the
    # survey_dashboard.html works with the new main.py entry point.
    # -----------------------------------------------------------------------

    def _get_vertex_model():
        """Return the best available Vertex AI GenerativeModel."""
        import vertexai
        from vertexai.preview.generative_models import GenerativeModel
        _vproject  = os.environ.get("VERTEX_PROJECT_ID", "")
        _vlocation = os.environ.get("VERTEX_LOCATION", "us-central1")
        if not _vproject:
            try:
                import config as _cfg
                _vproject  = getattr(_cfg, "VERTEX_PROJECT_ID", "")
                _vlocation = getattr(_cfg, "VERTEX_LOCATION", "us-central1")
            except Exception:
                pass
        vertexai.init(project=_vproject, location=_vlocation)
        for mid in ["gemini-2.0-flash-001", "gemini-2.0-flash",
                    "gemini-1.5-flash-002", "gemini-1.5-flash-001", "gemini-1.5-flash"]:
            try:
                m = GenerativeModel(mid)
                logger.info("Dashboard AI: using model %s", mid)
                return m
            except Exception as e:
                logger.warning("Model %s unavailable: %s", mid, e)
        raise RuntimeError("No Vertex AI model available")

    @app.post("/api/ai")
    async def ai_endpoint(request: Request):
        """
        LLM endpoint used by the analytics dashboard.
        Body: { "system": "...", "prompt": "..." }
        Returns: { "text": "..." }
        """
        try:
            body        = await request.json()
            system_text = body.get("system", "")
            user_prompt = body.get("prompt", "")
            full_prompt = f"{system_text}\n\n{user_prompt}" if system_text else user_prompt
            model       = _get_vertex_model()
            response    = model.generate_content(full_prompt)
            text        = response.candidates[0].content.parts[0].text.strip()
            return JSONResponse({"text": text})
        except Exception as e:
            logger.exception("AI endpoint error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/survey-data")
    async def get_survey_data():
        """
        Legacy survey-data endpoint used by the dashboard as a fallback.
        Reads from the DB responses table (works without Google Sheets).
        Returns rows in the same shape the dashboard expects.
        """
        try:
            from sqlalchemy.future import select as sa_select
            from database.db import AsyncSessionLocal
            from database.models import Response as ResponseModel, Campaign, Question

            async with AsyncSessionLocal() as db:
                # Get all responses with their campaign questions
                resp_result = await db.execute(
                    sa_select(ResponseModel).order_by(ResponseModel.created_at.desc())
                )
                responses = resp_result.scalars().all()

                if not responses:
                    return JSONResponse({"data": [], "count": 0})

                # Build flat rows — one per response
                rows = []
                for r in responses:
                    campaign = await db.get(Campaign, r.campaign_id)
                    answers  = r.structured_answers or {}

                    # Get questions for column names
                    q_result = await db.execute(
                        sa_select(Question)
                        .where(Question.campaign_id == r.campaign_id)
                        .order_by(Question.question_order)
                    )
                    questions = q_result.scalars().all()

                    row = {
                        "Call_id":          r.call_id,
                        "Gender":           r.detected_gender or "Unknown",
                        "Campaign":         campaign.name if campaign else r.campaign_id,
                        "Created_at":       r.created_at.isoformat() if r.created_at else "",
                    }
                    for q in questions:
                        col = f"Q{q.question_order} - {q.question_text[:60]}"
                        row[col] = answers.get(str(q.id), "No Response")

                    rows.append(row)

                return JSONResponse({"data": rows, "count": len(rows)})
        except Exception as e:
            logger.exception("Survey data fetch error: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

    # -----------------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {
            "service": PROJECT_NAME,
            "version": VERSION,
            "status": "healthy",
        }

    return app


app = create_app()

# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
    )