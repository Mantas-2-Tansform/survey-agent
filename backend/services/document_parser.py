"""
services/document_parser.py

Parse survey questions from uploaded documents (PDF, DOCX, CSV, XLSX, TXT, images).

Three-stage pipeline:
  1. Rich text extraction  — format-specific, table-aware, structure-preserving
  2. Structure normalisation — converts tables/grids to human-readable markdown
  3. LLM structuring       — Vertex AI → JSON list of questions

Key improvements over v1:
  • PDF: pdfplumber table extraction preserves grid structure; falls back to
    layout-aware text if no tables found.
  • DOCX: reads both paragraphs AND tables (python-docx Table API).
  • Images (PNG/JPG/WEBP): sends raw image bytes to Gemini multimodal for OCR.
  • LLM prompt: understands grid questions (one parent + N sub-questions),
    custom scale ranges (e.g. 1=Daily ... 4=Never, 8=NR), and NR/CS codes.
  • Produces one Question row per sub-item in a grid so the DB stores each
    sub-question individually (e.g. Q1a, Q1b, Q1c, Q1d).
"""
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vertex AI
# ---------------------------------------------------------------------------
_VERTEX_PROJECT: str = ""
_VERTEX_LOCATION: str = ""
_VERTEX_CANDIDATE_MODELS: List[str] = [
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash",
]


def _configure_vertex(project: str, location: str) -> None:
    global _VERTEX_PROJECT, _VERTEX_LOCATION
    _VERTEX_PROJECT = project
    _VERTEX_LOCATION = location


def _get_model() -> GenerativeModel:
    vertexai.init(project=_VERTEX_PROJECT, location=_VERTEX_LOCATION)
    for model_id in _VERTEX_CANDIDATE_MODELS:
        try:
            m = GenerativeModel(model_id)
            logger.info("DocumentParser: using model %s", model_id)
            return m
        except Exception as e:
            logger.warning("Model %s unavailable: %s", model_id, e)
    raise RuntimeError("No Vertex AI model available for document parsing")


# ---------------------------------------------------------------------------
# LLM extraction prompt  (v2 — table / grid / sub-question aware)
# ---------------------------------------------------------------------------
_EXTRACTION_SYSTEM_PROMPT = """
You are an expert survey digitisation assistant. Extract every survey question
from the document and return them as a strict JSON array.

QUESTION TYPES (use exactly one):
  "scale"           - A rating/frequency/Likert scale with labelled integer values.
                      Put ALL scale values in options as "N=Label" strings.
                      Example: ["1=Daily","2=Sometimes","3=Rarely","4=Never","8=No response"]
  "multiple_choice" - Discrete named options, respondent picks one.
                      Options are plain label strings, no number prefix.
                      Example: ["Development only for Rich","Development for all","No development at all","No response"]
  "yes_no"          - Strictly binary Yes/No question. options=[].
  "numeric"         - Pure number answer (age, count, etc.). options=[].
  "open_text"       - Free-form text answer. options=[].

GRID / TABLE QUESTIONS (MOST IMPORTANT RULE):
When a parent question has a list of sub-items all sharing the same response
scale (a grid/matrix question), emit ONE separate question object per sub-item.
- The sub-item's question_text must be self-contained and include context from
  the parent stem so it reads as a standalone question.
- Append the sub-item label in parentheses: (Q1a), (Q1b), etc.
- All sub-items inherit the same options from the parent scale.

EXAMPLE:
  Source: "Q1. How regularly do you do the following - daily, sometimes, rarely or never?
             a. Watch news on television?  b. Listen to news on radio?  c. Read newspaper/s?"
  Scale columns: Daily=1, Sometimes=2, Rarely=3, Never=4, No response (NR)=8

  -> Emit 3 questions:
  {"question_number":1,"question_text":"How regularly do you watch news on television? (Q1a)","question_type":"scale","options":["1=Daily","2=Sometimes","3=Rarely","4=Never","8=No response"]}
  {"question_number":2,"question_text":"How regularly do you listen to news on radio? (Q1b)","question_type":"scale","options":["1=Daily","2=Sometimes","3=Rarely","4=Never","8=No response"]}
  {"question_number":3,"question_text":"How regularly do you read the newspaper/s? (Q1c)","question_type":"scale","options":["1=Daily","2=Sometimes","3=Rarely","4=Never","8=No response"]}

NR / CS / DK CODES:
- "8. NR" or "98. NR" or "98. NR/CS" or "99. DK" are valid response codes.
- For scale questions: add the NR code as the last option, e.g. "8=No response".
- For multiple_choice: add "No response" as the last plain-string option.
- Never drop these codes - they are valid answer choices.

CODEBOOK REFERENCES:
- "Note answer & consult codebook" / "consult party codes in codebook" means
  the answer is free-form text. Use question_type="open_text", options=[].

OUTPUT FORMAT:
Return ONLY a valid JSON array, no markdown fences, no explanation.
Each element:
{
  "question_number": <integer, sequential from 1 — renumber after expansion>,
  "question_text":   <complete self-contained question text>,
  "question_type":   <scale | multiple_choice | yes_no | numeric | open_text>,
  "options":         <array of strings — required for scale and multiple_choice>
}

RULES:
1. question_number must be sequential integers starting at 1.
2. question_text must be fully self-contained — never rely on another row for context.
3. Never invent questions not present in the document.
4. Never merge distinct sub-items into a single question row.
5. scale options: always "N=Label" format, ordered by numeric value.
6. multiple_choice options: plain label strings, no numeric prefix.
7. options=[] for yes_no, numeric, open_text.
""".strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _llm_structure(
    raw_text: str,
    extraction_hint: str = "",
    image_parts: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Send extracted text (and optionally image parts) to Vertex AI.
    Returns structured question list with sequential numbering.
    """
    model = _get_model()

    hint_block = (
        f"\n\nADMIN INSTRUCTIONS:\n{extraction_hint.strip()}"
        if extraction_hint.strip() else ""
    )

    if image_parts:
        # Multimodal: images + instruction text
        contents = image_parts + [
            f"{_EXTRACTION_SYSTEM_PROMPT}{hint_block}\n\n"
            "The images above show the survey document. "
            "Extract every question exactly as shown, paying close attention to "
            "table structures, grid questions, and all sub-items.\n"
            "Return ONLY the JSON array."
        ]
        response = model.generate_content(
            contents,
            generation_config={"temperature": 0.0},
        )
    else:
        prompt = (
            f"{_EXTRACTION_SYSTEM_PROMPT}{hint_block}\n\n"
            f"---\nDocument text (tables shown as markdown):\n{raw_text[:22000]}\n\n"
            "Return ONLY the JSON array."
        )
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.0},
        )

    text = response.candidates[0].content.parts[0].text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"LLM returned unexpected structure: {type(data)}")

    # Enforce sequential numbering (LLM can repeat numbers after grid expansion)
    for i, q in enumerate(data, start=1):
        q["question_number"] = i

    return data


# ---------------------------------------------------------------------------
# Table -> markdown helper
# ---------------------------------------------------------------------------

def _table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """
    Convert a 2-D table to a markdown table string.
    The LLM reads this structure reliably for grid question detection.
    """
    if not headers and not rows:
        return ""

    col_count = max(
        len(headers),
        max((len(r) for r in rows), default=0)
    )
    h = list(headers) + [""] * (col_count - len(headers))
    lines = ["| " + " | ".join(str(c).strip() for c in h) + " |"]
    lines.append("|" + "|".join(" --- " for _ in h) + "|")
    for row in rows:
        padded = list(row) + [""] * (col_count - len(row))
        lines.append("| " + " | ".join(str(c or "").strip() for c in padded) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF extraction (table-aware)
# ---------------------------------------------------------------------------

def _extract_pdf(content: bytes) -> str:
    """
    Extract text from PDF preserving table structure.

    Per-page strategy:
      1. Find and extract all tables -> convert to markdown [TABLE]...[/TABLE] blocks.
      2. Extract non-table text with layout-preservation.
      3. Stitch in vertical order.
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    page_texts: List[str] = []

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            sections: List[Tuple[float, str]] = []
            table_bboxes: List[Any] = []

            # -- Extract tables --
            try:
                tables = page.find_tables()
            except Exception:
                tables = []

            for table in tables:
                try:
                    extracted = table.extract()
                    if not extracted:
                        continue

                    def clean(c: Any) -> str:
                        return str(c or "").strip().replace("\n", " ")

                    all_rows = [[clean(c) for c in row] for row in extracted]

                    # Heuristic: if first row cells look like column headers
                    # (short, text-ish, not starting with digits used as data values),
                    # treat as header row.
                    header_row = all_rows[0] if all_rows else []
                    data_rows = all_rows[1:] if len(all_rows) > 1 else []

                    md = _table_to_markdown(header_row, data_rows)
                    y_top = table.bbox[1] if table.bbox else page_num * 1000
                    sections.append((y_top, f"\n[TABLE]\n{md}\n[/TABLE]\n"))
                    table_bboxes.append(table.bbox)

                except Exception as e:
                    logger.warning("Page %d: table error: %s", page_num, e)

            # -- Extract non-table text --
            try:
                non_table_page = page
                for bbox in table_bboxes:
                    try:
                        non_table_page = non_table_page.outside_bbox(bbox)
                    except Exception:
                        pass

                # layout=True preserves column spacing which helps with
                # inline scale labels like "Daily  Sometimes  Rarely  Never"
                text = non_table_page.extract_text(layout=True) or ""
                if not text.strip():
                    text = non_table_page.extract_text() or ""
                if text.strip():
                    sections.append((0.0, text))

            except Exception as e:
                try:
                    text = page.extract_text() or ""
                    if text.strip():
                        sections.append((0.0, text))
                except Exception:
                    logger.warning("Page %d: text extraction error: %s", page_num, e)

            sections.sort(key=lambda x: x[0])
            page_text = "\n".join(s[1] for s in sections)
            if page_text.strip():
                page_texts.append(f"--- Page {page_num} ---\n{page_text}")

    return "\n\n".join(page_texts)


# ---------------------------------------------------------------------------
# DOCX extraction (paragraphs + tables)
# ---------------------------------------------------------------------------

def _extract_docx(content: bytes) -> str:
    """
    Extract text from DOCX preserving paragraph order and table structure.
    Iterates the raw XML body so paragraphs and tables stay interleaved.
    """
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")

    doc = Document(io.BytesIO(content))
    blocks: List[str] = []

    _W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def get_cell_text(tc: Any) -> str:
        parts = []
        for p in tc.findall(f".//{{{_W_NS}}}p"):
            text = "".join(
                node.text or ""
                for node in p.iter()
                if node.tag.endswith("}t")
            )
            if text.strip():
                parts.append(text.strip())
        return " ".join(parts)

    for child in doc.element.body:
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if local == "p":
            text = "".join(
                node.text or ""
                for node in child.iter()
                if node.tag.endswith("}t")
            ).strip()
            if text:
                blocks.append(text)

        elif local == "tbl":
            table_rows: List[List[str]] = []
            for tr in child.findall(f".//{{{_W_NS}}}tr"):
                cells = [
                    get_cell_text(tc)
                    for tc in tr.findall(f".//{{{_W_NS}}}tc")
                ]
                if cells:
                    table_rows.append(cells)

            if table_rows:
                md = _table_to_markdown(table_rows[0], table_rows[1:])
                blocks.append(f"\n[TABLE]\n{md}\n[/TABLE]\n")

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Image extraction (multimodal OCR)
# ---------------------------------------------------------------------------

def _extract_image(content: bytes, filename: str) -> Tuple[str, List[Any]]:
    """
    For image uploads (PNG/JPG/WEBP/GIF), return the image as a Gemini Part
    for multimodal processing.  Text extraction is empty — the image itself
    is passed directly to the LLM.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    mime_map = {
        "png":  "image/png",
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif":  "image/gif",
    }
    mime = mime_map.get(ext, "image/png")
    image_part = Part.from_data(data=content, mime_type=mime)
    return "", [image_part]


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def _extract_txt(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _extract_csv(content: bytes) -> str:
    import csv
    rows: List[str] = []
    reader = csv.reader(io.StringIO(content.decode("utf-8", errors="replace")))
    all_rows = list(reader)
    if all_rows:
        md = _table_to_markdown(all_rows[0], all_rows[1:])
        return f"[TABLE]\n{md}\n[/TABLE]"
    return ""


# ---------------------------------------------------------------------------
# XLSX (sheet -> markdown tables)
# ---------------------------------------------------------------------------

def _extract_xlsx(content: bytes) -> str:
    try:
        import openpyxl
    except ImportError:
        raise RuntimeError("openpyxl not installed. Run: pip install openpyxl")

    wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
    blocks: List[str] = []

    for sheet in wb.worksheets:
        rows: List[List[str]] = []
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c or "").strip() for c in row]
            if any(cells):
                rows.append(cells)

        if not rows:
            continue

        md = _table_to_markdown(rows[0], rows[1:])
        blocks.append(f"Sheet: {sheet.title}\n[TABLE]\n{md}\n[/TABLE]")

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# File type router
# ---------------------------------------------------------------------------

_TEXT_EXTRACTORS: Dict[str, Any] = {
    "txt":  _extract_txt,
    "text": _extract_txt,
    "pdf":  _extract_pdf,
    "docx": _extract_docx,
    "doc":  _extract_docx,
    "csv":  _extract_csv,
    "xlsx": _extract_xlsx,
    "xls":  _extract_xlsx,
}

_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_document(
    file_content: bytes,
    filename: str,
    vertex_project: str = "",
    vertex_location: str = "",
    extraction_hint: str = "",
) -> List[Dict[str, Any]]:
    """
    Parse a survey document and return structured question dicts.

    Supports: PDF, DOCX, CSV, XLSX, TXT, PNG, JPG, WEBP, GIF

    Each returned dict:
        {
            "question_number": int,       # sequential from 1
            "question_text":   str,       # fully self-contained question text
            "question_type":   str,       # scale | multiple_choice | yes_no | numeric | open_text
            "options":         list[str], # [] when not applicable
        }

    For scale questions, options use "N=Label" format:
        ["1=Daily", "2=Sometimes", "3=Rarely", "4=Never", "8=No response"]

    Grid/matrix questions (one parent + N sub-items) are expanded into N
    individual question rows, each with a fully self-contained question_text.

    Raises ValueError  if file type is unsupported or document is empty.
    Raises RuntimeError if a required dependency is missing.
    """
    if vertex_project:
        _configure_vertex(vertex_project, vertex_location)

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    image_parts: List[Any] = []
    raw_text = ""

    if ext in _IMAGE_EXTENSIONS:
        logger.info("Image detected — using multimodal OCR: %s", filename)
        raw_text, image_parts = _extract_image(file_content, filename)

    elif ext in _TEXT_EXTRACTORS:
        extractor = _TEXT_EXTRACTORS[ext]
        logger.info("Extracting structured text from %s (type=%s)", filename, ext)
        raw_text = extractor(file_content)

        if not raw_text.strip():
            raise ValueError(
                "Document appears to be empty — no text or tables could be extracted."
            )
        logger.info(
            "Extracted %d chars (tables preserved as markdown)", len(raw_text)
        )

    else:
        supported = sorted(set(_TEXT_EXTRACTORS.keys()) | _IMAGE_EXTENSIONS)
        raise ValueError(
            f"Unsupported file type '.{ext}'. Supported: {', '.join(supported)}"
        )

    logger.info("Sending to LLM for question structuring…")
    questions = _llm_structure(
        raw_text,
        extraction_hint=extraction_hint,
        image_parts=image_parts if image_parts else None,
    )
    logger.info(
        "Structured %d questions (after grid expansion)", len(questions)
    )
    return questions