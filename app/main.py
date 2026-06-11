"""FastAPI application exposing the RFP extraction and proposal pipelines."""

from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agents.drafter import run_drafter
from app.agents.extraction import run_extraction
from app.schemas.rfp import RFPExtraction

ALLOWED_EXTENSIONS = {"pdf", "docx"}

app = FastAPI(
    title="RFP Agent",
    description="An AI-powered pipeline that reads an RFP document and drafts a professional proposal.",
    version="0.1.0",
)


def _validate_upload(file: UploadFile) -> None:
    suffix = (
        file.filename.lower().rsplit(".", 1)[-1]
        if file.filename and "." in file.filename
        else ""
    )
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")


@app.get("/health")
def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


@app.post("/api/v1/extract", response_model=RFPExtraction)
async def extract_endpoint(file: UploadFile = File(...)) -> RFPExtraction:
    """Parse an RFP document and return structured extraction as JSON."""
    _validate_upload(file)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return run_extraction(file.filename, data)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Extraction failed: {exc}") from exc


@app.post("/api/v1/process")
async def process_endpoint(file: UploadFile = File(...)) -> StreamingResponse:
    """Parse an RFP document, draft a proposal, and return it as a .docx download."""
    _validate_upload(file)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        extraction = run_extraction(file.filename, data)
        _, docx_bytes = run_drafter(extraction)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Proposal generation failed: {exc}") from exc

    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=proposal.docx"},
    )
