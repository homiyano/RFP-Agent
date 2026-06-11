# RFP Agent

An open-source, AI-powered pipeline that reads a Request for Proposal (RFP)
document and drafts a professional proposal — grounded strictly in the
information extracted from the source document.

## How it works

1. **Upload** a PDF or DOCX RFP via the API.
2. **Parse & chunk** — `app/parsers/document.py` extracts text and splits it
   into logical sections using heading heuristics (ALL CAPS lines, lines
   ending in `:`, numbered sections like `1.` / `2.1)`), falling back to
   1000-character chunks if no structure is detected.
3. **Extract** — a LangGraph pipeline (`app/agents/extraction.py`) sends the
   parsed text to GPT-4o with a structured output schema
   (`RFPExtraction`), pulling out company name, project title, deadline,
   budget, requirements, evaluation criteria, a confidence score, and review
   flags.
4. **Validate** — confidence and completeness are checked. Low-confidence or
   incomplete extractions are flagged for human review but never block the
   pipeline.
5. **Draft** — a second LangGraph pipeline (`app/agents/drafter.py`) drafts
   each proposal section (executive summary, technical approach, timeline,
   pricing, team) as its own node, using **only** the extracted JSON as
   context so the output stays grounded in the source RFP.
6. **Render** — `app/templates/proposal.py` renders the drafted sections into
   a downloadable `.docx` proposal, including any review flags as an
   internal notes section.

## API

### `POST /api/v1/extract`

Upload a PDF/DOCX RFP and get back the structured `RFPExtraction` JSON.

```bash
curl -F "file=@rfp.pdf" http://localhost:8000/api/v1/extract
```

### `POST /api/v1/process`

Upload a PDF/DOCX RFP and get back a drafted `proposal.docx`.

```bash
curl -F "file=@rfp.pdf" http://localhost:8000/api/v1/process -o proposal.docx
```

### `GET /health`

Basic liveness check.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env  # add your OPENAI_API_KEY

uvicorn app.main:app --reload
```

The API docs are then available at `http://localhost:8000/docs`.

## Running with Docker

```bash
cp .env.example .env  # add your OPENAI_API_KEY
docker compose up --build
```

## Tests

```bash
pytest
```

Tests mock all LLM calls, so no `OPENAI_API_KEY` is required to run the suite.

## Project structure

```
app/
├── main.py                  # FastAPI app: /api/v1/extract, /api/v1/process
├── agents/
│   ├── extraction.py        # LangGraph: parse -> extract -> validate
│   └── drafter.py            # LangGraph: draft each section -> render
├── parsers/
│   └── document.py          # PDF + DOCX parsing, section-aware chunking
├── schemas/
│   └── rfp.py               # Pydantic models (RFPExtraction, ProposalSection, ...)
└── templates/
    └── proposal.py          # python-docx renderer
tests/
├── test_parser.py
├── test_extraction.py
└── test_drafter.py
```

## License

MIT — see [LICENSE](LICENSE).
