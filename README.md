# Enterprise Policy Conflict & Staleness Detector

A policy intelligence platform that analyzes enterprise policy documents to identify conflicts, redundancies, stale requirements, and governance risks.

The system combines deterministic policy analysis with optional AI verification and presents prioritized findings through an interactive web dashboard.

## Features

* Upload and analyze multiple policy documents.
* Extract normalized policies and obligations.
* Detect policy conflicts, redundancies, and stale requirements.
* Display finding severity and confidence scores.
* Compare source and target policy evidence.
* View optional AI verification status and observability metrics.
* Run the platform without requiring Ollama.
* Responsive dashboard for reviewing analysis results.

## Architecture

```text
React + TypeScript Frontend
            |
            v
       FastAPI Backend
            |
            v
   Policy Analysis Service
            |
            v
      Policy Engine
            |
            v
       AnalysisResult
            |
            v
      Dashboard Results
```

The backend isolates the policy engine behind an adapter boundary, allowing the platform API to be tested independently while remaining compatible with:

```python
from engine.pipeline import analyze_policy_documents
```

## Tech Stack

**Frontend**

* React
* TypeScript
* Vite
* CSS

**Backend**

* FastAPI
* Pydantic
* Pytest

**Policy Intelligence Engine**

* Python
* Sentence Transformers
* Semantic embeddings
* Deterministic policy analysis
* Optional local LLM verification with Ollama

## Project Structure

```text
Enterprise-Policy-Conflict-Staleness-Detector/
├── backend/
│   ├── app/
│   └── tests/
├── engine/
├── frontend/
│   └── src/
├── shared/
├── README.md
└── requirements.txt
```

## Run the Backend

Install backend dependencies:

```powershell
python -m pip install -r backend\requirements.txt
```

Start the API:

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

The backend runs at `http://localhost:8000`.

API documentation is available at `http://localhost:8000/docs`.

## Run the Frontend

```powershell
Set-Location frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:5173`.

## Run Tests

Backend platform tests:

```powershell
python -m pytest backend\tests -v
```

Frontend checks:

```powershell
Set-Location frontend
npm run lint
npm run build
```

## API

### Health Check

```text
GET /health
```

### Analyze Policy Documents

```text
POST /api/v1/analyses
```

Upload one or more `.pdf`, `.docx`, `.txt`, or `.md` policy documents using the multipart form field `files`.

The API returns policies, obligations, findings, statistics, warnings, and processing time for display in the dashboard.

## Configuration

The frontend optionally supports:

```text
VITE_API_BASE_URL=http://localhost:8000
```

If not configured, the frontend uses `http://localhost:8000`.

The platform itself does not require Ollama. AI verification remains optional.

## Demo Flow

1. Start the FastAPI backend.
2. Start the React frontend.
3. Upload multiple policy documents.
4. Run policy analysis.
5. Review policy and obligation metrics.
6. Inspect prioritized findings.
7. Compare source and target evidence.
8. Review AI verification telemetry when available.
