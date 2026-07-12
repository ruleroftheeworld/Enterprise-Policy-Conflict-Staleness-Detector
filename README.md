# Enterprise Policy Conflict & Staleness Detector

An enterprise policy intelligence platform that ingests policy documents, extracts normalized obligations, builds policy relationships, detects governance risks, and presents prioritized findings through an interactive web dashboard.

The platform combines deterministic policy analysis, semantic similarity, NetworkX-based relationship analysis, optional local LLM verification, and observability metrics. AI verification is fail-safe and optional: the core platform runs without Ollama or any local model.

## Key Features

* Upload and analyze multiple enterprise policy documents.
* Support `.pdf`, `.docx`, `.txt`, and `.md` policy files.
* Parse policy metadata and document sections.
* Extract normalized obligations including:

  * subject
  * action
  * object
  * modality
  * topic
  * scope
  * frequency
  * condition
  * exception
  * technology references
* Generate semantic embeddings for policy obligations.
* Build candidate obligation pairs for deterministic analysis.
* Detect:

  * direct contradictions
  * frequency mismatches
  * modality inconsistencies
  * redundant requirements
  * stale policies
  * stale technology or regulatory references
* Build and analyze policy relationships using NetworkX.
* Assign deterministic confidence and severity scores.
* Optionally verify findings using a local Ollama model.
* Continue operating safely when Ollama is unavailable.
* Expose AI verification observability metrics.
* Analyze policies through a FastAPI REST API.
* Review policies, obligations, findings, evidence, severity, confidence, and verification status through a React dashboard.
* Validate engine output against supplied ground-truth datasets.

## Architecture

```text
                    Policy Documents
               (.pdf / .docx / .txt / .md)
                           |
                           v
                 Ingestion and Parsing
                           |
                           v
                  Policy Normalization
                           |
                           v
                 Obligation Extraction
                           |
                           v
                 Semantic Embeddings
                           |
                           v
                Candidate Pair Generation
                           |
                           v
              Deterministic Policy Detection
                           |
                           v
               NetworkX Relationship Analysis
                           |
                           v
             Optional Local LLM Verification
                      (Ollama)
                           |
                           v
                    AnalysisResult
                           |
                  +--------+--------+
                  |                 |
                  v                 v
           FastAPI REST API    Validation Scripts
                  |
                  v
        React + TypeScript Dashboard
```

The backend isolates the policy engine behind an application service and adapter boundary. The engine can also be invoked directly:

```python
from engine.pipeline import analyze_policy_documents
```

## Tech Stack

### Frontend

* React
* TypeScript
* Vite
* CSS

### Backend

* FastAPI
* Pydantic
* Uvicorn
* Pytest

### Policy Intelligence Engine

* Python
* Sentence Transformers
* PyTorch
* scikit-learn
* Semantic embeddings
* NetworkX
* Deterministic policy analysis
* Optional local LLM verification with Ollama

### Supported Document Formats

* Markdown
* Plain text
* PDF
* DOCX

## Project Structure

```text
policy-intelligence-platform/
|-- backend/
|   |-- app/
|   `-- tests/
|-- docs/
|-- engine/
|   |-- detection/
|   |-- extraction/
|   |-- ingestion/
|   `-- pipeline.py
|-- examples/
|-- frontend/
|   |-- public/
|   `-- src/
|-- sample_data/
|   `-- problem_11/
|       |-- policies/
|       |-- findings_labels.csv
|       |-- obligation_extracts_labels.csv
|       `-- policy_metadata.csv
|-- scripts/
|   |-- validate_against_labels.py
|   |-- validate_with_llm.py
|   `-- diagnostic utilities
|-- shared/
|   `-- contracts/
|-- tests/
|   `-- engine/
|-- README.md
`-- requirements.txt
```

## Prerequisites

Install the following before running the platform:

* Python 3.11 or newer
* Node.js compatible with the installed Vite version
* npm

Ollama is optional and is required only when local LLM verification is enabled.

## Setup

Clone the repository and enter the project directory:

```powershell
git clone <repository-url>
Set-Location policy-intelligence-platform
```

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install engine dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install backend dependencies:

```powershell
python -m pip install -r backend\requirements.txt
```

Install frontend dependencies:

```powershell
npm --prefix frontend install
```

## Run the Backend

From the repository root:

```powershell
python -m uvicorn backend.app.main:app --reload --port 8000
```

Backend API:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## Run the Frontend

From the repository root:

```powershell
npm --prefix frontend run dev
```

Frontend dashboard:

```text
http://localhost:5173
```

## Run Tests

Run the complete Python test suite:

```powershell
python -m pytest -q
```

Run backend tests only:

```powershell
python -m pytest backend\tests -q
```

Run engine extraction tests:

```powershell
python -m pytest tests\engine\test_extraction.py -q
```

Run frontend lint checks:

```powershell
npm --prefix frontend run lint
```

Build the frontend for production:

```powershell
npm --prefix frontend run build
```

## Validation Dataset

The repository includes a labeled policy dataset under:

```text
sample_data/problem_11/
```

The dataset contains 30 policy documents together with:

* policy metadata labels
* obligation extraction labels
* expected finding labels

Run deterministic validation:

```powershell
python scripts\validate_against_labels.py
```

Additional diagnostic scripts are available under `scripts/` for inspecting extraction coverage, candidate generation, redundancy behavior, label matching, and validation misses.

The validation scripts are diagnostic and evaluation utilities. Benchmark metrics may expose known gaps in deterministic detection and should not be interpreted as production accuracy guarantees.

## Optional Local LLM Verification

The deterministic policy engine does not require Ollama.

When local LLM verification is configured, findings can be passed through the Ollama provider for an additional verification step.

The verification layer is designed to be fail-safe:

* the deterministic pipeline continues when Ollama is unavailable
* verification failures do not prevent policy analysis
* verification results are exposed separately from deterministic findings
* observability metrics track verification attempts and outcomes

Check installed Ollama models:

```powershell
ollama list
```

Run LLM-assisted validation when a compatible model is available:

```powershell
python scripts\validate_with_llm.py
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

Upload one or more policy documents using the multipart form field:

```text
files
```

Supported formats:

```text
.pdf
.docx
.txt
.md
```

The API returns an analysis result containing:

* normalized policies
* extracted obligations
* detected findings
* severity and confidence information
* supporting evidence
* statistics
* warnings
* processing time
* optional AI verification information

## Frontend Configuration

The frontend supports the environment variable:

```text
VITE_API_BASE_URL=http://localhost:8000
```

If the variable is not configured, the frontend uses:

```text
http://localhost:8000
```

## Demo Flow

1. Start the FastAPI backend.
2. Start the React frontend.
3. Open the dashboard.
4. Upload multiple policy documents.
5. Run policy analysis.
6. Review policy and obligation statistics.
7. Inspect prioritized governance findings.
8. Compare source and target policy evidence.
9. Review severity and deterministic confidence scores.
10. Review AI verification telemetry when Ollama verification is enabled.

## Current Limitations

The platform is a hackathon prototype and has known limitations:

* Deterministic contradiction detection requires further calibration for broader policy language patterns.
* Benchmark-level direct-conflict and partial-conflict detection remain areas for improvement.
* Redundancy detection can produce false positives for highly repetitive policy language.
* Stale-reference detection depends on the configured deprecated-technology and regulation rules.
* Obligation extraction is rule-based and may not capture all complex natural-language policy structures.
* Semantic embedding model downloads may be required on first use if the model is not already cached.
* Local LLM verification depends on Ollama availability, installed models, and available compute resources.

## Design Principles

The project follows several core design principles:

* deterministic analysis remains the primary source of findings
* AI verification is optional and fail-safe
* engine contracts are shared across platform components
* backend integration is isolated behind service and adapter boundaries
* findings include evidence for review and explainability
* evaluation scripts and labeled datasets support reproducible validation
* the platform remains runnable without requiring local LLM infrastructure

## Repository Integration

The final integration branch combines work from:

* policy intelligence engine
* local LLM verification
* AI verification observability
* backend API integration
* frontend dashboard
* validation datasets and diagnostic scripts

All feature branches are integrated into `integration/merge-engine-platform` before final delivery verification.
