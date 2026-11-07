# VentureIQ — AI-Powered Startup Evaluation & Risk Intelligence Platform

VentureIQ is a complete, industry-level enterprise decision-support SaaS platform for startup founders, venture capitalists, and angel investors. It combines machine learning predictive modelling, evidence-based Retrieval-Augmented Generation (RAG), and large language models (LLMs) to evaluate startup viability, classify systemic risks, analyze early-stage ideas, and generate strategic business roadmaps.

---

## 1. System Architecture

```
                          ┌─────────────────────────────────────┐
                          │     Next.js 14 + React Frontend     │
                          │   (TypeScript, Motion, Recharts)   │
                          └──────────────────┬──────────────────┘
                                             │ HTTP REST / JSON
                                             ▼
                          ┌─────────────────────────────────────┐
                          │        FastAPI REST Backend         │
                          │   (Validation, Orchestration, CORS) │
                          └──────┬───────────┬───────────┬──────┘
                                 │           │           │
            ┌────────────────────┘           │           └────────────────────┐
            ▼                                ▼                                ▼
┌───────────────────────┐        ┌───────────────────────┐        ┌───────────────────────┐
│  Machine Learning     │        │  RAG Knowledge Engine │        │   Groq LLM Client     │
│  - Gradient Boosting  │        │  - 12 Methodology     │        │  - Llama 3.3 70B      │
│  - Scikit-Learn       │        │    Primers            │        │  - Bounded Retries    │
│  - Log-Odds Explainer │        │  - Hybrid BM25/FAISS  │        │  - Offline Fallback   │
│  - Crunchbase Dataset │        │  - Reciprocal Fusion  │        │  - Schema Validation  │
└───────────────────────┘        └───────────────────────┘        └───────────────────────┘
            │                                                            │
            └────────────────────────────┬───────────────────────────────┘
                                         ▼
                          ┌─────────────────────────────────────┐
                          │   SQLite Persistence & Reporting   │
                          │   - Full Analysis History           │
                          │   - Side-by-Side Comparison         │
                          │   - ReportLab PDF & Markdown Export │
                          └─────────────────────────────────────┘
```

---

## 2. Core Modules

1. **Cinematic Brand Opening Animation:**
   - 5–7 second branded intro featuring atmospheric particle grid, intelligence pulse, VentureIQ wordmark reveal, and tagline: *"See the Potential. Understand the Risk."*
   - Plays once per session (`sessionStorage`), includes `[Skip Intro]` control, and respects `prefers-reduced-motion`.
2. **Module 1 — Executive Dashboard:**
   - Real-time KPIs: Total Analyses, Average Viability Score, Low Risk Ratio, Saved Reports.
   - Donut chart showing Risk Distribution (Low Risk 41%, Moderate Risk 33%, High Risk 25%).
   - Line chart showing monthly analysis trends with interactive tooltips.
   - Jump list of recent startup evaluations.
3. **Module 2 — Predict Startup Success:**
   - 4-step wizard: Basic Info → Market & Product → Team & Financials → Review.
   - Validated inputs: funding (USD), rounds, relationships, milestones, VC, Angel, Series A, Series B.
   - Automated business rules (e.g., Series B requires Series A; investor backing implies funding rounds).
   - Instant calculation of viability probability (0–100) and risk classification (Lower, Moderate, Higher).
4. **Module 3 — Explore Startup Idea:**
   - Comprehensive pre-funding idea assessment with prompt input and suggested topic pills.
   - Friendly AI robot intelligence mascot with feature checklist.
   - Two-stage market intelligence generation covering customer pain, buyer personas, and geography.
5. **Module 4 — Market Intelligence:**
   - TAM / SAM / SOM market sizing calculations with explicitly stated methodologies.
   - Competitor mapping with direct vs. indirect identification, moats, and vulnerabilities.
   - Differentiation matrix and unique value proposition synthesis.
6. **Module 5 — AI Risk & Strategy Lab:**
   - 4-quadrant SWOT analysis (Strengths, Weaknesses, Opportunities, Threats).
   - Systematic risk taxonomy across Market, Execution, Financial, and Regulatory domains.
   - Prioritized 30-60-90 day strategic action roadmaps and testable validation experiments.
7. **Module 6 — Analysis Results & Transparency:**
   - Circular radial progress meter, risk badge, and tabbed deep-dive panels.
   - Model feature attribution displaying exact log-odds contributions.
   - Source transparency banner clearly identifying *Live AI Analysis* vs. *Rule-Based Offline Analysis*.
8. **Module 7 — Analysis History & Comparison:**
   - Searchable and filterable history table with risk level and date filters.
   - Side-by-side comparison modal comparing any two evaluated startups.
   - One-click PDF download (via ReportLab) and Markdown export.
9. **Module 8 — Settings & System Status:**
   - Runtime diagnostics table verifying Python, Model SHA-256 integrity, SQLite DB, Groq API, and RAG index.
   - **Model Retraining Studio**: Toggle between the Synthetic Benchmark Dataset and the **Crunchbase Historical Real-World Dataset** (923 startups with acquired vs. closed outcomes).
   - Secure API key manager and intro replay toggle.

---

## 3. Quick Start (Windows One-Click Launch)

Double-click `START_VENTUREIQ.bat` in the project root folder.

The automated launcher will:
1. Verify Python 3.11/3.12 and Node.js are installed.
2. Create and activate `.venv` if not already present.
3. Install required Python and Next.js dependencies.
4. Verify/train the ML model and build the RAG knowledge index.
5. Start the FastAPI backend on `http://127.0.0.1:8000`.
6. Start the Next.js frontend on `http://localhost:3000`.
7. Automatically open your default browser to `http://localhost:3000`.

---

## 4. Manual Setup & Execution

### Prerequisites
- **Python**: 3.11 or 3.12 (from [python.org](https://www.python.org/))
- **Node.js**: v18+ or v20+ LTS (from [nodejs.org](https://nodejs.org/))

### Backend Setup (PowerShell)
```powershell
# 1. Navigate to the project root
cd "C:\Users\user\OneDrive\Desktop\VentureIQ"

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install backend dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Optional: Configure Groq API Key (for Live AI mode)
$env:GROQ_API_KEY = "your_actual_groq_api_key_here"

# 5. Build ML Model and RAG Index
python train_model.py
python scripts\build_index.py

# 6. Start FastAPI REST Backend
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend Setup (Separate PowerShell Terminal)
```powershell
cd "C:\Users\user\OneDrive\Desktop\VentureIQ\frontend"

# Install UI dependencies
npm install

# Start Next.js Development Server
npm run dev
```

Open your browser to **`http://localhost:3000`** (Frontend) or **`http://127.0.0.1:8000/docs`** (Interactive Swagger API documentation).

---

## 5. Machine Learning & Real Crunchbase Data

VentureIQ includes two datasets in `data/`:
1. `data/synthetic_startups.csv`: 8,000 synthetic records with controlled feature distributions and noise.
2. `data/crunchbase_real_startups.csv`: 923 verified real-world startups extracted from the historical Crunchbase dataset with observed acquired vs. closed outcomes.

To retrain the model on the real Crunchbase dataset from the command line:
```powershell
python train_model.py --data data/crunchbase_real_startups.csv --data-note "Crunchbase Historical Dataset: 923 startups with real-world acquired vs closed status outcomes"
```
Or use the **Model Retraining Studio** in the *Settings* tab inside the web UI with one click!

---

## 6. Automated Testing Verification

The project includes automated test coverage across both the backend logic and the REST API endpoints:

```powershell
python -m unittest discover -t . -s tests
```

**Verification Results:**
- **85 tests executed and passing** (`Ran 85 tests in 16.105s, OK`).
- Includes model training, feature validation, log-odds feature importance, RAG index creation, Groq HTTP retry logic, SQLite persistence, PDF/Markdown export, and full FastAPI REST endpoints.

---

## 7. Limitations & Transparency Disclosures

- **Decision-Support Only:** Viability scores and risk classifications are analytical indicators to guide due diligence, not guarantees of startup success or financial advice.
- **Offline Fallback:** When a Groq API key is not configured or network access is offline, VentureIQ transparently falls back to structured rule-based assessments, clearly marked with the banner *"Rule-Based Offline Analysis (No AI Model Used)"*.
- **No Fabricated Statistics:** TAM/SAM/SOM figures and competitor notes are presented as methodology hypotheses, with source citations explicitly tagged to knowledge base primers.
