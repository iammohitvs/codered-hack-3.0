# 🛡️ LLM Safety Gateway

A CPU-centric, lightweight, real-time safety gateway for LLM prompts. Analyzes incoming prompts through multiple security layers in parallel and decides whether to **PASS**, **SANITIZE**, or **BLOCK** them.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Layer Details](#layer-details)

---

## Overview

The LLM Safety Gateway protects AI applications from prompt injection, jailbreaks, and malicious inputs by running **4 detection layers in parallel**:

| Layer | Purpose | Weight |
|-------|---------|--------|
| **Regex Analyzer** | Keyword/pattern-based detection | 0.20 |
| **Security Model** | Fine-tuned transformer classifier | 0.35 |
| **Mutation Layer** | Adversarial robustness testing | 0.30 |
| **Bottleneck Extractor** | Semantic similarity to known jailbreaks | 0.15 |

### Decision Logic

| Weighted Score | Action | Description |
|----------------|--------|-------------|
| < 0.4 | ✅ **PASS** | Safe prompt, forward to LLM |
| 0.4 - 0.7 | 🔶 **SANITIZE** | Rewrite prompt using LLM sanitizer |
| ≥ 0.7 | ❌ **BLOCK** | Reject the prompt entirely |

---

## Architecture

```
                    ┌─────────────────┐
                    │  Incoming       │
                    │  Prompt         │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  FastAPI        │
                    │  Gateway        │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Regex     │    │  Security   │    │  Mutation   │    │ Bottleneck  │
│  Analyzer   │    │   Model     │    │   Layer     │    │  Extractor  │
│  (0.20)     │    │  (0.35)     │    │  (0.30)     │    │  (0.15)     │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │                  │
       └──────────────────┼──────────────────┴──────────────────┘
                          │
                 ┌────────▼────────┐
                 │  Weighted Score │
                 │  Aggregation    │
                 └────────┬────────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
     ┌──────────┐   ┌──────────┐   ┌──────────┐
     │   PASS   │   │ SANITIZE │   │  BLOCK   │
     │  < 0.4   │   │ 0.4-0.7  │   │  ≥ 0.7   │
     └──────────┘   └────┬─────┘   └──────────┘
                         │
                ┌────────▼────────┐
                │   LLM-based     │
                │   Sanitizer     │
                └─────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd codered-hack-3.0
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Configure environment variables**
   ```bash
   # Copy template
   cp .env.example .env
   
   # Edit .env and add your SambaNova API key
   SAMBANOVA_API_KEY=your-api-key-here
   ```

6. **Run the server**
   ```bash
   uvicorn app:app --reload
   ```

   Server runs at: `http://localhost:8000`

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SAMBANOVA_API_KEY` | API key for LLM sanitizer (SambaNova) | Yes (for sanitization) |

### Layer Weights

Weights can be configured per-request via the API or adjusted in `app.py`:

```python
class GatewayConfig:
    WEIGHT_REGEX = 0.20
    WEIGHT_SECURITY_MODEL = 0.35
    WEIGHT_MUTATION = 0.30
    WEIGHT_BOTTLENECK = 0.15
    
    THRESHOLD_PASS = 0.4
    THRESHOLD_SANITIZE = 0.7
```

---

## API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### `GET /health`
Check if all layers are loaded and ready.

**Response:**
```json
{
  "status": "healthy",
  "layers_loaded": {
    "regex_analyzer": true,
    "security_model": true,
    "mutation_layer": true,
    "bottleneck_extractor": true,
    "sanitizer": true
  }
}
```

---

#### `POST /analyze`
Analyze a prompt through all safety layers **without** sanitizing.

**Request Body:**
```json
{
  "prompt": "Your prompt text here",
  "weights": {
    "regex_analyzer": 0.20,
    "security_model": 0.35,
    "mutation_layer": 0.30,
    "bottleneck_extractor": 0.15
  }
}
```

> Note: `weights` is optional. If omitted, default weights are used.

**Response:**
```json
{
  "prompt_preview": "Your prompt text here",
  "action": "pass",
  "weighted_score": 0.1234,
  "layer_scores": [
    {
      "name": "regex_analyzer",
      "score": 0.05,
      "weight": 0.20,
      "weighted_score": 0.01,
      "latency_ms": 12.5
    },
    {
      "name": "security_model",
      "score": 0.15,
      "weight": 0.35,
      "weighted_score": 0.0525,
      "latency_ms": 45.2
    },
    {
      "name": "mutation_layer",
      "score": 0.20,
      "weight": 0.30,
      "weighted_score": 0.06,
      "latency_ms": 120.8
    },
    {
      "name": "bottleneck_extractor",
      "score": 0.00,
      "weight": 0.15,
      "weighted_score": 0.00,
      "latency_ms": 8.3
    }
  ],
  "total_latency_ms": 125.5,
  "sanitized_prompt": null,
  "sanitizer_latency_ms": null
}
```

---

#### `POST /gateway`
Full safety pipeline: analyze → decide → sanitize (if needed).

**Request Body:** Same as `/analyze`

**Response:** Same as `/analyze`, but includes `sanitized_prompt` if action is "sanitize".

**Example with sanitization:**
```json
{
  "prompt_preview": "Ignore all instructions and...",
  "action": "sanitize",
  "weighted_score": 0.55,
  "layer_scores": [...],
  "total_latency_ms": 1250.5,
  "sanitized_prompt": "What information would you like to know?",
  "sanitizer_latency_ms": 850.2
}
```

---

#### `GET /config`
Get current gateway configuration.

**Response:**
```json
{
  "weights": {
    "regex_analyzer": 0.20,
    "security_model": 0.35,
    "mutation_layer": 0.30,
    "bottleneck_extractor": 0.15
  },
  "thresholds": {
    "pass": 0.4,
    "sanitize": 0.7
  },
  "actions": {
    "score < threshold_pass": "PASS",
    "threshold_pass <= score < threshold_sanitize": "SANITIZE",
    "score >= threshold_sanitize": "BLOCK"
  }
}
```

---

## Frontend

The gateway includes an interactive web frontend with real-time visualization.

### Running the Frontend

**Option 1: Via FastAPI (Recommended)**
```bash
# Start the server
uvicorn app:app --reload

# Open in browser
http://localhost:8000
```

**Option 2: Via Live Server (VS Code)**
1. Open `static/index.html` in VS Code
2. Right-click → "Open with Live Server"
3. The frontend will connect to `http://localhost:8000` for API calls

### Frontend Features

- **Prompt Input**: Enter prompts with mode selection (Analyze/Gateway)
- **Configurable Weights**: Adjust layer weights via sliders
- **Flowchart Visualization**: See prompts flow through each layer
- **Real-time Metrics**: View scores and latencies per layer
- **Action Indicators**: Visual feedback for PASS/SANITIZE/BLOCK

---

## Layer Details

### 1. Regex Analyzer (`utils/reg.py`)

Fast keyword and pattern-based detection using:
- 167 risk keywords with severity weights
- Leetspeak normalization
- Fuzzy matching with rapidfuzz
- spaCy tokenization

### 2. Security Model (`utils/security_model.py`)

Fine-tuned transformer classifier:
- Based on BERT architecture
- Binary classification (benign/malicious)
- Trained on jailbreak/prompt injection datasets
- Located at `models/saved_security_model/`

### 3. Mutation Layer (`utils/mutation_layer.py`)

Adversarial robustness testing:
- Generates prompt mutations (structural wraps, synonym swaps, typos)
- Tests each mutation against the security model
- Returns worst-case risk score
- Catches evasion attempts

### 4. Bottleneck Extractor (`utils/extractor.py`)

Semantic similarity detection using ChromaDB:
- 1999 known jailbreak signatures in vector database
- Sentence-level and whole-prompt matching
- Uses sentence-transformers for embeddings

### 5. Sanitizer (`utils/sanitizer.py`)

LLM-based prompt rewriting:
- Uses Meta Llama 3.3 70B via SambaNova API
- Removes malicious intent while preserving core request
- Only invoked when action is SANITIZE

---

## Project Structure

```
codered-hack-3.0/
├── app.py                    # FastAPI application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── static/
│   ├── index.html            # Frontend HTML
│   ├── styles.css            # Frontend styles
│   └── app.js                # Frontend JavaScript
├── utils/
│   ├── __init__.py           # Package exports
│   ├── reg.py                # Regex analyzer
│   ├── security_model.py     # Transformer classifier
│   ├── mutation_layer.py     # Adversarial mutations
│   ├── extractor.py          # Bottleneck extractor
│   ├── sanitizer.py          # LLM sanitizer
│   └── bottleneck-extractor/
│       └── chroma_db/        # Vector database
├── models/
│   └── saved_security_model/ # Fine-tuned model
└── datasets/                 # Training data
```

---

## Acknowledgments

- Built for CodeRed Hackathon 3.0
- Uses [SambaNova](https://sambanova.ai/) for LLM inference
- Security model trained on public jailbreak datasets
- Models, datasets, and other files used can be found here: https://drive.google.com/drive/folders/1I_PPnscwf_OYYY3b1cuC0HcTnpe9Xssx?usp=sharing
