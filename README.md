# LLM Agents for Technical Metadata Extraction

> Hybrid ML + LLM pipeline that automatically extracts and validates technical metadata from web sources and internal databases. Combines Random Forest & SVM classifiers with LLM agents for high-accuracy, structured output from unstructured engineering text.

---

## Project overview

A two-stage intelligence system:

1. **ML stage** (fast, local): Random Forest + SVM classifiers on TF-IDF features classify component type with confidence scores
2. **LLM agent stage** (deep understanding): LLM validates, corrects, and enriches metadata — filling gaps the ML model misses

**Output**: Structured JSON with component type, voltage, capacity, weight, certifications, manufacturer, part number, and a full audit trail of corrections.

---

## Architecture

```
llm-agents-metadata-extraction/
├── src/
│   ├── llm_agent.py       # LLM agent + web scraper + rule-based regex extractor
│   ├── ml_classifier.py   # RF + SVM ensemble classifier (TF-IDF features)
│   └── pipeline.py        # End-to-end orchestration
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/PRATdoppelEK/llm-agents-metadata-extraction.git
cd llm-agents-metadata-extraction
pip install -r requirements.txt
```

For local LLM (recommended for data privacy):
```bash
# Install Ollama: https://ollama.ai
ollama pull mistral
```

---

## Quickstart

### Demo mode — no LLM needed, runs immediately
```bash
cd src
python pipeline.py
```

### With your own text
```bash
cd src
python pipeline.py --texts "Samsung 50Ah NMC cell 3.65V UN38.3 certified -30°C to 60°C"
```

### With URLs (scrapes page automatically)
```bash
cd src
python pipeline.py --urls https://example.com/battery-spec --output results/metadata.json
```

---

## Sample output

```json
{
  "item_id": "demo_cell",
  "component_type": "battery_cell",
  "material": "NMC",
  "voltage_v": 3.65,
  "capacity_ah": 50.0,
  "temperature_range": "-30°C to 60°C",
  "certifications": ["UN 38.3"],
  "ml_confidence": 0.94,
  "llm_validated": true,
  "llm_corrections": {}
}
```

---

## Key technical highlights

- **Ensemble ML**: RF + SVM with TF-IDF (1–3 gram) for robust text classification across 9 component classes
- **LLM validation**: Structured JSON extraction with full audit trail of corrections
- **Privacy-first**: Works fully offline with local LLMs via Ollama (Mistral, LLaMA 3)
- **Rule-based pre-pass**: Regex extraction for voltage, capacity, certifications before LLM call
- **Web scraping**: BeautifulSoup-based scraper for product and specification pages

---

## Tech stack

`scikit-learn` · `BeautifulSoup4` · `Ollama` · `OpenAI-compatible API` · `Python 3.10+`

---

## Author

**Prateek Gaur** — ML Engineer | Battery & Engineering AI
[LinkedIn](https://www.linkedin.com/in/prateek-gaur-15a629b4) · [GitHub](https://github.com/PRATdoppelEK) · prateekgaur@gmx.de
