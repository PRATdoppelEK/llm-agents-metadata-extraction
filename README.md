# LLM Agents for Technical Metadata Extraction

> Hybrid ML + LLM pipeline that automatically extracts and validates technical metadata from web sources and internal databases. Combines Random Forest & SVM classifiers with LLM agents for high-accuracy, structured output from unstructured engineering text.

---

## 🔍 Project Overview

A two-stage intelligence system:

1. **ML Stage** (fast, local): Random Forest + SVM classifiers trained on TF-IDF features to classify component type with confidence scores
2. **LLM Agent Stage** (deep understanding): LLM validates, corrects, and enriches metadata — filling gaps the ML model misses

**Output**: Structured JSON with component type, voltage, capacity, weight, certifications, manufacturer, part number, and more — with a full audit trail of corrections.

---

## 🏗️ Architecture

```
llm-agents-metadata-extraction/
├── src/
│   ├── llm_agent.py       # LLM agent + web scraper + rule-based extractor
│   ├── ml_classifier.py   # RF + SVM ensemble classifier
│   └── pipeline.py        # End-to-end orchestration
├── data/
├── configs/
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/PRATdoppelEK/llm-agents-metadata-extraction.git
cd llm-agents-metadata-extraction
pip install -r requirements.txt
```

**For local LLM** (recommended for privacy):
```bash
# Install Ollama: https://ollama.ai
ollama pull mistral
```

**For OpenAI API**:
```bash
export OPENAI_API_KEY=sk-...
```

---

## 🚀 Quickstart

### Demo mode (no LLM required — runs ML only)
```bash
python src/pipeline.py
```

### With URLs
```bash
python src/pipeline.py \
  --urls https://www.batteryspace.com/prod-specs/... \
  --llm_url http://localhost:11434/v1/chat/completions \
  --output results/metadata.json
```

### With raw text input
```bash
python src/pipeline.py \
  --texts "Samsung 50Ah NMC cell 3.65V UN38.3 certified" \
  --api_key $OPENAI_API_KEY \
  --llm_url https://api.openai.com/v1/chat/completions \
  --llm_model gpt-4o-mini
```

### Batch JSON input
```json
[
  {"id": "comp_001", "url": "https://example.com/battery-spec"},
  {"id": "comp_002", "text": "LFP cell 3.2V 100Ah automotive grade..."}
]
```
```bash
python src/pipeline.py --input_json batch.json --output results/batch_metadata.json
```

---

## 📊 Sample Output

```json
{
  "item_id": "demo_cell",
  "component_type": "battery_cell",
  "material": "NMC",
  "voltage_v": 3.65,
  "capacity_ah": 50.0,
  "weight_kg": 0.9,
  "temperature_range": "-30°C to 60°C",
  "certifications": ["UN 38.3"],
  "manufacturer": "Samsung SDI",
  "part_number": "INR21700-50E",
  "ml_label": "battery_cell",
  "ml_confidence": 0.94,
  "llm_validated": true,
  "llm_corrections": {}
}
```
*ML classifier achieves ~94% confidence on battery component classification. 
LLM agent corrects and enriches ~30% of ML outputs with additional structured fields.*
---
## 📈 Results

### Pipeline demo — 3 components extracted

See full output: [`results/metadata.json`](results/metadata.json)

**Extraction summary:**

| Component | Type | Voltage | Capacity | LLM Corrections | LLM Validated |
|-----------|------|---------|----------|-----------------|---------------|
| demo_cell | lithium-ion cell (NMC) | 3.65V | 50Ah | 5 (type, material, weight, manufacturer, part no.) | ✅ |
| demo_pack | battery_pack | 400V | 80kWh | 3 (capacity, temp range, part no.) | ✅ |
| demo_motor | BLDC hub motor | 48V | — | 2 (type, IP65 certification) | ✅ |

**Key observations:**
- ML classifier flags low confidence (15–20%) → automatically triggers LLM validation
- LLM corrects critical errors e.g. `weight_kg: 900.0 → 0.9`, `capacity_ah: 80 → 80000`
- Full correction audit trail available in `results/metadata.json`

---

## 🧠 Key Technical Highlights

- **Ensemble ML**: RF + SVM with TF-IDF (1–3 gram) for robust text classification
- **LLM Validation**: Structured JSON extraction with audit trail of corrections
- **Privacy-first**: Works fully offline with local LLMs (Ollama/Mistral/Llama)
- **Rule-based pre-pass**: Regex extraction for voltage, capacity, certifications before LLM
- **Web scraping**: BeautifulSoup-based scraper for product pages

---

## 🔧 Tech Stack

`scikit-learn` · `LangChain-compatible API` · `BeautifulSoup4` · `Ollama` · `OpenAI API` · `Python 3.10+`

---

## 👤 Author

**Prateek Gaur** — ML Engineer | Battery & Engineering AI  
[LinkedIn](https://www.linkedin.com/in/prateek-gaur-15a629b4) · [GitHub](https://github.com/PRATdoppelEK)
