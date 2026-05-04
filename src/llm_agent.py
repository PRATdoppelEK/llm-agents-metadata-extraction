"""
LLM Agent for Technical Metadata Extraction & Validation.
Author: Prateek Gaur

Combines classical ML classifiers (Random Forest, SVM) with LLM agents
to automatically extract and validate technical metadata from web sources
and internal databases.
"""

import os
import re
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class TechnicalMetadata:
    """Structured technical metadata for an engineering component/product."""
    item_id:          str
    raw_text:         str
    source_url:       str = ""
    # Extracted fields
    component_type:   str = ""
    material:         str = ""
    voltage_v:        Optional[float] = None
    capacity_ah:      Optional[float] = None
    weight_kg:        Optional[float] = None
    temperature_range: str = ""
    certifications:   List[str] = field(default_factory=list)
    manufacturer:     str = ""
    part_number:      str = ""
    # Validation
    ml_label:         str = ""
    ml_confidence:    float = 0.0
    llm_validated:    bool = False
    llm_corrections:  Dict[str, Any] = field(default_factory=dict)
    validation_notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── Web Scraper ───────────────────────────────────────────────────────────────

class TechnicalWebScraper:
    """Fetches and cleans technical text from URLs."""

    HEADERS = {"User-Agent": "Mozilla/5.0 (TechnicalMetadataBot/1.0)"}

    def fetch(self, url: str, timeout: int = 10) -> str:
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            return " ".join(soup.get_text(separator=" ").split())
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return ""

    def fetch_batch(self, urls: List[str]) -> Dict[str, str]:
        return {url: self.fetch(url) for url in urls}


# ── Rule-based Extractor ──────────────────────────────────────────────────────

class RuleBasedExtractor:
    """
    Fast regex-based extraction of common technical parameters.
    Acts as the first-pass extractor before LLM refinement.
    """

    PATTERNS = {
        "voltage_v":   r"(\d+(?:\.\d+)?)\s*V(?:olt)?",
        "capacity_ah": r"(\d+(?:\.\d+)?)\s*(?:Ah|mAh|kWh)",
        "weight_kg":   r"(\d+(?:\.\d+)?)\s*(?:kg|g)\b",
        "part_number": r"(?:Part\s*#|P/N|SKU)[:\s]*([A-Z0-9\-]{4,20})",
        "temperature": r"(-?\d+)\s*°?C\s*(?:to|~|-)\s*(-?\d+)\s*°?C",
        "certifications": r"\b(CE|UL|IEC\s*\d+|ISO\s*\d+|UN\s*38\.3|RoHS|REACH)\b",
    }

    def extract(self, text: str) -> dict:
        results = {}

        for field, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if not matches:
                continue
            if field == "voltage_v":
                results["voltage_v"] = float(matches[0])
            elif field == "capacity_ah":
                val, unit = float(matches[0]), "Ah"
                if "mAh" in text[:text.find(str(matches[0])) + 10]:
                    val /= 1000
                    unit = "Ah (converted)"
                results["capacity_ah"] = val
            elif field == "weight_kg":
                results["weight_kg"] = float(matches[0])
            elif field == "part_number":
                results["part_number"] = matches[0]
            elif field == "temperature":
                results["temperature_range"] = f"{matches[0][0]}°C to {matches[0][1]}°C"
            elif field == "certifications":
                results["certifications"] = list(set(matches))

        return results


# ── LLM Agent ────────────────────────────────────────────────────────────────

class LLMMetadataAgent:
    """
    LLM-powered agent that validates and enriches metadata extracted by
    rule-based + ML methods. Supports OpenAI-compatible APIs and local LLMs.
    """

    SYSTEM_PROMPT = """You are a technical metadata extraction expert specializing in 
battery systems, automotive components, and industrial engineering parts.

Given raw text from a technical document or product page, extract and validate the following:
- component_type: (e.g., "lithium-ion cell", "BMS module", "battery pack", "capacitor")
- material: (e.g., "NMC", "LFP", "steel", "aluminum")  
- voltage_v: nominal voltage in Volts (float)
- capacity_ah: capacity in Amp-hours (float)
- weight_kg: weight in kilograms (float)
- temperature_range: operating temperature range (string, e.g., "-20°C to 60°C")
- certifications: list of certifications (e.g., ["CE", "UN 38.3", "IEC 62133"])
- manufacturer: company name (string)
- part_number: part/model number (string)

Respond ONLY with a valid JSON object. If a field cannot be determined, use null.
Do not include any explanation or markdown — pure JSON only."""

    def __init__(
        self,
        api_url: str  = "http://localhost:11434/v1/chat/completions",  # Ollama default
        model: str    = "mistral",
        api_key: str  = "",
        max_tokens: int = 512,
    ):
        self.api_url   = api_url
        self.model     = model
        self.api_key   = api_key or os.getenv("OPENAI_API_KEY", "")
        self.max_tokens = max_tokens

    def _truncate(self, text: str, max_chars: int = 3000) -> str:
        return text[:max_chars] + "..." if len(text) > max_chars else text

    def extract(self, text: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": f"Extract technical metadata from:\n\n{self._truncate(text)}"},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip markdown code fences if present
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM JSON parse error: {e}")
            return {}
        except Exception as e:
            logger.warning(f"LLM API error: {e}")
            return {}

    def validate_and_correct(self, metadata: TechnicalMetadata, raw_text: str) -> TechnicalMetadata:
        """Use the LLM to validate existing metadata and fill gaps."""
        current = {
            k: v for k, v in metadata.to_dict().items()
            if k in ["component_type","material","voltage_v","capacity_ah",
                     "weight_kg","temperature_range","certifications","manufacturer","part_number"]
        }
        prompt = (
            f"Current extracted metadata:\n{json.dumps(current, indent=2)}\n\n"
            f"Source text:\n{self._truncate(raw_text)}\n\n"
            "Validate the metadata, correct any errors, and fill in missing fields. "
            "Return corrected JSON only."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp    = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            content = resp.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE)
            corrections = json.loads(content)
            # Apply non-null corrections
            for k, v in corrections.items():
                if v is not None and hasattr(metadata, k):
                    old_val = getattr(metadata, k)
                    setattr(metadata, k, v)
                    if old_val != v:
                        metadata.llm_corrections[k] = {"before": old_val, "after": v}
            metadata.llm_validated = True
        except Exception as e:
            logger.warning(f"LLM validation failed: {e}")
            metadata.validation_notes = str(e)

        return metadata
