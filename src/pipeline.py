"""
Main orchestration pipeline: ML classifier → LLM agent → validated metadata.
Author: Prateek Gaur

Usage:
    python pipeline.py --urls https://example.com/battery-spec \
                       --llm_url http://localhost:11434/v1/chat/completions \
                       --output results.json
"""

import argparse
import json
import logging
import os
from typing import List

from llm_agent     import LLMMetadataAgent, TechnicalWebScraper, RuleBasedExtractor, TechnicalMetadata
from ml_classifier import ComponentClassifier, get_seed_classifier

logger = logging.getLogger(__name__)


class MetadataExtractionPipeline:
    """
    Full pipeline:
    1. Fetch text from URLs or use provided raw text
    2. Rule-based fast extraction (regex patterns)
    3. ML classifier → component type + confidence
    4. LLM agent → validate, enrich, and correct
    5. Export structured JSON
    """

    def __init__(
        self,
        llm_url:   str = "http://localhost:11434/v1/chat/completions",
        llm_model: str = "mistral",
        api_key:   str = "",
        clf_path:  str = "",
    ):
        self.scraper   = TechnicalWebScraper()
        self.extractor = RuleBasedExtractor()
        self.llm_agent = LLMMetadataAgent(api_url=llm_url, model=llm_model, api_key=api_key)

        if clf_path and os.path.exists(clf_path):
            self.classifier = ComponentClassifier.load(clf_path)
            logger.info(f"Loaded classifier from {clf_path}")
        else:
            logger.info("Training seed classifier...")
            self.classifier = get_seed_classifier()

    def process_url(self, url: str, item_id: str = "") -> TechnicalMetadata:
        text = self.scraper.fetch(url)
        return self.process_text(text, item_id=item_id or url, source_url=url)

    def process_text(self, text: str, item_id: str = "item_0", source_url: str = "") -> TechnicalMetadata:
        meta = TechnicalMetadata(item_id=item_id, raw_text=text, source_url=source_url)

        # Step 1: Rule-based extraction
        rule_results = self.extractor.extract(text)
        for k, v in rule_results.items():
            if hasattr(meta, k):
                setattr(meta, k, v)

        # Step 2: ML classification
        labels, confs = self.classifier.predict([text])
        meta.ml_label      = labels[0]
        meta.ml_confidence = confs[0]
        meta.component_type = labels[0]

        # Step 3: LLM validation
        meta = self.llm_agent.validate_and_correct(meta, text)

        logger.info(
            f"[{item_id}] type={meta.component_type} "
            f"ml_conf={meta.ml_confidence:.2%} "
            f"llm_validated={meta.llm_validated} "
            f"corrections={len(meta.llm_corrections)}"
        )
        return meta

    def process_batch(self, inputs: List[dict]) -> List[TechnicalMetadata]:
        """
        inputs: list of dicts with keys 'url' OR 'text', plus optional 'id'
        """
        results = []
        for i, inp in enumerate(inputs):
            item_id = inp.get("id", f"item_{i}")
            if "url" in inp:
                results.append(self.process_url(inp["url"], item_id))
            elif "text" in inp:
                results.append(self.process_text(inp["text"], item_id))
        return results

    def export_json(self, results: List[TechnicalMetadata], output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        data = [r.to_dict() for r in results]
        # Remove raw_text from output to keep file clean
        for d in data:
            d.pop("raw_text", None)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results exported to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="LLM + ML Technical Metadata Extraction Pipeline")
    p.add_argument("--urls",      nargs="*", default=[], help="URLs to scrape")
    p.add_argument("--texts",     nargs="*", default=[], help="Raw text strings")
    p.add_argument("--input_json",default="",            help="JSON file with list of {id, url/text}")
    p.add_argument("--llm_url",   default="http://localhost:11434/v1/chat/completions")
    p.add_argument("--llm_model", default="mistral")
    p.add_argument("--api_key",   default="")
    p.add_argument("--clf_path",  default="")
    p.add_argument("--output",    default="results/metadata.json")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()

    pipeline = MetadataExtractionPipeline(
        llm_url=args.llm_url, llm_model=args.llm_model,
        api_key=args.api_key, clf_path=args.clf_path,
    )

    inputs = []
    for i, url  in enumerate(args.urls):
        inputs.append({"id": f"url_{i}", "url": url})
    for i, text in enumerate(args.texts):
        inputs.append({"id": f"text_{i}", "text": text})
    if args.input_json and os.path.exists(args.input_json):
        with open(args.input_json) as f:
            inputs.extend(json.load(f))

    # Demo mode: use built-in sample if nothing provided
    if not inputs:
        logger.info("No input provided — running demo with sample texts")
        inputs = [
            {"id": "demo_cell",  "text": "Samsung SDI 50Ah NMC prismatic cell 3.65V nominal, operating range -30°C to 60°C, UN 38.3 certified, weight 900g, part number INR21700-50E"},
            {"id": "demo_pack",  "text": "400V 80kWh liquid-cooled battery pack with integrated 14S8P configuration, BMS, and CAN 2.0B interface for EV applications"},
            {"id": "demo_motor", "text": "BLDC hub motor 1500W 48V, Kt=0.3 Nm/A, IP65, weight 4.2kg, operating temperature -20°C to 70°C"},
        ]

    results = pipeline.process_batch(inputs)
    pipeline.export_json(results, args.output)

    print("\n── Summary ────────────────────────────────────────────")
    for r in results:
        print(f"  {r.item_id:<20} | {r.component_type:<20} | conf={r.ml_confidence:.0%} | llm={r.llm_validated}")


if __name__ == "__main__":
    main()
