"""
ML Classifiers (Random Forest + SVM) for technical component classification.
Author: Prateek Gaur

These models provide the first-pass component-type label and confidence score
that the LLM agent then validates and enriches.
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Tuple, Optional

from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.pipeline        import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics         import classification_report

logger = logging.getLogger(__name__)

COMPONENT_CLASSES = [
    "battery_cell",
    "battery_pack",
    "bms_module",
    "capacitor",
    "motor_component",
    "structural_part",
    "sensor",
    "connector",
    "unknown",
]


# ── Training Data (seed examples) ─────────────────────────────────────────────

SEED_TEXTS = [
    ("Lithium-ion prismatic cell 3.7V 50Ah NMC cathode automotive grade", "battery_cell"),
    ("LFP cylindrical cell 3.2V 200Ah deep cycle energy storage", "battery_cell"),
    ("Battery management system 14S BMS with balancing cell protection", "bms_module"),
    ("Active cell balancing module with SOC estimation CAN interface", "bms_module"),
    ("Battery pack 48V 100Ah with integrated BMS thermal management", "battery_pack"),
    ("High voltage battery system 400V 60kWh EV automotive application", "battery_pack"),
    ("Aluminum electrolytic capacitor 450V 1000uF 105°C industrial", "capacitor"),
    ("Film capacitor 630V DC-link power electronics inverter", "capacitor"),
    ("Brushless DC motor 48V 3kW BLDC controller integrated encoder", "motor_component"),
    ("Stator winding assembly 3-phase induction motor lamination stack", "motor_component"),
    ("Structural bracket aluminum alloy 6061-T6 load bearing", "structural_part"),
    ("Chassis component high-strength steel UHSS 1500 MPa stamped", "structural_part"),
    ("Temperature sensor NTC thermistor 10kΩ battery thermal monitoring", "sensor"),
    ("Hall effect current sensor 500A CAN output BMS integration", "sensor"),
    ("High-current connector 200A IP67 EV charging port automotive", "connector"),
    ("Busbars copper tinned 150A battery module interconnect", "connector"),
    ("Component datasheet engineering specification", "unknown"),
]


# ── ML Pipeline ───────────────────────────────────────────────────────────────

def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_svm_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])


class ComponentClassifier:
    """
    Ensemble of Random Forest + SVM with soft voting for component type classification.
    """

    def __init__(self, use_rf: bool = True, use_svm: bool = True):
        self.rf_pipe  = build_rf_pipeline()  if use_rf  else None
        self.svm_pipe = build_svm_pipeline() if use_svm else None
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> "ComponentClassifier":
        y = self.label_encoder.fit_transform(labels)

        if self.rf_pipe:
            self.rf_pipe.fit(texts, y)
            logger.info("Random Forest fitted")

        if self.svm_pipe:
            self.svm_pipe.fit(texts, y)
            logger.info("SVM fitted")

        self.is_fitted = True
        return self

    def predict_proba_ensemble(self, texts: List[str]) -> np.ndarray:
        probas = []
        if self.rf_pipe:
            probas.append(self.rf_pipe.predict_proba(texts))
        if self.svm_pipe:
            probas.append(self.svm_pipe.predict_proba(texts))
        return np.mean(probas, axis=0) if probas else np.zeros((len(texts), len(COMPONENT_CLASSES)))

    def predict(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        proba = self.predict_proba_ensemble(texts)
        idx   = np.argmax(proba, axis=1)
        labels = self.label_encoder.inverse_transform(idx)
        confs  = proba[np.arange(len(idx)), idx].tolist()
        return list(labels), confs

    def evaluate(self, texts: List[str], labels: List[str]) -> str:
        y = self.label_encoder.transform(labels)
        preds, _ = self.predict(texts)
        y_pred = self.label_encoder.transform(preds)
        return classification_report(y, y_pred, target_names=self.label_encoder.classes_)

    def cross_validate(self, texts: List[str], labels: List[str], cv: int = 5) -> dict:
        y = self.label_encoder.fit_transform(labels)
        results = {}
        if self.rf_pipe:
            scores = cross_val_score(self.rf_pipe, texts, y, cv=cv, scoring="f1_macro")
            results["rf_f1_macro"] = f"{scores.mean():.3f} ± {scores.std():.3f}"
        if self.svm_pipe:
            scores = cross_val_score(self.svm_pipe, texts, y, cv=cv, scoring="f1_macro")
            results["svm_f1_macro"] = f"{scores.mean():.3f} ± {scores.std():.3f}"
        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ComponentClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)


def get_seed_classifier() -> ComponentClassifier:
    """Train a classifier on the built-in seed dataset (no external data needed)."""
    texts  = [t for t, _ in SEED_TEXTS]
    labels = [l for _, l in SEED_TEXTS]
    clf = ComponentClassifier()
    clf.fit(texts, labels)
    return clf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clf = get_seed_classifier()

    test_texts = [
        "NMC pouch cell 3.65V 30Ah slim design EV battery module",
        "Battery pack thermal management liquid cooling 800V 100kWh",
        "Axial flux motor 15kW peak power 96V e-bike drive unit",
    ]
    labels, confs = clf.predict(test_texts)
    for text, label, conf in zip(test_texts, labels, confs):
        print(f"  [{conf:.2%}] {label:20s}  | {text[:60]}")
