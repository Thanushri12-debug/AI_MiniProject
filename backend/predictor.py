"""
predictor.py
Bayesian Disease Prediction Engine.

Uses a trained BernoulliNB / MultinomialNB model together with the
raw Bayes formula to produce per-disease posterior probabilities.

Bayes' Theorem:
    P(Disease | Symptoms) = P(Symptoms | Disease) * P(Disease)
                            ──────────────────────────────────
                                       P(Symptoms)

The sklearn model already computes the log-posteriors; we exponentiate
and normalise to get human-readable percentages.
"""

import numpy as np
import pickle
import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model.pkl")
LABEL_PATH    = os.path.join(BASE_DIR, "label_encoder.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")

# ── Disease metadata ──────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Flu": {
        "icon": "🤧",
        "color": "#4FC3F7",
        "description": "Influenza — viral respiratory infection.",
        "advice": "Rest, stay hydrated, take antivirals if severe. Consult a doctor if symptoms worsen.",
        "risk": "Moderate",
    },
    "Malaria": {
        "icon": "🦟",
        "color": "#FF7043",
        "description": "Parasitic disease transmitted by Anopheles mosquitoes.",
        "advice": "Seek immediate medical attention. Antimalarial medication required. Blood test recommended.",
        "risk": "High",
    },
    "Dengue": {
        "icon": "🩸",
        "color": "#EF5350",
        "description": "Viral hemorrhagic fever transmitted by Aedes mosquitoes.",
        "advice": "Hospitalisation may be needed. Monitor platelet count. Avoid aspirin/NSAIDs.",
        "risk": "High",
    },
    "Cold": {
        "icon": "🌡️",
        "color": "#66BB6A",
        "description": "Common viral upper respiratory tract infection.",
        "advice": "Rest and fluids. OTC decongestants may help. Usually resolves in 7–10 days.",
        "risk": "Low",
    },
}

# ── Loader ────────────────────────────────────────────────────────────────────

def load_model():
    """Load trained model artefacts from disk."""
    with open(MODEL_PATH,    "rb") as f: model    = pickle.load(f)
    with open(LABEL_PATH,    "rb") as f: le       = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: features = pickle.load(f)
    return model, le, features

# ── Core Predictor ────────────────────────────────────────────────────────────

class BayesianDiseasePredictor:
    """
    Wraps a trained Naive Bayes sklearn model and exposes a clean
    predict() interface that returns full posterior probabilities.

    Mathematical detail
    -------------------
    Naive Bayes assumes conditional independence of symptoms given disease:

        P(s1, s2, ..., sn | D) = ∏ P(si | D)

    Combined with Bayes' theorem:

        P(D | symptoms) ∝ P(D) · ∏ P(si | D)

    Working in log-space avoids underflow:

        log P(D | s) = log P(D) + Σ log P(si | D)

    sklearn's predict_proba() handles all of this internally.
    We expose the probabilities as percentages along with the formula steps.
    """

    def __init__(self):
        self.model, self.le, self.features = load_model()
        self._n_features = len(self.features)
        self._classes    = list(self.le.classes_)

    def _build_input_vector(self, symptoms: list[str]) -> np.ndarray:
        """Convert a list of symptom names to a binary feature vector."""
        vec = np.zeros(self._n_features, dtype=float)
        for s in symptoms:
            if s in self.features:
                vec[self.features.index(s)] = 1.0
        return vec.reshape(1, -1)

    def predict(self, symptoms: list[str]) -> dict:
        """
        Parameters
        ----------
        symptoms : list of symptom name strings (must be keys in FEATURES list)

        Returns
        -------
        dict with keys:
            top_disease     – highest-probability disease name
            probabilities   – {disease: percentage} for all diseases
            formula_steps   – human-readable Bayes formula walkthrough
            advice          – medical advice string
            risk            – risk level string
            symptom_vector  – the binary input used
        """
        vec = self._build_input_vector(symptoms)

        # Posterior log-probabilities from sklearn (already normalised)
        log_proba = self.model.predict_log_proba(vec)[0]
        proba     = np.exp(log_proba)
        proba    /= proba.sum()   # re-normalise for numerical safety

        results = {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(self._classes, proba)
        }

        top_disease = max(results, key=results.get)

        # ── Build Bayesian formula walkthrough ────────────────────────────────
        prior = {cls: round(1 / len(self._classes) * 100, 1) for cls in self._classes}

        # Class log-likelihoods from the model (log feature probs per class)
        feature_log_prob = self.model.feature_log_prob_   # shape (n_classes, n_features)
        symptom_indices  = [self.features.index(s) for s in symptoms if s in self.features]

        likelihood = {}
        for ci, cls in enumerate(self._classes):
            lp = sum(feature_log_prob[ci][si] for si in symptom_indices)
            likelihood[cls] = round(float(np.exp(lp)) * 100, 4)

        formula_steps = {
            "theorem": "P(Disease | Symptoms) = P(Symptoms | Disease) × P(Disease) / P(Symptoms)",
            "prior_probabilities": prior,
            "likelihoods": likelihood,
            "posterior_probabilities": results,
            "top_prediction": top_disease,
            "confidence": results[top_disease],
        }

        info = DISEASE_INFO.get(top_disease, {})

        return {
            "top_disease":    top_disease,
            "probabilities":  results,
            "formula_steps":  formula_steps,
            "description":    info.get("description", ""),
            "advice":         info.get("advice", "Consult a doctor."),
            "risk":           info.get("risk", "Unknown"),
            "icon":           info.get("icon", "🏥"),
            "color":          info.get("color", "#FFFFFF"),
            "symptom_vector": vec[0].tolist(),
            "active_symptoms": symptoms,
        }


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = BayesianDiseasePredictor()

    tests = [
        (["fever", "cough", "headache", "fatigue", "body_aches", "chills"], "Expected: Flu"),
        (["fever", "headache", "fatigue", "chills", "sweating", "joint_pain"], "Expected: Malaria"),
        (["fever", "headache", "fatigue", "nausea", "joint_pain", "rash"],   "Expected: Dengue"),
        (["cough", "sore_throat", "runny_nose", "fatigue"],                  "Expected: Cold"),
    ]

    for symptoms, note in tests:
        r = predictor.predict(symptoms)
        print(f"\n{note}")
        print(f"  Symptoms  : {symptoms}")
        print(f"  Prediction: {r['icon']} {r['top_disease']} ({r['probabilities'][r['top_disease']]}%)")
        print(f"  All probs : {r['probabilities']}")
