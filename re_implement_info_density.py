import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer

BASE_LOG_DIR = "logs/"

INFO_DENSITY_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_info_density_score(dialogue_history):
    doctor_utterances = [entry["text"] for entry in dialogue_history if entry["speaker"] == "Doctor" and entry["phase"] == "patient"]
    if len(doctor_utterances) > 1:
        try:
            embeddings = INFO_DENSITY_MODEL.encode(doctor_utterances)
            norms = np.linalg.norm(embeddings, axis=1)

            sims = np.triu(np.inner(embeddings, embeddings) /
                        (norms[:, None] * norms), 1)
            
            denom = len(doctor_utterances) * (len(doctor_utterances) - 1) / 2
            if denom == 0:
                return 1.0
            
            avg_sim = np.sum(sims) / denom

            score = 1 - avg_sim
            return score if np.isfinite(score) else 1.0
        except Exception as e:
            print(f"[INFO_DENSITY_ERROR]: {e}")
            return 1.0
    return 1.0

info_densities = []

for filename in os.listdir(BASE_LOG_DIR):
    if filename.endswith("_log.json"):
        filepath = os.path.join(BASE_LOG_DIR, filename)

        with open(filepath, "r") as f:
            data = json.load(f)

        # Add info_density to each case
        for case in data:
            score = calculate_info_density_score(case.get("dialogue_history"))
            case["info_density"] = float(score)

        # Write updated JSON once per file
        output_path = os.path.join(BASE_LOG_DIR, f"Modified_{filename}")
        with open(output_path, "w") as g:
            json.dump(data, g, indent=2)

        print(f"Processed {filename} → {output_path}")
