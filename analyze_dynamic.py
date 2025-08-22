import json
import numpy as np

NUM_FILES = 4
CONFIG_PATHS = "logs/MedQA_Dynamic_log.json"


with open(CONFIG_PATHS, "r") as f:
    data = json.load(f)

N = len(data)

# Initialize accumulators
metrics = {
    "patient_turns": [],
    "doctor_questions": [],
    "test_count": [],
    "info_density_score": [],
    "best_similarity": [],
    "embedding_similarity": [],
    "embedding_rank": [],
    "final_correct": 0,
    "top1_correct": 0,
    "top3_correct": 0,
    "top5_correct": 0,
    "top7_correct": 0,
    "top10_correct": 0,
    "top1_rank": [],
    "top3_rank": [],
    "top5_rank": [],
    "top7_rank": [],
    "top10_rank": [],
    "avg_doctor_considered": []
}

for case in data:
    metrics["patient_turns"].append(case["patient_interaction_turns"])
    metrics["doctor_questions"].append(case["doctor_questions"])
    metrics["test_count"].append(case["tests_requested_count"])
    metrics["info_density_score"].append(case["info_density_score"])
    metrics["best_similarity"].append(case["best_embedding_similarity"])
    metrics["embedding_rank"].append(case["best_embedding_similarity_rank"])
    metrics["embedding_similarity"].append(case["embedding_similarity"])
    if "consultation_analysis" in case and "diagnoses_considered" in case["consultation_analysis"]:
        metrics["avg_doctor_considered"].append(
        len(case["consultation_analysis"]["diagnoses_considered"])
    )
    else:
        # handle missing consultation gracefully
        metrics["avg_doctor_considered"].append(0) 



    for k in [1, 3, 5, 7, 10]:
        if case.get(f"Top_{k} is_correct"):
            metrics[f"top{k}_correct"] += 1
            rank_key = f"Top_{k} correct rank"
            if rank_key in case:
                metrics[f"top{k}_rank"].append(case[rank_key])

# Output
def avg(arr): return round(np.mean(arr), 3)
def acc(n): return round(n / N * 100, 2)

print(f"Total Cases: {N}\n")
print(f"=== Interaction Stats for === {CONFIG_PATHS.replace("logs/MedQA_", '').replace("_log.json", '')}")
print(f"Avg. Patient Turns: {avg(metrics['patient_turns'])}")
print(f"Avg. Doctor Questions: {avg(metrics['doctor_questions'])}")
print(f"Avg. Tests Requested: {avg(metrics['test_count'])}")
print(f"Avg. Info Density Score: {avg(metrics['info_density_score'])}")
print(f"Avg. Best Embedding Similarity: {avg(metrics['best_similarity'])}")
print(f"Avg. Embedding Similarity Rank: {avg(metrics['embedding_rank'])}")
print((f"Avg. Embedding Similarity: {avg(avg(metrics['embedding_similarity']))}"))
print(f"Avg. Diagnoses considered Count: {avg(metrics['avg_doctor_considered'])}")

print("\n=== Accuracy ===")
print(f"Final Diagnosis Accuracy: {acc(metrics['final_correct'])}%")
print(f"Top-1 Accuracy: {acc(metrics['top1_correct'])}%")
print(f"Top-3 Accuracy: {acc(metrics['top3_correct'])}%")
print(f"Top-5 Accuracy: {acc(metrics['top5_correct'])}%")
print(f"Top-7 Accuracy: {acc(metrics['top7_correct'])}%")
print(f"Top-10 Accuracy: {acc(metrics['top10_correct'])}%")

print("\n=== Average Ranks (Correct Only) ===")
print(f"Top-1 Correct Avg. Rank: {avg(metrics['top1_rank'])}")
print(f"Top-3 Correct Avg. Rank: {avg(metrics['top3_rank'])}")
print(f"Top-5 Correct Avg. Rank: {avg(metrics['top5_rank'])}")
print(f"Top-7 Correct Avg. Rank: {avg(metrics['top7_rank'])}")
print(f"Top-10 Correct Avg. Rank: {avg(metrics['top10_rank'])}")

