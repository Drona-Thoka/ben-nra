import json
import numpy as np

NUM_FILES = 4
CONFIG_PATHS = ["logs/MedQA_Base-Case_log.json", "logs/MedQA_Augmented-Doctor_log.json", "logs/MedQA_Doctoral-Team_log.json", "logs/MedQA_Minimalist_log.json"]

for i in range(NUM_FILES):
    with open(CONFIG_PATHS[i], "r") as f:
        data = json.load(f)

    N = len(data)

    # Initialize accumulators
    metrics = {
        "patient_turns": [],
        "doctor_questions": [],
        "test_count": [],
        "info_density": [],
        "best_similarity": [],
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
        "top10_rank": []
    }

    for case in data:
        metrics["patient_turns"].append(case["patient_interaction_turns"])
        metrics["doctor_questions"].append(case["doctor_questions"])
        metrics["test_count"].append(case["tests_requested_count"])
        metrics["info_density"].append(case["info_density_score"])
        metrics["best_similarity"].append(case["best_embedding_similarity"])
        metrics["embedding_rank"].append(case["best_embedding_similarity_rank"])

        # Accuracies
        if case.get("final_diagnosis_is_correct"):
            metrics["final_correct"] += 1
        if case.get("Top_1 is_correct"):
            metrics["top1_correct"] += 1
            metrics["top1_rank"].append(case["Top_1 correct rank"])
        if case.get("Top_3 is_correct"):
            metrics["top3_correct"] += 1
            metrics["top3_rank"].append(case["Top_3 correct rank"])
        if case.get("Top_5 is_correct"):
            metrics["top5_correct"] += 1
            metrics["top5_rank"].append(case["Top_5 correct rank"])
        if case.get("Top_7 is_correct"):
            metrics["top7_correct"] += 1
            metrics["top7_rank"].append(case["Top_7 correct rank"])
        if case.get("Top_10 is_correct"):
            metrics["top10_correct"] += 1
            metrics["top10_rank"].append(case["Top_10 correct rank"])

    # Output
    def avg(arr): return round(np.mean(arr), 3)
    def acc(n): return round(n / N * 100, 2)

    print(f"Total Cases: {N}\n")
    print(f"=== Interaction Stats for === {CONFIG_PATHS[i].replace("logs/MedQA_", "").replace("_log.json", "")}")
    print(f"Avg. Patient Turns: {avg(metrics['patient_turns'])}")
    print(f"Avg. Doctor Questions: {avg(metrics['doctor_questions'])}")
    print(f"Avg. Tests Requested: {avg(metrics['test_count'])}")
    print(f"Avg. Info Density Score: {avg(metrics['info_density'])}")
    print(f"Avg. Best Embedding Similarity: {avg(metrics['best_similarity'])}")
    print(f"Avg. Embedding Similarity Rank: {avg(metrics['embedding_rank'])}")

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
