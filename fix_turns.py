from datetime import datetime
import argparse, os, json, time
import pandas as pd
from util import compare_results, get_log_file, log_scenario_data, analyze_consultation, get_completed_scenarios
from scenario import ScenarioLoader
from agent import PatientAgent, DoctorAgent, MeasurementAgent, SpecialistAgent
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

""" 
RELEASE NOTES:
1. Credit to Hassan and Liu et al. along with AgentClinic authors for code template
2. Adapting original MAS for to simulate multi-agent architectures
"""

# --- Constants ---
BASE_LOG_DIR = "logs"
MODEL_NAME = "gpt-4.1" 

# --- Simulation Configuration Constants ---
AGENT_DATASET = "MedQA"  # Start with MedQA as requested
NUM_SCENARIOS = 1      # Minimum 50 scenarios per bia s-dataset combo
# TOTAL_INFERENCES = 5
CONSULTATION_TURNS = 5
K_Values = [1,3,5,7,10]
TOP_K = max(K_Values)

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

client=OpenAI()
def get_embedding(text: str, model: str = "text-embedding-3-large"):
    text = text.replace("\n", " ")
    resp = client.embeddings.create(input=[text], model=model, encoding_format="float")
    return np.array(resp.data[0].embedding)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# using text-embdding-3-large model, returning a vector, and then cossim function returns a score by doing a dot product of the vectors and then dividing by the magnitude of both their lengths.

# --- Main Simulation Logic ---
def run_single_scenario(scenario, dataset, total_inferences, max_consultation_turns, scenario_idx, bias=None):
    patient_agent = PatientAgent(scenario=scenario)
    doctor_agent = DoctorAgent(scenario=scenario, max_infs=total_inferences, bias=bias)
    meas_agent = MeasurementAgent(scenario=scenario)
    specialist_agent = None

    available_tests = scenario.get_available_tests()
    run_log = {
        "timestamp": datetime.now(),
        "model": MODEL_NAME,
        "dataset": dataset,
        "scenario_id": scenario_idx,
        "bias_applied": bias,
        "inf_len": total_inferences,
        "max_patient_turns": total_inferences,
        "max_consultation_turns": max_consultation_turns,
        "correct_diagnosis": scenario.diagnosis_information(),
        "diagnoses_considered_count": 0,
        "dialogue_history": [],
        "requested_tests": [],
        "tests_requested_count": 0,
        "available_tests": available_tests,
        "determined_specialist": None,
        "consultation_analysis": {},
        "final_doctor_diagnosis": None,
        "top_K_diagnoses": None,
        "top_K_diagnoses_count": TOP_K,
        "embedding_similarity": None,
        "best_embedding_similarity": None,
        "is_correct": None,
    }

    # --- Patient Interaction Phase --- # 
    # --------------------------------- # 
    print(f"\n--- Phase 1: Patient Interaction (Max {total_inferences} turns) ---")
    doctor_dialogue, state = doctor_agent.inference_doctor("Patient presents with initial information.", mode="patient")
    print(f"Doctor [Turn 0]: {doctor_dialogue}")
    run_log["dialogue_history"].append({"speaker": "Doctor", "turn": 0, "phase": "patient", "text": doctor_dialogue})
    meas_agent.add_hist(f"Doctor: {doctor_dialogue}")
    next_input_for_doctor = scenario.examiner_information()

    for turn in range(1, total_inferences + 1):
        if "REQUEST TEST" in doctor_dialogue: #Interact with measurement agent 
            try:
                test_name = doctor_dialogue.split("REQUEST TEST:", 1)[1].strip().rstrip('.?!')
                if test_name:
                    run_log["requested_tests"].append(test_name)
                    print(f"System: Logged test request - {test_name}")
            except IndexError:
                print("Warning: Could not parse test name from doctor request.")
                test_name = "Unknown Test"

            result = meas_agent.inference_measurement(doctor_dialogue)
            print(f"Measurement [Turn {turn}]: {result}")
            next_input_for_doctor = result
            run_log["dialogue_history"].append({"speaker": "Measurement", "turn": turn, "phase": "patient", "text": result})
            history_update = f"Doctor: {doctor_dialogue}\n\nMeasurement: {result}"
            patient_agent.add_hist(history_update)
            meas_agent.add_hist(history_update)
        else: #Interact with patient agent 
            patient_response = patient_agent.inference_patient(doctor_dialogue)
            print(f"Patient [Turn {turn}]: {patient_response}")
            next_input_for_doctor = patient_response
            run_log["dialogue_history"].append({"speaker": "Patient", "turn": turn, "phase": "patient", "text": patient_response})
            history_update = f"Patient: {patient_response}"
            meas_agent.add_hist(f"Doctor: {doctor_dialogue}\n\nPatient: {patient_response}") #TODO ~ Why don't we update the history for the pat agent here? 

        # Transition state ~
        doctor_dialogue, state = doctor_agent.inference_doctor(next_input_for_doctor, mode="patient")
        print(f"Doctor [Turn {turn}]: {doctor_dialogue}")
        run_log["dialogue_history"].append({"speaker": "Doctor", "turn": turn, "phase": "patient", "text": doctor_dialogue})
        meas_agent.add_hist(f"Doctor: {doctor_dialogue}")

        if state == "consultation_needed" or turn == total_inferences:
             print("\nPatient interaction phase complete.")
             break

        summary = {}

        summary["Number of Tests Requested"] = run_log.get("tests_requested_count", [])
        summary["Number of Diagnoses Considered"] = run_log.get("diagnoses_considered_count", [])
        # print(f"Summary: Number of Tests Requested: {summary['Number of Tests Requested']} Number of Diagnoses Considered: {summary['Number of Diagnoses Considered']}")
        

        time.sleep(0.5)


    run_log["tests_requested_count"] = len(run_log["requested_tests"])
    run_log["tests_left_out"] = list(set(available_tests) - set(run_log["requested_tests"]))
    print(f"Total tests requested during patient interaction: {run_log['tests_requested_count']}")
    print(f"Tests left out: {run_log['tests_left_out']}")

    # --- Specialist Determination Phase --- # 
    # -------------------------------------- # 
    print(f"\n--- Phase 2: Determining Specialist ---")
    specialist_type, specialist_reason = doctor_agent.determine_specialist()
    run_log["determined_specialist"] = specialist_type
    run_log["specialist_reason"] = specialist_reason
    specialist_agent = SpecialistAgent(scenario=scenario, specialty=specialist_type)
    specialist_agent.agent_hist = doctor_agent.agent_hist
    last_specialist_response = "I have reviewed the patient's case notes. Please share your thoughts to begin the consultation."
    run_log["dialogue_history"].append({"speaker": "System", "turn": total_inferences + 1, "phase": "consultation", "text": f"Consultation started with {specialist_type}. Reason: {specialist_reason}"})



    # --- Specialist Consultation Phase --- #
    # ------------------------------------- # 
    print(f"\n--- Phase 3: Specialist Consultation (Max {max_consultation_turns} turns) ---")
    for consult_turn in range(1, max_consultation_turns + 1):
        full_turn = total_inferences + consult_turn

        # Doctor response
        doctor_consult_msg, state = doctor_agent.inference_doctor(last_specialist_response, mode="consultation")
        print(f"Doctor [Consult Turn {consult_turn}]: {doctor_consult_msg}")
        doctor_entry = {"speaker": "Doctor", "turn": full_turn, "phase": "consultation", "text": doctor_consult_msg}
        run_log["dialogue_history"].append(doctor_entry)

        # Specialist response
        specialist_response = specialist_agent.inference_specialist(doctor_consult_msg)
        print(f"Specialist ({specialist_type}) [Consult Turn {consult_turn}]: {specialist_response}")
        specialist_entry = {"speaker": f"Specialist ({specialist_type})", "turn": full_turn, "phase": "consultation", "text": specialist_response}
        run_log["dialogue_history"].append(specialist_entry)
        last_specialist_response = specialist_response
        time.sleep(0.5)

        run_log["info_density_score"] = float(calculate_info_density_score(run_log["dialogue_history"]))


    # --- Final Diagnosis Phase --- # 
    # ----------------------------- #
        # --- Final Diagnosis Phase ---
        print("\n--- Phase 4: Final Diagnosis ---")
        final_diagnosis_full = doctor_agent.get_final_diagnosis()
        print(f"FINAL DIAGNOSES FULL RAW: {final_diagnosis_full} ")
        if "DIAGNOSIS READY:" in final_diagnosis_full:
            final_diagnosis_text = final_diagnosis_full.split("DIAGNOSIS READY:", 1)[-1].strip()
            diagnoses = [d.strip() for d in final_diagnosis_text.split("|") if d.strip()][:TOP_K]
            print(f"FULL DIAGNOSIS LIST: {diagnoses}")
            run_log["top_K diagnoses"] = diagnoses
        else:
            final_diagnosis_text = "No diagnosis provided in correct format."
        print(f"\nFinal Diagnoses by Doctor: {diagnoses}")
        print(f"Correct Diagnosis: {scenario.diagnosis_information()}")
        # Compute prediction embeddings
        try:
            pred_embed = [get_embedding(diagnosis.strip().lower()) for diagnosis in diagnoses[:TOP_K]]
        except Exception as e:
            print(f"Embedding error (predictions): {e}")
            pred_embed = None
        # Compute ground truth embedding
        try:
            true_embed = get_embedding(scenario.diagnosis_information().strip().lower())
        except Exception as e:
            print(f"Embedding error (correct diagnosis): {e}")
            true_embed = None
        for k in K_Values:
            sliced = diagnoses[:min(k, len(diagnoses))]
            is_correct, final_diagnosis = compare_results(sliced, scenario.diagnosis_information(), k)
            print(f"Scenario {scenario_idx} | Top-{k} Diagnosis was {'CORRECT' if is_correct else 'INCORRECT'}")
            run_log[f"Top_{k}"] = sliced
            run_log[f"Top_{k} is_correct"] = is_correct
            if is_correct and final_diagnosis in sliced:
                run_log[f"Top_{k} correct rank"] = sliced.index(final_diagnosis) + 1
            if k == TOP_K:
                run_log["is_correct"] = is_correct
                run_log["final_doctor_diagnosis"] = final_diagnosis
                run_log["final_diagnosis_is_correct"] = is_correct
        # Compute embedding similarities
        if pred_embed is not None and true_embed is not None:
            embed_sim = [cosine_similarity(pred, true_embed) for pred in pred_embed]
            run_log["embedding_similarity"] = embed_sim
            run_log["best_embedding_similarity"] = max(embed_sim) if embed_sim else None
            if embed_sim:
                run_log["best_embedding_similarity_rank"] = embed_sim.index(run_log["best_embedding_similarity"]) + 1
        else:
            run_log["embedding_similarity"] = None
            run_log["best_embedding_similarity"] = None
            run_log["best_embedding_similarity_rank"] = None

    # --- Consultation Analysis Phase (Moved here) --- #
    # ------------------------------------------------ # 
    print("\n--- Phase 5: Consultation Analysis ---")
    consultation_history_text = "\n".join([f"{entry['speaker']}: {entry['text']}" for entry in run_log["dialogue_history"] if entry["phase"] == "consultation"])
    if consultation_history_text:
        consultation_analysis_results = analyze_consultation(consultation_history_text)
        print(f"\nConsultation analysis results: {consultation_analysis_results}")
        print(f"Kaylin Diagnoses Considered: {consultation_analysis_results.get('diagnoses_considered_count')} ")
        run_log["diagnoses_considered_count"] = consultation_analysis_results.get('diagnoses_considered_count')
        run_log["consultation_analysis"] = consultation_analysis_results
        print("Consultation Analysis Results:")
        if consultation_analysis_results:
            for key, value in consultation_analysis_results.items():
                if key != "test_density":
                     print(f"- {key.replace('_', ' ').title()}: {value}")
        else:
            print("Analysis could not be performed.")
    else:
        print("No consultation dialogue to analyze.")
        run_log["consultation_analysis"] = {"error": "No consultation dialogue recorded"}
    return run_log, run_log.get("is_correct", False)

def run_bias_dataset_combination(dataset, bias, num_scenarios, total_inferences, consultation_turns):
    """Run a single bias-dataset combination test"""
    log_file = get_log_file(dataset, total_inferences)
    # completed_scenario_ids = get_completed_scenarios(log_file) # Renamed for clarity
    completed_scenario_ids = []

    print(f"\n=== Testing {bias} bias on {dataset} dataset ===")
    print(f"Log file: {log_file}")
    # print(f"Already completed scenario IDs: {len(completed_scenario_ids)}")

    # Calculate number of correct scenarios from previous runs
    num_correct_from_previous_runs = 0
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, 'r') as f:
            try:
                previous_log_data = json.load(f)
                if isinstance(previous_log_data, list):
                    num_correct_from_previous_runs = sum(
                        1 for entry in previous_log_data
                        if entry.get("scenario_id") in completed_scenario_ids and entry.get("is_correct")
                    )
            except json.JSONDecodeError:
                print(f"Warning: Could not parse log file {log_file} for calculating previous accuracy.")

    scenario_loader = ScenarioLoader(dataset=dataset)
    max_available = scenario_loader.num_scenarios
    scenarios_to_run = min(num_scenarios, max_available)

    total_correct_current_session = 0 # Renamed for clarity
    total_simulated_current_session = 0 # Renamed for clarity

    # Create a list of scenarios to run, skipping already completed ones
    scenarios_to_process = [i for i in range(scenarios_to_run) if i not in completed_scenario_ids]
    print(f"Scenarios to run in this session: {len(scenarios_to_process)} of {scenarios_to_run} total planned")

    for scenario_idx in scenarios_to_process:
        print(f"\n--- Running Scenario {scenario_idx + 1}/{scenarios_to_run} with {bias} bias ---")
        scenario = scenario_loader.get_scenario(id=scenario_idx)
        # breakpoint()
        if scenario is None:
            print(f"Error loading scenario {scenario_idx}, skipping.")
            continue

        total_simulated_current_session += 1
        run_log, is_correct = run_single_scenario(
            scenario, dataset, total_inferences, consultation_turns, scenario_idx, bias
        )
        # breakpoint()

        if is_correct:
            total_correct_current_session += 1

        log_scenario_data(run_log, log_file)
        print(f"Tests requested in Scenario {scenario_idx + 1}: {run_log.get('requested_tests', [])}")
        print(f"Dataset used: [{dataset}]")

        # Update progress
        if total_simulated_current_session > 0:
            accuracy_current_session = (total_correct_current_session / total_simulated_current_session) * 100
            print(f"\nCurrent Accuracy for this session ({bias} bias on {dataset}): {accuracy_current_session:.2f}% ({total_correct_current_session}/{total_simulated_current_session})")

            # Calculate overall progress including previously completed scenarios
            overall_completed_count = len(completed_scenario_ids) + total_simulated_current_session
            overall_correct_count = num_correct_from_previous_runs + total_correct_current_session

            overall_accuracy_so_far = (overall_correct_count / overall_completed_count) * 100 if overall_completed_count > 0 else 0
            # The original problematic line was:
            # overall_accuracy = ((len([s for s in completed_scenarios if s in run_log.get("is_correct", False)]) + total_correct) /
            #                    overall_completed) * 100 if overall_completed > 0 else 0
            # This is now correctly calculated as overall_accuracy_so_far.
            print(f"Overall Progress for {bias} on {dataset}: {overall_completed_count}/{scenarios_to_run} scenarios completed. Overall Accuracy: {overall_accuracy_so_far:.2f}% ({overall_correct_count}/{overall_completed_count})")

    # Calculate final statistics for this combination
    final_completed_count = len(completed_scenario_ids) + total_simulated_current_session
    if final_completed_count > 0:
        # Load all results to get accurate count
        all_results = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    all_results = json.load(f)
                    if not isinstance(all_results, list): # Ensure it's a list
                        all_results = []
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse final log file {log_file} for final stats. Results may be inaccurate.")
                    all_results = []

        correct_count_total = sum(1 for entry in all_results if entry.get("is_correct")) # Ensure entry.get("is_correct") is True

        # Ensure final_completed_count matches the number of entries if all were logged correctly
        # This uses the actual number of entries in the log file for accuracy if possible.
        actual_entries_in_log = len(all_results)

        final_accuracy = (correct_count_total / actual_entries_in_log) * 100 if actual_entries_in_log > 0 else 0

        print(f"\n=== Results for {bias} bias on {dataset} dataset ===")
        print(f"Total Scenarios Logged: {actual_entries_in_log} (planned: {scenarios_to_run}, completed this/prev sessions: {final_completed_count})")
        print(f"Final Accuracy: {final_accuracy:.2f}% ({correct_count_total}/{actual_entries_in_log})")

    return final_completed_count >= scenarios_to_run

def remove_existing_log_file(dataset, bias):
    # """
    # Removes the specified log file if it exists.
    #
    # Parameters:
    # file_path (str): The full path to the log file.
    #
    # Returns:
    # bool: True if the file was deleted, False if it did not exist.
    # """
    file_path = get_log_file(dataset, bias)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted log file: {file_path}")


def main():
    # Create argument parser for optional parameters
    parser = argparse.ArgumentParser(description='Run medical diagnosis simulation with bias testing')
    parser.add_argument('--dataset', choices=['MedQA', 'NEJM', 'all'], default='all',
                      help='Which dataset to use (default: all)')
    parser.add_argument('--bias', help='Specific bias to test (default: test all biases)')
    parser.add_argument('--scenarios', type=int, default=NUM_SCENARIOS,
                      help=f'Number of scenarios to run per combination (default: {NUM_SCENARIOS})')
    args = parser.parse_args()
    
    # Determine which datasets to test
    datasets_to_test = ['MedQA', 'NEJM'] if args.dataset == 'all' else [args.dataset]
    
    # Determine which biases to test
    biases_to_test = [args.bias] if args.bias else ['none'] + list(ALL_BIASES.keys())
    
    print(f"Starting comprehensive bias testing across {len(datasets_to_test)} datasets and {len(biases_to_test)} biases")
    # print(f"Base settings: {args.scenarios} scenarios per combination, {total_inferences} patient interactions, {CONSULTATION_TURNS} consultation turns")
    #
    # Create summary report structures
    summary = {
        "start_time": datetime.now().isoformat(),
        "completed_combinations": 0,
        "total_combinations": len(datasets_to_test) * len(biases_to_test),
        "results_by_combination": {}
    }

if __name__ == "__main__":
    main()

# this is fixed turns, changing length
# lengths = [5, 10, 15, 20, 25]
max_lengths = 5

output_report = []
summary_report = []

datasets_to_test= ['MedQA']

total_simulated_current_session = 0
total_correct_session = 0
bias = "none"
max_consultation_turns = 5

summary = {
    "start_time": datetime.now().isoformat(),
    "completed_combinations": 0,
    "total_combinations": (NUM_SCENARIOS) * (max_lengths/5),
    "results_by_combination": {}
}
# remove_existing_log_file(datasets_to_test, total_inferences)

for dataset in datasets_to_test:
    for total_inferences in range(5, max_lengths + 1, 5):
        print(f"Kaylin inf_len: {total_inferences}")

        log_file = get_log_file(dataset, total_inferences)
        scenario_loader = ScenarioLoader(dataset=dataset)

        print(
            f"Dataset: {dataset} "
            f"Total Inferences: {total_inferences} "
            f"Max Consultation Turns: {max_consultation_turns} "
            f"Number of Scenario: {NUM_SCENARIOS} "
            f"Bias: {bias}")

        try:
            completed = run_bias_dataset_combination(
                dataset, bias, NUM_SCENARIOS, total_inferences, CONSULTATION_TURNS
            )

            # Update summary
            combination_key = f"{dataset}_{total_inferences}"
            print("Log File Name:", log_file)
            # log_file = get_log_file(dataset, bias)

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    results = json.load(f)
                    correct_count = sum(1 for entry in results if entry.get("is_correct", False))
                    total_count = len(results);



                    diagnoses_considered_count = sum(entry.get("diagnoses_considered_count", 0) for entry in results)
                    tests_requested_count = sum(entry.get("tests_requested_count", 0) for entry in results)


                    summary["results_by_combination"][combination_key] = {
                        "completed": completed,
                        "scenarios_run": total_count,
                        "correct_diagnoses": correct_count,
                        "diagnoses_considered_count": diagnoses_considered_count,
                        "accuracy": (correct_count / total_count) * 100 if total_count > 0 else 0
                    }

                    print(
                        f"dataset: {dataset} "
                        f"bias: {bias} "
                        f"scenarios: {total_count} "
                        f"total_inferences: {total_inferences} "
                        f"max_consultation_turns: {max_consultation_turns} "
                        f"correct_diagnoses: {correct_count} "
                        f"accuracy: {(correct_count / total_count) * 100 if total_count > 0 else 0} "
                        f"diagnoses_considered_count: {diagnoses_considered_count} "
                        f"tests_requested_count: {tests_requested_count} ")

                    summary_report.append(summary)

                    output = {
                        "dataset": dataset,
                        "bias": bias,
                        "scenarios": total_count,
                        "total_inferences": total_inferences,
                        "max_consultation_turns": max_consultation_turns,
                        "diagnoses_considered_count": diagnoses_considered_count,
                        "correct_diagnoses": correct_count,
                        "accuracy": (correct_count / total_count) * 100 if total_count > 0 else 0,
                        "diagnoses_considered_count_average": diagnoses_considered_count/NUM_SCENARIOS,
                        "tests_requested_count": tests_requested_count,
                        "tests_requested_count_average": tests_requested_count/NUM_SCENARIOS
                    }
                    print(output)
                    output_report.append(output)

            if completed:
                summary["completed_combinations"] += 1

        except Exception as e:
            print(f"Error running {dataset} with {bias} bias: {e}")
            # Continue with next combination even if this one fails

        # Save summary report
        summary["end_time"] = datetime.now().isoformat()
        summary["total_duration_seconds"] = (datetime.fromisoformat(summary["end_time"]) -
                                             datetime.fromisoformat(summary["start_time"])).total_seconds()

        with open(os.path.join(BASE_LOG_DIR, "bias_testing_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n\n=== TESTING COMPLETE ===")
        print(f"Completed {summary['completed_combinations']}/{summary['total_combinations']} combinations")
        print(f"Total duration: {summary['total_duration_seconds'] / 3600:.2f} hours")
        print(f"Full results saved to {os.path.join(BASE_LOG_DIR, 'bias_testing_summary.json')}")

df = pd.DataFrame(output_report)
df.to_csv("length_constraints_outputs.csv", index=False)
print("\nSaved results to length_constraints-outputs.csv")

df = pd.DataFrame(summary_report)
df.to_csv("length_constraints_summary.csv", index=False)
print("\nSaved results to length_constraints_results.csv")