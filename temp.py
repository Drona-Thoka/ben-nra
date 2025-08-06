
# --- Main Simulation Logic ---
def run_single_scenario(scenario, dataset, total_inferences, max_consultation_turns, scenario_idx):
    patient_agent = PatientAgent(scenario=scenario)

    doctor_agent = DoctorAgent(scenario=scenario, max_infs=total_inferences, prompt="")
    meas_agent = MeasurementAgent(scenario=scenario)

    available_tests = scenario.get_available_tests()
    run_log = {
        "timestamp": datetime.now(),
        "model": MODEL_NAME,
        "dataset": dataset,
        "scenario_id": scenario_idx,
        "max_patient_turns": total_inferences,
        "max_consultation_turns": max_consultation_turns,
        "patient_interaction_turns" : 0,
        "doctor_questions": 0,
        "correct_diagnosis": scenario.diagnosis_information(),
        "dialogue_history": [],
        "requested_tests": [],
        "tests_requested_count": 0,
        "available_tests": available_tests,
        "determined_specialist": None,
        "consultation_analysis": {},
        "final_doctor_diagnosis": None,
        "top_K diagnoses": None,
        "top_K": TOP_K,
        "is_correct": None,
        "embedding_similarity": None,
        "best_embedding_similarity": None,
    }

    # --- Patient Interaction Phase ---
    print(f"\n--- Phase 1: Patient Interaction (Max {total_inferences} turns) ---")
    doctor_dialogue, state = doctor_agent.inference_doctor("Patient presents with initial information.", mode="patient")
    print(f"Doctor [Turn 0]: {doctor_dialogue}")
    run_log["dialogue_history"].append({"speaker": "Doctor", "turn": 0, "phase": "patient", "text": doctor_dialogue})
    meas_agent.add_hist(f"Doctor: {doctor_dialogue}")

    next_input_for_doctor = scenario.examiner_information()

    for turn in range(1, total_inferences + 1):
        current_speaker = "Patient"
        if "REQUEST TEST" in doctor_dialogue:
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
            meas_agent.add_hist(history_update)
            current_speaker = "Measurement"
            history_update = f"Doctor: {doctor_dialogue}"            
            patient_agent.add_hist(history_update)
        else:
            patient_response = patient_agent.inference_patient(doctor_dialogue)
            print(f"Patient [Turn {turn}]: {patient_response}")
            next_input_for_doctor = patient_response
            run_log["dialogue_history"].append({"speaker": "Patient", "turn": turn, "phase": "patient", "text": patient_response})
            history_update = f"Patient: {patient_response}"
            meas_agent.add_hist(f"Doctor: {doctor_dialogue}\n\nPatient: {patient_response}")
            current_speaker = "Patient"

        doctor_dialogue, state = doctor_agent.inference_doctor(next_input_for_doctor, mode="patient")
        print(f"Doctor [Turn {turn}]: {doctor_dialogue}")
        run_log["dialogue_history"].append({"speaker": "Doctor", "turn": turn, "phase": "patient", "text": doctor_dialogue})
        meas_agent.add_hist(f"Doctor: {doctor_dialogue}")

        if ((state == "consultation_needed") or turn == total_inferences):
             print("\nPatient interaction phase complete.")
             break

        time.sleep(0.5)

    run_log["tests_requested_count"] = len(run_log["requested_tests"])
    run_log["tests_left_out"] = list(set(available_tests) - set(run_log["requested_tests"]))
    print(f"Total tests requested during patient interaction: {run_log['tests_requested_count']}")
    print(f"Tests left out: {run_log['tests_left_out']}")

    # --- Specialist Determination Phase, if and only if configured---
    print(f"\n--- Phase 2: Determining Specialist ---")
    specialist_type, specialist_reason = doctor_agent.determine_specialist()
    run_log["determined_specialist"] = specialist_type
    run_log["specialist_reason"] = specialist_reason
    specialist_agent = SpecialistAgent(scenario=scenario, specialty=specialist_type)
    specialist_agent.agent_hist = doctor_agent.agent_hist
    last_specialist_response = "I have reviewed the patient's case notes. Please share your thoughts to begin the consultation."
    run_log["dialogue_history"].append({"speaker": "System", "turn": total_inferences + 1, "phase": "consultation", "text": f"Consultation started with {specialist_type}. Reason: {specialist_reason}"})


    # --- Specialist Consultation Phase ---
    print(f"\n--- Phase 3: Specialist Consultation (Max {max_consultation_turns} turns) ---")
    consultation_dialogue_entries = []
    for consult_turn in range(1, max_consultation_turns + 1):
        full_turn = total_inferences + consult_turn

        doctor_consult_msg, state = doctor_agent.inference_doctor(last_specialist_response, mode="consultation")
        print(f"Doctor [Consult Turn {consult_turn}]: {doctor_consult_msg}")
        doctor_entry = {"speaker": "Doctor", "turn": full_turn, "phase": "consultation", "text": doctor_consult_msg}
        run_log["dialogue_history"].append(doctor_entry)
        consultation_dialogue_entries.append(doctor_entry)

        specialist_response = specialist_agent.inference_specialist(doctor_consult_msg)
        print(f"Specialist ({specialist_type}) [Consult Turn {consult_turn}]: {specialist_response}")
        specialist_entry = {"speaker": f"Specialist ({specialist_type})", "turn": full_turn, "phase": "consultation", "text": specialist_response}
        run_log["dialogue_history"].append(specialist_entry)
        consultation_dialogue_entries.append(specialist_entry)
        last_specialist_response = specialist_response

        run_log["CONFIDENCE_EPSILON"] = CONFIDENCE_EPSILON
        run_log["Switch_Confidence"] = confidence
        run_log["NUM_SWITCHES_up_to_this_point"] = NUM_SWITCHES
        if ("SWITCH" in run_log["dialogue_history"]) and (confidence >= CONFIDENCE_EPSILON) and (NUM_SWITCHES < SWITCH_CAP):
            NUM_SWITCHES += 1
            doctor_agent = DoctorAgent()
            break

        time.sleep(0.5)

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

    # Compute embedding similairites 
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

    # --- Consultation Analysis Phase (Moved here) ---
    print("\n--- Phase 5: Consultation Analysis ---")
    consultation_history_text = "\n".join([f"{entry['speaker']}: {entry['text']}" for entry in run_log["dialogue_history"] if entry["phase"] == "consultation"])
    if consultation_history_text:
        consultation_analysis_results = analyze_consultation(consultation_history_text)
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


    run_log["info_density_score"] = float(calculate_info_density_score(run_log["dialogue_history"]))
    return run_log, run_log.get("is_correct", False)

