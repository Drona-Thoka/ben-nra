import openai, re, random, time, json, os
from datetime import datetime
import argparse
import glob
from dotenv import load_dotenv
import os

load_dotenv()


# --- Constants ---
BASE_LOG_DIR = "logs"
MODEL_NAME = "gpt-4.1" 

# --- Simulation Configuration Constants ---
AGENT_DATASET = "MedQA"  # Start with MedQA as requested
NUM_SCENARIOS = 10       # Minimum 50 scenarios per config-dataset combo
TOTAL_INFERENCES = 10
CONSULTATION_TURNS = 5

# --- Agent Choice Abalation Constants, 6 valid combos including base case, base case initally depicted ---
USE_DOCTOR = True
USE_MEASUREMENT = True
USE_SPECIALIST = True

#-- Prompts for each A-D situtation --
MINIMALIST_PROMPT = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {self.MAX_INFS} questions total before you must make a decision. You have asked {self.infs} questions so far. Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\""
AUGMENTED_DOCTOR_PROMPT = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {self.MAX_INFS} questions total before you must make a decision. You have asked {self.infs} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\""
DOCTOR_TEAM_PROMPT = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {self.MAX_INFS} questions total before you must make a decision. You have asked {self.infs} questions so far. You will be given a chance to consult with a specialist doctor during the session. Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\""
BASELINE_PROMPT = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {self.MAX_INFS} questions total before you must make a decision. You have asked {self.infs} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". You will be given a chance to consult with a specialist doctor during the session. Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\""

DOCTOR_PROMPTS = {"MINIMALIST_PROMPT": MINIMALIST_PROMPT, "AUGMENTED_DOCTOR_PROMPT": AUGMENTED_DOCTOR_PROMPT, "DOCTOR_TEAM_PROMPT": DOCTOR_TEAM_PROMPT, "BASELINE_PROMPT" : BASELINE_PROMPT}

# --- Utility Functions ---
def query_model(prompt, system_prompt, max_tokens=200):
    api_key = os.environ.get("OPENAI_API_KEY")    
    client = openai.OpenAI(api_key=api_key)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.05,
        max_tokens=max_tokens,
    )
    answer = response.choices[0].message.content.strip()
    return re.sub(r"\s+", " ", answer)

def compare_results(diagnosis, correct_diagnosis):
    prompt = f"Here is the correct diagnosis: {correct_diagnosis}\nHere was the doctor dialogue/diagnosis: {diagnosis}\nAre these referring to the same underlying medical condition? Please respond only with Yes or No."
    system_prompt = "You are an expert medical evaluator. Determine if the provided doctor's diagnosis matches the correct diagnosis in meaning, even if phrased differently. Respond only with 'Yes' or 'No'."
    answer = query_model(prompt, system_prompt)
    return answer.strip().lower() == "yes"

def get_log_file(dataset, config_name):
    """Create a log file name based on dataset and config"""
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    return os.path.join(BASE_LOG_DIR, f"{dataset}_{config_name}_log.json")

def log_scenario_data(data, log_file):
    """Log data to a specific log file"""
    # Ensure datetime is serializable
    if isinstance(data.get("timestamp"), datetime):
        data["timestamp"] = data["timestamp"].isoformat()
    
    existing_data = []
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, 'r') as f:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []
    
    existing_data.append(data)
    with open(log_file, 'w') as f:
        json.dump(existing_data, f, indent=2)

def analyze_consultation(consultation_history):
    """
    Analyzes the doctor-specialist consultation dialogue using an LLM.

    Args:
        consultation_history (str): The string containing the dialogue between
                                     the doctor and the specialist.

    Returns:
        dict: A dictionary containing the analysis metrics.
              Returns an empty dict if analysis fails.
    """
    prompt = f"""
Analyze the following medical consultation dialogue between a primary doctor and a specialist. Provide the analysis in JSON format with the following keys:
- "premature_conclusion": (Boolean) Did the primary doctor jump to a conclusion without sufficient discussion or evidence gathering during the consultation?
- "diagnoses_considered": (List) List all distinct potential diagnoses explicitly mentioned or discussed during the consultation.
- "diagnoses_considered_count": (Integer) Count the number of distinct potential diagnoses explicitly mentioned or discussed during the consultation.
- "disagreements": (Integer) Count the number of explicit disagreements or significant divergences in opinion between the doctor and the specialist.

Consultation Dialogue:
---
{consultation_history}
---

Respond ONLY with the JSON object.
"""
    system_prompt = "You are a medical education evaluator analyzing a consultation dialogue. Extract specific metrics and provide them in JSON format."

    analysis_json_str = query_model(prompt, system_prompt, max_tokens=300)

    try:
        # Clean potential markdown code block fences
        if analysis_json_str.startswith("```json"):
            analysis_json_str = analysis_json_str[7:]
        if analysis_json_str.endswith("```"):
            analysis_json_str = analysis_json_str[:-3]
        
        analysis_results = json.loads(analysis_json_str.strip())
        required_keys = ["premature_conclusion", "diagnoses_considered", "diagnoses_considered_count", "disagreements"]
        if all(key in analysis_results for key in required_keys):
            return analysis_results
        else:
            print(f"Warning: LLM analysis response missing required keys. Response: {analysis_json_str}")
            return {}
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse LLM analysis response as JSON. Response: {analysis_json_str}")
        return {}
    except Exception as e:
        print(f"Warning: An error occurred during consultation analysis: {e}")
        return {}

def get_completed_scenarios(log_file):
    """Get list of scenario IDs that have already been completed"""
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        return []
    
    with open(log_file, 'r') as f:
        try:
            data = json.load(f)
            return [entry.get("scenario_id") for entry in data if entry.get("scenario_id") is not None]
        except json.JSONDecodeError:
            print(f"Warning: Could not parse log file {log_file}. Starting from scratch.")
            return []

# --- Base Scenario Class ---
class BaseScenario:
    def __init__(self, scenario_dict):
        self.scenario_dict = scenario_dict
        self._init_data()
    
    def _init_data(self):
        # To be implemented by subclasses
        pass
    
    def patient_information(self):
        return self.patient_info
    
    def examiner_information(self):
        return self.examiner_info
    
    def exam_information(self):
        return self.exam_info
    
    def diagnosis_information(self):
        return str(self.diagnosis)

# --- Concrete Scenario Classes ---
class ScenarioMedQA(BaseScenario):
    def _init_data(self):
        self.tests = self.scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = self.scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info = self.scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info = self.scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = self.scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
        self.exam_info = {**self.physical_exams, "tests": self.tests}
    
    def get_available_tests(self):
        return list(self.tests.keys())

class ScenarioNEJM(BaseScenario):
    def _init_data(self):
        self.question = self.scenario_dict["question"]
        self.diagnosis = [_sd["text"] for _sd in self.scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = self.scenario_dict["patient_info"]
        self.physical_exams = self.scenario_dict["physical_exams"]
        self.examiner_info = "What is the most likely diagnosis?"
        self.exam_info = self.physical_exams
    
    def get_available_tests(self):
        # Use LLM to extract test names
        prompt = f"Extract the list of medical tests mentioned in the following text:\n\n{self.physical_exams}\n\nRespond with a comma-separated list of test names."
        system_prompt = "You are a medical assistant. Extract test names from the provided text."
        response = query_model(prompt, system_prompt)
        return [test.strip() for test in response.split(",") if test.strip()]

# --- Scenario Loader ---
class ScenarioLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.scenario_map = {
            "MedQA": (ScenarioMedQA, "agentclinic_medqa.jsonl"),
            "MedQA_Ext": (ScenarioMedQA, "agentclinic_medqa_extended.jsonl"),
            "NEJM": (ScenarioNEJM, "agentclinic_nejm.jsonl"),
            "NEJM_Ext": (ScenarioNEJM, "agentclinic_nejm_extended.jsonl"),
        }
        
        if dataset not in self.scenario_map:
            raise ValueError(f"Dataset '{dataset}' not recognized. Choices are: {list(self.scenario_map.keys())}")
        
        self._load_scenarios()
    
    def _load_scenarios(self):
        scenario_class, filename = self.scenario_map[self.dataset]
        
        with open(filename, "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [scenario_class(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def get_scenario(self, id=None):
        if not self.scenarios:
            return None
        if id is None:
            return random.choice(self.scenarios)
        if 0 <= id < self.num_scenarios:
            return self.scenarios[id]
        else:
            print(f"Warning: Scenario ID {id} out of range (0-{self.num_scenarios-1}). Returning None.")
            return None

# --- Agent Classes ---
class Agent:
    def __init__(self, scenario=None):
        self.scenario = scenario
        self.agent_hist = ""
        self.reset()
    
    def reset(self):
        self.agent_hist = ""
        self._init_data()
    
    def _init_data(self):
        # To be implemented by subclasses
        pass
    
    def add_hist(self, hist_str):
        self.agent_hist += hist_str + "\n\n"
    
    def system_prompt(self):
        # To be implemented by subclasses
        return ""

class PatientAgent(Agent):
    def _init_data(self):
        self.symptoms = self.scenario.patient_information()
    
    def system_prompt(self):
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = f"\n\nBelow is all of your information. {self.symptoms}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked."
        return base + symptoms
    
    def inference_patient(self, question):
        prompt = f"\nHere is a history of your dialogue:\n{self.agent_hist}\nHere was the doctor response:\n{question}\nNow please continue your dialogue\nPatient: "
        answer = query_model(prompt, self.system_prompt())
        return answer

class DoctorAgent(Agent):
    def __init__(self, scenario=None, max_infs=20, prompt=DOCTOR_PROMPTS["BASELINE_PROMPT"]):
        self.MAX_INFS = max_infs
        self.infs = 0
        self.specialist_type = None
        self.consultation_turns = 0
        self.system_prompt_template = prompt
        super().__init__(scenario)
    
    def _init_data(self):
        self.presentation = self.scenario.examiner_information()

    def get_system_prompt(self):
        return self.system_prompt_template.format(infs=self.infs, MAX_INFS=self.MAX_INFS, self=self)
    
    def system_prompt(self):
        presentation = f"\n\nBelow is all of the information you have. {self.presentation}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams."
        return self.system_prompt_template + presentation
    
    def determine_specialist(self):
        """Queries the LLM to determine the best specialist based on dialogue history."""
        prompt = f"Based on the following patient interaction history, what type of medical specialist (e.g., Cardiologist, Neurologist, Pulmonologist, Gastroenterologist, Endocrinologist, Infectious Disease Specialist, Oncologist, etc.) would be most appropriate to consult for a potential diagnosis? Please respond with only the specialist type.\n\nHistory:\n{self.agent_hist}"
        specialist = query_model(prompt, self.get_system_prompt())
        self.specialist_type = specialist.replace("Specialist", "").strip()
        explanation_prompt = f"Explain why a {self.specialist_type} is the most appropriate specialist based on the following dialogue history:\n\n{self.agent_hist}"
        explanation = query_model(explanation_prompt, self.get_system_prompt())
        print(f"Doctor decided to consult: {self.specialist_type}")
        print(f"Reason for choice: {explanation}")
        return self.specialist_type, explanation

    def inference_doctor(self, last_response, mode="patient"):
        """Generates the doctor's response, adapting to patient interaction or specialist consultation."""
        if mode == "patient":
             if self.infs > 0 or "Patient presents with initial information." not in last_response:
                 self.add_hist(f"Patient: {last_response}")
        elif mode == "consultation":
             self.add_hist(f"Specialist ({self.specialist_type}): {last_response}")

        if mode == "patient":
            if self.infs >= self.MAX_INFS:
                 return "Okay, I have gathered enough information from the patient. I need to analyze this and potentially consult a specialist.", "consultation_needed"

            prompt = f"\nHere is a history of your dialogue with the patient:\n{self.agent_hist}\nHere was the patient response:\n{last_response}\nNow please continue your dialogue with the patient. You have {self.MAX_INFS - self.infs} questions remaining for the patient. Remember you can REQUEST TEST: [test].\nDoctor: "
            system_prompt = f"You are a doctor named Dr. Agent interacting with a patient. You have {self.MAX_INFS - self.infs} questions left. Your goal is to gather information. {self.presentation}"
            answer = query_model(prompt, self.get_system_prompt())
            self.add_hist(f"Doctor: {answer}")
            self.infs += 1
            if "DIAGNOSIS READY:" in answer:
                answer = "Let me gather a bit more information first."
            return answer, "patient_interaction"

        elif mode == "consultation":
            prompt = f"\nHere is the full history (Patient interaction followed by consultation):\n{self.agent_hist}\nYou are consulting with a {self.specialist_type}.\nHere was the specialist's latest response:\n{last_response}\nContinue the consultation. Ask questions or share your thoughts to refine the diagnosis.\nDoctor: "
            system_prompt = f"You are Dr. Agent, consulting with a {self.specialist_type} about a patient case. Discuss the findings and differential diagnoses based on the history provided. Aim to reach a conclusion."
            answer = query_model(prompt, system_prompt)
            self.add_hist(f"Doctor: {answer}")
            self.consultation_turns += 1
            if "DIAGNOSIS READY:" in answer:
                 pass
            return answer, "consultation"

    def get_final_diagnosis(self):
        """Generates the final diagnosis prompt after all interactions."""
        prompt = f"\nHere is the complete history of your dialogue with the patient and the specialist ({self.specialist_type}):\n{self.agent_hist}\nBased on this entire consultation, please provide your final diagnosis now in the format 'DIAGNOSIS READY: [Your Diagnosis Here]'."
        system_prompt = f"You are Dr. Agent. You have finished interviewing the patient and consulting with a {self.specialist_type}. Review the entire history and provide your single, most likely final diagnosis in the required format."
        response = query_model(prompt, system_prompt)

        if "DIAGNOSIS READY:" not in response:
            return f"DIAGNOSIS READY: {response}"
        diagnosis_text = response.split("DIAGNOSIS READY:", 1)[-1].strip()
        return f"DIAGNOSIS READY: {diagnosis_text}"

class MeasurementAgent(Agent):
    def _init_data(self):
        self.information = self.scenario.exam_information()
    
    def system_prompt(self):
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = f"\n\nBelow is all of the information you have. {self.information}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS."
        return base + presentation
    
    def inference_measurement(self, doctor_request):
        prompt = f"\nHere is a history of the dialogue:\n{self.agent_hist}\nHere was the doctor measurement request:\n{doctor_request}"
        answer = query_model(prompt, self.system_prompt())
        return answer

# --- Specialist Agent Class ---
class SpecialistAgent(Agent):
    def __init__(self, scenario=None, specialty="General Medicine"):
        self.specialty = specialty
        super().__init__(scenario)
        self.information = scenario.exam_information()

    def _init_data(self):
        pass

    def system_prompt(self):
        base = f"You are a consulting specialist in {self.specialty}. You are discussing a case with the primary doctor (Dr. Agent). Review the provided dialogue history and the doctor's latest message. Provide your expert opinion, ask clarifying questions, or suggest next steps/differential diagnoses. Respond concisely (1-3 sentences) as dialogue."
        return base

    def inference_specialist(self, doctor_consult_message):
        self.add_hist(f"Doctor: {doctor_consult_message}")

        prompt = f"\nHere is the history of the case discussion:\n{self.agent_hist}\nHere was the primary doctor's latest message:\n{doctor_consult_message}\nPlease provide your specialist input.\nSpecialist ({self.specialty}): "
        answer = query_model(prompt, self.system_prompt())

        self.add_hist(f"Specialist ({self.specialty}): {answer}")
        return answer

# --- Somthing to aviod repetitve manual checks, make measurer useless
class NullMeasurmentAgent:
    def inference_measurement(self, *args, **kwargs):
        return ""
    
    def add_hist(self, *args, **kwargs):
        pass

# --- Main Simulation Logic ---
def run_single_scenario(scenario, dataset, total_inferences, max_consultation_turns, scenario_idx, AGENT_CONFIG):
    patient_agent = PatientAgent(scenario=scenario)

    doctor_agent = DoctorAgent(scenario=scenario, max_infs=total_inferences, prompt=DOCTOR_PROMPTS[AGENT_CONFIG["prompt_type"]])

    # Instantiation by config
    meas_agent = MeasurementAgent(scenario=scenario) if AGENT_CONFIG["use_measurement"] else NullMeasurmentAgent()

    available_tests = scenario.get_available_tests()
    run_log = {
        "timestamp": datetime.now(),
        "model": MODEL_NAME,
        "dataset": dataset,
        "scenario_id": scenario_idx,
        "max_patient_turns": total_inferences,
        "max_consultation_turns": max_consultation_turns,
        "correct_diagnosis": scenario.diagnosis_information(),
        "dialogue_history": [],
        "requested_tests": [],
        "tests_requested_count": 0,
        "available_tests": available_tests,
        "determined_specialist": None,
        "consultation_analysis": {},
        "final_doctor_diagnosis": None,
        "is_correct": None,
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

        if ((AGENT_CONFIG["use_specialist"] and state == "consultation_needed") or turn == total_inferences):
             print("\nPatient interaction phase complete.")
             break

        time.sleep(0.5)

    run_log["tests_requested_count"] = len(run_log["requested_tests"])
    run_log["tests_left_out"] = list(set(available_tests) - set(run_log["requested_tests"]))
    print(f"Total tests requested during patient interaction: {run_log['tests_requested_count']}")
    print(f"Tests left out: {run_log['tests_left_out']}")

    if AGENT_CONFIG["use_specialist"]:
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

            time.sleep(0.5)

    # --- Final Diagnosis Phase ---
    print("\n--- Phase 4: Final Diagnosis ---")
    final_diagnosis_full = doctor_agent.get_final_diagnosis()
    if "DIAGNOSIS READY:" in final_diagnosis_full:
         final_diagnosis_text = final_diagnosis_full.split("DIAGNOSIS READY:", 1)[-1].strip()
    else:
         final_diagnosis_text = "No diagnosis provided in correct format."

    print(f"\nFinal Diagnosis by Doctor: {final_diagnosis_text}")
    print(f"Correct Diagnosis: {scenario.diagnosis_information()}")

    is_correct = compare_results(final_diagnosis_text, scenario.diagnosis_information())
    print(f"Scenario {scenario_idx}: Diagnosis was {'CORRECT' if is_correct else 'INCORRECT'}")

    run_log["final_doctor_diagnosis"] = final_diagnosis_text
    run_log["is_correct"] = is_correct

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

    return run_log, run_log.get("is_correct", False)

def run_experiment_three(dataset, total_inferences, consultation_turns, max_scenarios=1):
    """Run a single config-dataset combination test"""
 
    # Create a list of scenarios to run
    scenarios_to_process = [{"name": "Base-Case", "use_measurement": True, "use_specialist": True, "prompt_type": "BASELINE_PROMPT"}, 
                            {"name": "Augmented-Doctor", "use_measurement": True, "use_specialist": False, "prompt_type": "AUGMENTED_DOCTOR_PROMPT"}, 
                            {"name": "Doctoral-Team", "use_measurement": False, "use_specialist": True, "prompt_type": "DOCTOR_TEAM_PROMPT"}, 
                            {"name": "Minimalist", "use_measurement": False, "use_specialist": False, "prompt_type": "MINIMALIST_PROMPT"}]
    
    scenario_loader = ScenarioLoader(dataset=dataset)
    scenarios_to_run = len(scenarios_to_process)    
        
    total_simulated_current_session = 0 
    total_correct_current_session = 0
    
    scenario_idx = 0
    
    for config in scenarios_to_process:
        config_name = config["name"]

        log_file = get_log_file(dataset, config_name)
        completed_scenario_ids = get_completed_scenarios(log_file)

        print(f"\n=== Testing {config_name} configuration on {dataset} dataset ===")
        print(f"Log file: {log_file}")
        print(f"Already completed scenario IDs: {len(completed_scenario_ids)}")
        print(f"Scenarios to run in this session: {len(scenarios_to_process)} of {scenarios_to_run} total planned")
        print(f"\n--- Running Scenario {scenario_idx + 1}/{scenarios_to_run} with {config_name} configuration ---")

        for scenario_idx in range(min(NUM_SCENARIOS, max_scenarios)):
            if scenario_idx in completed_scenario_ids:
                print(f"Completed, skipping scenario: {scenario_idx}", scenario_idx)
                continue

            scenario = scenario_loader.get_scenario(id=scenario_idx)

            if scenario is None:
                print(f"Error loading scenario {scenario_idx}, skipping.")
                continue

            
            total_simulated_current_session += 1
            run_log, is_correct = run_single_scenario(
                scenario, dataset, total_inferences, consultation_turns, scenario_idx, config 
            )

            if is_correct:
                total_correct_current_session += 1

            log_scenario_data(run_log, log_file)
            print(f"Tests requested in Scenario {scenario_idx + 1}: {run_log.get('requested_tests', [])}")
            
        # Update progress
        if total_simulated_current_session > 0:
            accuracy_current_session = (total_correct_current_session / total_simulated_current_session) * 100
            print(f"\nCurrent Accuracy for this session ({config_name} configuration on {dataset}): {accuracy_current_session:.2f}% ({total_correct_current_session}/{total_simulated_current_session})")
            
            # Calculate overall progress including previously completed scenarios
            overall_completed_count = len(completed_scenario_ids) + total_simulated_current_session
            overall_correct_count = total_correct_current_session
            
            overall_accuracy_so_far = (overall_correct_count / overall_completed_count) * 100 if overall_completed_count > 0 else 0
            print(f"Overall Progress for {config_name} on {dataset}: {overall_completed_count}/{scenarios_to_run} scenarios completed. Overall Accuracy: {overall_accuracy_so_far:.2f}% ({overall_correct_count}/{overall_completed_count})")
    
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
        
        print(f"\n=== Results for {config_name} configuration on {dataset} dataset ===")
        print(f"Total Scenarios Logged: {actual_entries_in_log} (planned: {scenarios_to_run}, completed this/prev sessions: {final_completed_count})")
        print(f"Final Accuracy: {final_accuracy:.2f}% ({correct_count_total}/{actual_entries_in_log})")
    
    return final_completed_count >= scenarios_to_run

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
    #datasets_to_test = ['MedQA', 'NEJM'] if args.dataset == 'all' else [args.dataset]
    datasets_to_test = ['MedQA'] if args.dataset == 'all' else [args.dataset]
    
    # Determine which configs to test
    scenarios_to_process = [{"name": "Base-Case", "use_measurement": True, "use_specialist": True, "prompt_type": "BASELINE_PROMPT"}, 
                            {"name": "Augmented-Doctor", "use_measurement": True, "use_specialist": False, "prompt_type": "AUGMENTED_DOCTOR_PROMPT"}, 
                            {"name": "Doctoral-Team", "use_measurement": False, "use_specialist": True, "prompt_type": "DOCTOR_TEAM_PROMPT"}, 
                            {"name": "Minimalist", "use_measurement": False, "use_specialist": False, "prompt_type": "MINIMALIST_PROMPT"}]


    print(f"Base settings: {args.scenarios} scenarios per combination, {TOTAL_INFERENCES} patient interactions, {CONSULTATION_TURNS} consultation turns")
    
    # Create summary report structures
    summary = {
        "start_time": datetime.now().isoformat(),
        "completed_combinations": 0,
        "total_combinations": len(datasets_to_test) * len(scenarios_to_process),
        "results_by_combination": {}
    }
    
    # Run each combination
    for dataset in datasets_to_test:
        
        try:
            completed = run_experiment_three(
                dataset, TOTAL_INFERENCES, CONSULTATION_TURNS, 10
            )                
        except Exception as e:
            import traceback
            print(f"Error running {dataset} with config: {e}")
            traceback.print_exc()                
            # Continue with next combination even if this one fails

        for config in scenarios_to_process:

            print(f"\n\n{'='*80}")
            print(f"TESTING: Dataset={dataset}, Config={config["name"]}")
            print(f"{'='*80}")

            # Update summary
            combination_key = f"{dataset}_{config["name"]}"
            log_file = get_log_file(dataset, config["name"])
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    results = json.load(f)
                    correct_count = sum(1 for entry in results if entry.get("is_correct", False))
                    total_count = len(results)
                    
                    summary["results_by_combination"][combination_key] = {
                        "completed": completed,
                        "scenarios_run": total_count,
                        "correct_diagnoses": correct_count,
                        "accuracy": (correct_count / total_count) * 100 if total_count > 0 else 0
                    }
            
            if completed:
                summary["completed_combinations"] += 1
            
    
    # Save summary report
    summary["end_time"] = datetime.now().isoformat()
    summary["total_duration_seconds"] = (datetime.fromisoformat(summary["end_time"]) - 
                                        datetime.fromisoformat(summary["start_time"])).total_seconds()
    
    with open(os.path.join(BASE_LOG_DIR, "config_testing_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n\n=== CONFIG TESTING COMPLETE ===")
    print(f"Completed {summary['completed_combinations']}/{summary['total_combinations']} combinations")
    print(f"Total duration: {summary['total_duration_seconds']/3600:.2f} hours")
    print(f"Full results saved to {os.path.join(BASE_LOG_DIR, 'config_testing_summary.json')}")


if __name__ == "__main__":
    main()
