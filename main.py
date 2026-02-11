import logging
import os
from typing import List, Dict, Any

# Import from project modules
from config import load_config
from src.utils import load_json, save_json, load_prompt
# Import shared client functions and specific mode processors
from src.llm_clients import initialize_openai_client
from src.dialogue_manager import DialogueSimulator
from src.evaluator import DialogueEvaluator
from src.one_step_processor import run_one_step_evaluation # Import the new function

# Setup logging
# Consider using RotatingFileHandler or TimedRotatingFileHandler for large runs
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
# Root Logger Setup
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) # Set root level to INFO
root_logger.addHandler(console_handler)
# Optional: File Handler
# file_handler = logging.FileHandler("evaluation.log", mode='a')
# file_handler.setFormatter(log_formatter)
# root_logger.addHandler(file_handler)


def main():
    """Main function to run the medical dialogue simulation and evaluation."""
    logging.info("=================================================")
    logging.info("Starting the medical dialogue evaluation process...")
    logging.info(f"Current time: {__import__('datetime').datetime.now()}") # Use current time from context
    logging.info("=================================================")


    # 1. Load Configuration
    config = load_config()
    logging.info(f"Configuration loaded.")
    # Log key configurations (avoid logging sensitive keys directly in production)
    logging.info(f"Mode: {config.mode}")
    logging.info(f"Input cases: {config.structed_case_path}")
    logging.info(f"Output results: {config.output_json_path}")
    logging.info(f"Test Model Name: {config.test_model_name}")
    logging.info(f"Expert Model Name: {config.expert_model_name}")
    if config.mode == 'multi-turn':
        logging.info(f"Simulation Model Name: {config.sim_model_name}")
        logging.info(f"Max Dialogue Turns: {config.max_dialogue_turns}")


    # 2. Load Prompts based on mode
    prompts = {}
    required_prompts = []
    try:
        if config.mode == 'multi-turn':
            required_prompts = [
                ("patient", config.patient_prompt_file),
                ("doctor", config.doctor_prompt_file),
                ("assistant", config.assistant_prompt_file),
                ("router", config.router_prompt_file),
                ("expert", config.expert_prompt_file),
            ]
            logging.info("Loading prompts for multi-turn mode...")
        elif config.mode == 'one-step':
            required_prompts = [
                ("one_step_doctor", config.one_step_doctor_prompt_file),
                ("one_step_expert", config.one_step_expert_prompt_file),
            ]
            logging.info("Loading prompts for one-step mode...")
        else:
            # This case should be handled by argparse choices, but included defensively
            raise ValueError(f"Invalid mode specified: {config.mode}")

        for name, filename in required_prompts:
             prompts[name] = load_prompt(filename, prompts_dir=config.prompts_dir)
             logging.debug(f"Loaded prompt '{name}' from '{filename}'.")
        logging.info(f"Successfully loaded {len(prompts)} prompts for mode '{config.mode}'.")

    except Exception as e:
        logging.error(f"Failed to load prompts for mode '{config.mode}' from '{config.prompts_dir}': {e}. Exiting.")
        return

    # 3. Initialize LLM Clients
    sim_client = None
    if config.mode == 'multi-turn': # Only initialize sim_client if needed
         logging.info("Initializing Simulation LLM client...")
         sim_client = initialize_openai_client(config.sim_base_url, config.sim_api_key)
         if not sim_client:
              logging.error("Failed to initialize Simulation client (required for multi-turn mode). Exiting.")
              return

    logging.info("Initializing Expert LLM client...")
    expert_client = initialize_openai_client(config.expert_base_url, config.expert_api_key)
    logging.info("Initializing Test Model LLM client...")
    test_client = initialize_openai_client(config.test_base_url, config.test_api_key)

    # Check clients needed for the selected mode
    if not test_client:
        logging.error("Failed to initialize Test client (required for all modes). Exiting.")
        return
    if not expert_client:
        logging.error("Failed to initialize Expert client (required for all modes). Exiting.")
        return

    logging.info("Required LLM clients initialized successfully.")

    # Prepare clients dict for multi-turn simulator if needed
    clients = {"sim": sim_client, "test": test_client}

    # 4. Initialize Core Components (Mode dependent)
    simulator = None
    evaluator = None
    if config.mode == 'multi-turn':
        try:
            simulator = DialogueSimulator(config, clients, prompts)
            evaluator = DialogueEvaluator(config, expert_client, prompts['expert'])
            logging.info("Multi-turn Simulator and Evaluator initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize multi-turn components: {e}. Exiting.")
            return
    elif config.mode == 'one-step':
        logging.info("One-step mode selected. Using one_step_processor function.")


    # 5. Load Data and Results
    try:
        all_cases = load_json(config.structed_case_path)
        if not isinstance(all_cases, list):
             logging.error(f"Structured case file did not contain a list: {config.structed_case_path}")
             return
    except Exception as e:
        logging.error(f"Failed to load structured cases from {config.structed_case_path}: {e}. Exiting.")
        return

    results_list: List[Dict[str, Any]] = []
    if os.path.exists(config.output_json_path):
        try:
            results_list = load_json(config.output_json_path)
            if not isinstance(results_list, list):
                logging.warning(f"Output file ({config.output_json_path}) exists but is not a list. Starting fresh.")
                results_list = []
        except Exception as e:
             logging.warning(f"Could not load existing results from {config.output_json_path}: {e}. Starting fresh.")
             results_list = []


    start_index = len(results_list)
    processed_ids = {res.get('case_id') for res in results_list if res.get('case_id')} # Track processed IDs
    logging.info(f"Loaded {len(all_cases)} cases. Found {len(results_list)} existing results.")
    logging.info(f"Will process cases starting from index {start_index} (if not already processed by ID).")


    # 6. Process Cases
    cases_processed_this_run = 0
    cases_skipped = 0
    for i, case_data in enumerate(all_cases):
        # Determine case ID
        case_id = case_data.get('id', case_data.get('case_id'))
        if case_id is None: # Assign index if no ID found
             case_id = f"case_index_{i}"
             logging.debug(f"Case at index {i} missing 'id' or 'case_id', assigning temporary ID: {case_id}")

        # Skip if already processed (based on ID)
        if case_id in processed_ids:
            logging.debug(f"Skipping Case {i+1}/{len(all_cases)} (ID: {case_id}) - Already in results.")
            cases_skipped += 1
            continue

        logging.info(f"--- Processing Case {i+1}/{len(all_cases)} (ID: {case_id}) ---")
        final_result = None # Initialize result for this case

        # --- Execute based on mode ---
        try:
            if config.mode == 'one-step':
                # Pass only necessary clients and prompts for this mode
                final_result = run_one_step_evaluation(case_data, config, prompts)

            elif config.mode == 'multi-turn':
                if not simulator or not evaluator: # Safety check
                     logging.error("Simulator or Evaluator not initialized for multi-turn mode.")
                     final_result = {"case_id": case_id, "status": "Error: Components not initialized"}
                else:
                    dialogue_history = simulator.run_simulation(case_data)

                    if dialogue_history is None:
                        logging.error(f"Simulation failed for case {case_id}.")
                        final_result = {
                            "case_id": case_id, "status": "Simulation Failed",
                            "dialogue_history": None, "evaluation": None
                        }
                    else:
                        logging.info(f"Simulation completed. Dialogue length: {len(dialogue_history)} messages.")
                        evaluation_result = evaluator.evaluate_dialogue(case_data, dialogue_history)

                        # Assign 'dialogue_history' only if simulation was successful
                        current_result = {
                            "case_id": case_id,
                            "dialogue_history": dialogue_history
                        }

                        if evaluation_result is None or "error" in evaluation_result:
                            logging.error(f"Evaluation failed for case {case_id}.")
                            current_result["status"] = "Evaluation Failed"
                            current_result["evaluation"] = evaluation_result # Store error info
                        else:
                            logging.info(f"Evaluation completed for case {case_id}.")
                            current_result["status"] = "Completed"
                            current_result["evaluation"] = evaluation_result
                        final_result = current_result # Assign the fully formed dict
            else:
                 logging.error(f"Invalid mode '{config.mode}' encountered during processing.")
                 final_result = {"case_id": case_id, "status": f"Error: Invalid mode {config.mode}"}

        except Exception as e:
            logging.exception(f"Unhandled exception processing case {case_id} in mode '{config.mode}': {e}")
            # Ensure basic info is saved even on unexpected error
            final_result = {
                 "case_id": case_id,
                 "status": f"Unhandled Exception",
                 "error_message": str(e),
                 # Include relevant keys if they exist before exception
                 "model_output": final_result.get("model_output") if final_result else None,
                 "dialogue_history": final_result.get("dialogue_history") if final_result else None,
                 "evaluation": final_result.get("evaluation") if final_result else None,
            }


        # --- Append and Save Result ---
        if final_result: # Ensure we have a result dict
            # Ensure case_id is present before appending
            if 'case_id' not in final_result: final_result['case_id'] = case_id

            results_list.append(final_result)
            cases_processed_this_run += 1
            try:
                save_json(results_list, config.output_json_path)
                logging.debug(f"Result for case {case_id} saved. Total results: {len(results_list)}.")
            except Exception as e:
                # Log critical error but continue processing other cases
                logging.error(f"CRITICAL: Failed to save results to {config.output_json_path} after processing case {case_id}: {e}")
                # Optional: Implement backup saving mechanism here
        else:
            logging.error(f"No result generated for case {case_id}, skipping save for this case.")


    logging.info("-------------------------------------------------")
    logging.info(f"Processing finished.")
    logging.info(f"Cases processed in this run: {cases_processed_this_run}")
    logging.info(f"Cases skipped (already processed): {cases_skipped}")
    logging.info(f"Total results saved: {len(results_list)}")
    logging.info(f"Results saved to: {config.output_json_path}")
    logging.info("=================================================")


if __name__ == "__main__":
    main()