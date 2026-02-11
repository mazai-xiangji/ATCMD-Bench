# src/dialogue_manager.py
import logging
import json
from typing import List, Dict, Any, Tuple, Optional

# Use the generic response getter from llm_clients
from .llm_clients import OpenAI, get_llm_response
# Keep utils import if needed, e.g., for markers, though response extraction is in llm_clients
# from .utils import extract_final_response

# Use logging configured in main.py
logger = logging.getLogger(__name__)

# Constants for roles and markers
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_PATIENT = "患者"
ROLE_DOCTOR_ASSISTANT = "助理"
ROLE_EXPERT = "专家"
ASSISTANT_MARKER = "<对助理>" # Marker used by doctor model to signal talking to assistant


class DialogueSimulator:
    """Handles the multi-turn dialogue simulation process."""
    def __init__(self, config, clients, prompts):
        """
        Initializes the DialogueSimulator.

        Args:
            config: Configuration object (from load_config).
            clients: Dictionary of initialized OpenAI clients {'sim': sim_client, 'test': test_client}.
                     'sim' client is essential for multi-turn mode.
            prompts: Dictionary of loaded prompt templates for multi-turn mode.
        """
        self.config = config
        self.clients = clients
        self.prompts = prompts
        self.sim_client = clients.get('sim') # Get sim client, might be None if not initialized
        self.test_client = clients.get('test')
        if not self.test_client:
             # Test client is always needed, raise error if missing
             raise ValueError("Test client must be initialized.")
        if not self.sim_client:
             # Log a warning, run_simulation will fail if called without it
             logger.warning("Simulation client (sim_client) is not initialized. Multi-turn mode requires it.")


    def _determine_next_role(self, doctor_output: str) -> str:
        """Uses the router LLM (sim_client) to determine the next role."""
        # Ensure sim_client is available for routing
        if not self.sim_client:
             logger.error("Simulation client needed for routing is not available. Defaulting to Expert.")
             return ROLE_EXPERT # Cannot route without sim client

        router_prompt_template = self.prompts.get('router')
        if not router_prompt_template:
             logger.error("Router prompt template not found. Defaulting to Patient.")
             return ROLE_PATIENT

        try:
             # Ensure router prompt uses a known placeholder, e.g., 'dialogue_context' or 'latest_utterance'
             router_input_prompt = router_prompt_template.format(dialogue_context=doctor_output)
        except KeyError as e:
             logger.error(f"Placeholder error in router prompt template: {e}. Defaulting to Patient.")
             return ROLE_PATIENT
        except Exception as e:
             logger.error(f"Error formatting router prompt: {e}. Defaulting to Patient.")
             return ROLE_PATIENT

        # Router often just needs the latest turn + rules embedded in its system prompt
        router_messages = [{"role": ROLE_USER, "content": router_input_prompt}]

        # Use the generic function to call the router model
        router_output = get_llm_response(
            client=self.sim_client,
            model_name=self.config.sim_model_name,
            messages=router_messages,
            config=self.config, # Pass config object
            is_test_model=False # Router is not the model under test
        )

        if router_output:
            # Simple keyword check - make this more robust if needed (e.g., regex, JSON output)
            router_output_lower = router_output.lower() # Case-insensitive check
            if ROLE_PATIENT in router_output_lower:
                logger.debug("Router decided: Patient")
                return ROLE_PATIENT
            elif ROLE_DOCTOR_ASSISTANT in router_output_lower:
                logger.debug("Router decided: Assistant")
                return ROLE_DOCTOR_ASSISTANT
            elif ROLE_EXPERT in router_output_lower:
                logger.debug("Router decided: Expert")
                return ROLE_EXPERT
            else:
                # Handle potential ambiguous outputs
                logger.warning(f"Router output ('{router_output}') unrecognized or ambiguous. Defaulting to Patient.")
                return ROLE_PATIENT
        else:
            logger.error("Router failed to provide output. Defaulting to Patient.")
            return ROLE_PATIENT


    def run_simulation(self, case_data: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """
        Runs the multi-turn dialogue simulation for a single case.
        Requires self.sim_client and self.test_client to be initialized.
        """
        if not self.sim_client:
             logger.error("Simulation client (sim_client) is required for run_simulation but not initialized.")
             return None
        if not self.test_client:
             logger.error("Test client is required for run_simulation but not initialized.")
             return None

        # 1. Prepare initial data and format prompts
        patient_info = case_data.get('患者个人信息', {})
        consult_info = case_data.get('问诊信息', {})
        other_info = case_data.get('其余信息', {})

        # Format info for prompts - ensure complex objects are serialized (e.g., using json.dumps)
        patient_full_info = f"患者个人信息：{json.dumps(patient_info, ensure_ascii=False)}\n问诊信息：{json.dumps(consult_info, ensure_ascii=False)}"
        assistant_full_info = f"助理所掌握的患者信息：{json.dumps(consult_info, ensure_ascii=False)}\n其他信息：{json.dumps(other_info, ensure_ascii=False)}" # Corrected key based on one-step code

        try:
            patient_prompt_template = self.prompts.get('patient')
            doctor_prompt_template = self.prompts.get('doctor')
            assistant_prompt_template = self.prompts.get('assistant')
            if not all([patient_prompt_template, doctor_prompt_template, assistant_prompt_template]):
                 raise ValueError("One or more required multi-turn prompt templates are missing.")

            patient_system_prompt = patient_prompt_template.format(patient_full_info=patient_full_info)
            doctor_system_prompt = doctor_prompt_template # Doctor prompt usually static system instructions
            assistant_system_prompt = assistant_prompt_template.format(assistant_full_info=assistant_full_info)
        except KeyError as e:
            logger.error(f"Missing placeholder in system prompt template: {e}. Cannot run simulation.")
            return None
        except Exception as e:
            logger.error(f"Error formatting system prompts: {e}. Cannot run simulation.")
            return None


        # 2. Initialize message histories
        doctor_messages = [{"role": ROLE_SYSTEM, "content": doctor_system_prompt}]
        patient_messages = [{"role": ROLE_SYSTEM, "content": patient_system_prompt}]
        assistant_messages = [{"role": ROLE_SYSTEM, "content": assistant_system_prompt}]


        # 3. Initial Turn (Doctor usually asks first, patient responds)
        initial_doctor_utterance = "你好，请问有哪里不舒服的吗" # Standard opening
        patient_messages.append({"role": ROLE_USER, "content": initial_doctor_utterance})
        logger.info(f"Turn 0: Doctor initiates -> '{initial_doctor_utterance}'")

        # Get initial patient response using the generic function from llm_clients
        patient_output = get_llm_response(self.sim_client, self.config.sim_model_name, patient_messages, self.config, is_test_model=False)
        if not patient_output:
            logger.error("Failed to get initial patient response. Ending simulation.")
            return None # Cannot start simulation without initial response

        logger.info(f"Turn 0: Patient responds -> '{patient_output[:100]}...'")
        patient_messages.append({"role": ROLE_ASSISTANT, "content": patient_output})
        # Doctor receives patient's first response to start the main loop
        doctor_messages.append({"role": ROLE_USER, "content": patient_output})


        # 4. Dialogue Loop
        turn_count = 0
        max_turns = self.config.max_dialogue_turns
        while turn_count < max_turns:
            turn_count += 1
            logger.info(f"--- Turn {turn_count}/{max_turns} ---")

            # Doctor's turn (Model Under Test) - Use generic function
            # Log only essential parts of history if it gets too long
            log_history = doctor_messages[-min(len(doctor_messages), 5):] # Log last 5 messages max
            logger.debug(f"Doctor messages history (Turn {turn_count}, last {len(log_history)}):\n{json.dumps(log_history, ensure_ascii=False, indent=2)}")
            doctor_output = get_llm_response(self.test_client, self.config.test_model_name, doctor_messages, self.config, is_test_model=True)

            if doctor_output is None: # Check for None explicitly
                logger.error(f"Failed to get doctor response on turn {turn_count}. Ending simulation.")
                # Add error message to history for context before breaking
                doctor_messages.append({"role": ROLE_ASSISTANT, "content": "[ERROR: Doctor Model Failed to Respond]"})
                break # End simulation

            logger.info(f"Turn {turn_count}: Doctor output -> '{doctor_output[:150]}...'")
            # Add doctor's *intended* full response (including marker if any) to their history first
            doctor_messages.append({"role": ROLE_ASSISTANT, "content": doctor_output})

            # Router decides next step based on doctor's *full* output
            next_role = self._determine_next_role(doctor_output)

            # Handle flow based on router decision
            if next_role == ROLE_EXPERT:
                logger.info("Router directed to Expert. Dialogue concluded by router.")
                # Doctor's last message is already added.
                break # Exit loop

            # Prepare query for the next participant (Patient or Assistant)
            is_to_assistant = doctor_output.strip().startswith(ASSISTANT_MARKER)
            # Query sent to Patient/Assistant should NOT contain the marker
            actual_query = doctor_output.replace(ASSISTANT_MARKER, "", 1).strip() if is_to_assistant else doctor_output

            # Patient/Assistant Turn - Use generic function
            if next_role == ROLE_DOCTOR_ASSISTANT:
                logger.debug(f"Passing query to Assistant: '{actual_query[:100]}...'")
                assistant_messages.append({"role": ROLE_USER, "content": actual_query})
                # Use generic function for assistant response
                assistant_output = get_llm_response(self.sim_client, self.config.sim_model_name, assistant_messages, self.config, is_test_model=False)
                if assistant_output is None:
                    logger.error(f"Failed to get assistant response on turn {turn_count}. Ending simulation.")
                    doctor_messages.append({"role": ROLE_USER, "content":"[ERROR: Assistant Failed to Respond]"})
                    break # End simulation

                logger.info(f"Turn {turn_count}: Assistant responds -> '{assistant_output[:100]}...'")
                assistant_messages.append({"role": ROLE_ASSISTANT, "content": assistant_output})
                # Doctor receives assistant's response for the next turn
                doctor_messages.append({"role": ROLE_USER, "content": assistant_output})

            else: # Default or ROLE_PATIENT
                logger.debug(f"Passing query to Patient: '{actual_query[:100]}...'")
                patient_messages.append({"role": ROLE_USER, "content": actual_query})
                # Use generic function for patient response
                patient_output = get_llm_response(self.sim_client, self.config.sim_model_name, patient_messages, self.config, is_test_model=False)
                if patient_output is None:
                    logger.error(f"Failed to get patient response on turn {turn_count}. Ending simulation.")
                    doctor_messages.append({"role": ROLE_USER, "content":"[ERROR: Patient Failed to Respond]"})
                    break # End simulation

                logger.info(f"Turn {turn_count}: Patient responds -> '{patient_output[:100]}...'")
                patient_messages.append({"role": ROLE_ASSISTANT, "content": patient_output})
                # Doctor receives patient's response for the next turn
                doctor_messages.append({"role": ROLE_USER, "content": patient_output})


        # 5. Handle Max Turns Reached - Force Diagnosis from Test Model
        if turn_count >= max_turns:
            logger.warning(f"Maximum dialogue turns ({max_turns}) reached. Forcing final diagnosis from doctor model.")
            # Create summary prompt - Consider making this a template in prompts/
            # Exclude system prompt, maybe summarize if history is extremely long
            try:
                dialogue_summary_for_prompt = json.dumps(doctor_messages[1:], ensure_ascii=False, indent=2)
            except Exception as json_err:
                 logger.error(f"Could not serialize doctor messages for final prompt: {json_err}")
                 dialogue_summary_for_prompt = "[Error serializing dialogue history]"

            # This prompt should ideally be in prompts/ directory too
            final_diagnosis_prompt_text = (
                f"请根据你跟患者/助理的对话内容，推断出患者可能的疾病，"
                f"诊断结果包括病名和中医证型，同时给出详细的诊断依据。"
                f"对话内容如下：\n{dialogue_summary_for_prompt}"
            )
            # Use only USER role for this final, direct instruction
            final_messages = [{"role": ROLE_USER, "content": final_diagnosis_prompt_text}]

            # Call the Test model one last time using the generic function
            final_doctor_output = get_llm_response(
                 self.test_client,
                 self.config.test_model_name,
                 final_messages,
                 self.config,
                 is_test_model=True
            )

            if final_doctor_output:
                 logger.info(f"Final Forced Diagnosis (Max Turns): {final_doctor_output[:150]}...")
                 # Add this forced diagnosis to the history
                 doctor_messages.append({"role": ROLE_ASSISTANT, "content": final_doctor_output})
            else:
                 logger.error("Failed to get final diagnosis from doctor after max turns.")
                 doctor_messages.append({"role": ROLE_ASSISTANT, "content": "[ERROR: Failed to generate final diagnosis after max turns]"})


        # Return the complete doctor-centric dialogue history (excluding system prompt)
        # Ensure it's not empty before slicing
        return doctor_messages[1:] if len(doctor_messages) > 1 else []