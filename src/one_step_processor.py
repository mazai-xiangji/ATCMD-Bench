# src/one_step_processor.py
import logging
import json
from typing import Dict, Any, Optional, List

from openai import OpenAI
# Use the generic response getter and client initializer
from .llm_clients import initialize_openai_client, get_llm_response
# Use the generic evaluation parser and prompt loader
from .utils import parse_expert_evaluation, load_prompt

logger = logging.getLogger(__name__)

def run_one_step_evaluation(case_data: Dict[str, Any], config, prompts: Dict[str, str]) -> Dict[str, Any]:
    """
    Runs the one-step diagnosis and evaluation for a single case.

    Args:
        case_data: The structured medical case data.
        config: The configuration object.
        prompts: Dictionary containing loaded 'one_step_doctor' and 'one_step_expert' prompts.

    Returns:
        A dictionary containing the result ('case_id', 'model_output', 'evaluation', 'status').
    """
    case_id = case_data.get('id', case_data.get('case_id', 'unknown')) # Get case ID early
    result = {
        "case_id": case_id,
        "status": "Processing",
        "model_output": None, # Key specific to one-step results
        "evaluation": None
    }

    # --- Initialize Clients (Ensure they are available) ---
    # Clients should be initialized in main.py, get them from config or pass them?
    # Assuming main.py initializes and we just need to ensure they exist.
    # Let's re-initialize here for robustness or if called standalone.
    test_client = initialize_openai_client(config.test_base_url, config.test_api_key)
    expert_client = initialize_openai_client(config.expert_base_url, config.expert_api_key)

    if not test_client:
        logger.error(f"Failed to initialize test client for case {result['case_id']}.")
        result["status"] = "Error: Test client init failed"
        return result
    if not expert_client:
         logger.error(f"Failed to initialize expert client for case {result['case_id']}.")
         result["status"] = "Error: Expert client init failed"
         return result


    # --- Prepare Doctor Input ---
    # Extract necessary info - Combine patient, consult, other info for doctor prompt
    patient_info = case_data.get('患者个人信息', {})
    consult_info = case_data.get('问诊信息', {})
    other_info = case_data.get('其余信息', {}) # Check this key name matches data
    # Format into a single string as expected by the one-step doctor prompt
    # Ensure proper serialization for complex nested data
    try:
         doctor_input_info = (
              f"患者个人信息：{json.dumps(patient_info, ensure_ascii=False)}\n"
              f"问诊信息：{json.dumps(consult_info, ensure_ascii=False)}\n"
              f"其他信息：{json.dumps(other_info, ensure_ascii=False)}"
         )
    except Exception as json_err:
         logger.error(f"Error serializing input info for one-step doctor prompt (case {case_id}): {json_err}")
         result["status"] = "Error: Input data serialization failed"
         return result

    try:
        doctor_prompt_template = prompts.get('one_step_doctor')
        if not doctor_prompt_template: raise ValueError("Prompt 'one_step_doctor' not found.")
        # Placeholder in one_step_doctor.txt should match 'patient_full_info'
        # Prepare messages - Use SYSTEM for instructions, USER for data
        # Extract system instructions from the loaded prompt if structured that way, or hardcode base instruction
        # Assuming the template contains the full system message + placeholder:
        # Split the template maybe? Or just format the whole thing as system?
        # Let's assume the template IS the system prompt and needs formatting:
        doctor_system_prompt_formatted = doctor_prompt_template.format(patient_full_info=doctor_input_info)
        # Use only SYSTEM role if the prompt structure implies it contains the task + data placeholder
        # Or, use SYSTEM + USER if template is just instructions:
        # doctor_messages = [
        #     {"role": "system", "content": doctor_prompt_template}, # Assuming template is instructions only
        #     {"role": "user", "content": doctor_input_info}         # Data provided in user message
        # ]
        # Based on provided prompt, it seems better to format into SYSTEM:
        doctor_messages = [ {"role": "system", "content": doctor_system_prompt_formatted} ]
        # The original one-step code used sys+user, let's stick to that for consistency:
        doctor_messages = [
            {"role": "system", "content": "作为一名经验丰富的老中医，你的任务是根据所提供的患者的信息进行中医的辩证论治，最后输出患者的诊断结果（包括病名和中医证型）和详细的诊断依据。诊断依据要符合中医的相关理论，清晰地解释症状与诊断之间的联系。"},
            {"role": "user", "content": doctor_input_info}
        ]

    except KeyError as e: # Catch placeholder errors specifically
        logger.error(f"Placeholder error in one-step doctor prompt (case {case_id}): {e}")
        result["status"] = "Error: Doctor prompt formatting failed (placeholder)"
        return result
    except Exception as e:
        logger.error(f"Error loading/formatting one-step doctor prompt (case {case_id}): {e}")
        result["status"] = "Error: Doctor prompt loading/formatting error"
        return result

    # --- Get Doctor Model Output ---
    logger.info(f"Requesting one-step diagnosis from model '{config.test_model_name}' for case {case_id}...")
    doctor_output = get_llm_response(
        client=test_client,
        model_name=config.test_model_name,
        messages=doctor_messages,
        config=config,
        is_test_model=True
    )

    if doctor_output is None: # Check explicitly for None
        logger.error(f"Failed to get one-step diagnosis output for case {case_id}.")
        result["status"] = "Error: Test model failed to respond"
        # Still return partial result if needed (e.g., just case_id and status)
        return result

    result["model_output"] = doctor_output
    logger.info(f"Received one-step diagnosis output for case {case_id}: {doctor_output[:150]}...")


    # --- Prepare Expert Input ---
    # Extract full info needed by expert (including diagnosis/basis from original case)
    diagnosis_info = case_data.get('诊断结果', {})
    diagnosis_basis = case_data.get('诊断依据', {})
    # Reconstruct expert_full_info string as needed by the one_step_expert prompt
    try:
        expert_case_info_str = (
            f"患者个人信息：{json.dumps(patient_info, ensure_ascii=False)}\n"
            f"问诊信息：{json.dumps(consult_info, ensure_ascii=False)}\n"
            f"其他信息：{json.dumps(other_info, ensure_ascii=False)}\n" # Check key name '其他信息' vs '其余信息'
            f"诊断结果：{json.dumps(diagnosis_info, ensure_ascii=False)}\n"
            f"诊断依据：\n{json.dumps(diagnosis_basis, ensure_ascii=False)}" # Ensure diagnosis_basis is represented well
        )
    except Exception as json_err:
        logger.error(f"Error serializing expert info for one-step expert prompt (case {case_id}): {json_err}")
        result["status"] = "Error: Expert input data serialization failed"
        return result # Return result with model_output but failed eval prep


    try:
        expert_prompt_template = prompts.get('one_step_expert')
        if not expert_prompt_template: raise ValueError("Prompt 'one_step_expert' not found.")
        # Placeholders expected: {expert_full_info}, {doctor_output}
        # The json_format is hardcoded inside the one-step expert prompt itself.
        expert_prompt_filled = expert_prompt_template.format(
            expert_full_info=expert_case_info_str,
            doctor_output=doctor_output
        )
    except KeyError as e:
        logger.error(f"Missing placeholder in one-step expert prompt (case {case_id}): {e}")
        result["status"] = "Error: Expert prompt formatting failed (placeholder)"
        return result
    except Exception as e:
        logger.error(f"Error formatting one-step expert prompt (case {case_id}): {e}")
        result["status"] = "Error: Expert prompt formatting error"
        return result

    expert_messages = [{"role": "user", "content": expert_prompt_filled}]

    # --- Get Expert Evaluation ---
    logger.info(f"Requesting one-step evaluation from expert model '{config.expert_model_name}' for case {case_id}...")
    expert_response_content = get_llm_response(
        client=expert_client,
        model_name=config.expert_model_name,
        messages=expert_messages,
        config=config,
        is_test_model=False # Expert model is not the one under test
    )

    if expert_response_content is None: # Check explicitly for None
        logger.error(f"Failed to get one-step evaluation response for case {case_id}.")
        result["status"] = "Completed with Evaluation Error (No Response)"
        # Keep the model_output even if evaluation fails
        return result

    # --- Parse Expert Evaluation ---
    evaluation_data = parse_expert_evaluation(expert_response_content)
    result["evaluation"] = evaluation_data # Store parsed result (even if it's an error dict)

    if "error" in evaluation_data:
        logger.error(f"Failed to parse one-step evaluation for case {case_id}: {evaluation_data.get('error')}")
        result["status"] = "Completed with Evaluation Parsing Error"
    else:
        logger.info(f"Successfully evaluated one-step output for case {case_id}.")
        result["status"] = "Completed"

    return result