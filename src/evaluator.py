# src/evaluator.py
import logging
import json
from typing import List, Dict, Any, Optional

# Use generic client functions and utils
from .llm_clients import OpenAI, get_llm_response
from .utils import parse_expert_evaluation

# Use logging configured in main.py
logger = logging.getLogger(__name__)

# Define the expected JSON format structure for the MULTI-TURN expert prompt
# This structure matches the original multi-turn expert prompt expectations
MULTI_TURN_JSON_FORMAT = """{
    "问诊评分": {
        "reason": "给出该分数的简要理由",
        "score": "评分（1-10）"
    },
    "诊断依据评分": {
        "reason": "给出该分数的简要理由",
        "score": "评分（1-10）"
    },
    "诊断结果评分": {
        "reason": "给出该分数的简要理由",
        "score": "评分（1-10）"
    }
}"""

class DialogueEvaluator:
    """Handles the evaluation process for the MULTI-TURN dialogue mode."""
    def __init__(self, config, expert_client: Optional[OpenAI], expert_prompt_template: str):
        """
        Initializes the DialogueEvaluator for multi-turn evaluation.

        Args:
            config: Configuration object.
            expert_client: Initialized OpenAI client for the expert model.
            expert_prompt_template: The loaded MULTI-TURN expert prompt template string.
        """
        self.config = config
        self.expert_client = expert_client
        self.expert_prompt_template = expert_prompt_template
        # Use the specific format expected by the multi-turn expert prompt
        self.json_format_str = MULTI_TURN_JSON_FORMAT

    def evaluate_dialogue(self, case_data: Dict[str, Any], dialogue_history: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        Sends the multi-turn dialogue to the expert LLM for evaluation.

        Args:
            case_data: The original structured case data.
            dialogue_history: The recorded dialogue history (doctor's perspective).

        Returns:
            A dictionary containing the parsed evaluation scores and reasons,
            or a dictionary with an 'error' key if evaluation fails.
        """
        if not self.expert_client:
            logger.error("Expert client is not initialized. Cannot perform multi-turn evaluation.")
            return {"error": "Expert client not initialized"}
        if not dialogue_history:
            logger.warning("Dialogue history is empty for multi-turn evaluation. Skipping.")
            return {"error": "Empty dialogue history for evaluation"}

        # --- Prepare Expert Information String ---
        # Include all relevant parts from case_data needed by the multi-turn expert prompt
        parts = []
        # Define keys expected by the multi-turn expert prompt's {expert_full_info} placeholder
        keys_to_include = ['患者个人信息', '问诊信息', '其余信息', '诊断结果', '诊断依据']
        for key in keys_to_include:
             value = case_data.get(key, {}) # Default to empty dict if key missing
             # Format value appropriately (e.g., json.dumps for dicts/lists)
             formatted_value = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
             parts.append(f"{key}：{formatted_value}")
        expert_full_info = "\n".join(parts)
        # Add specific formatting like the original code if necessary:
        # Example reconstruction closer to original:
        patient_info=case_data.get('患者个人信息', {})
        consult_info=case_data.get('问诊信息', {})
        other_info=case_data.get('其余信息', {})
        diagnosis_info=case_data.get('诊断结果', {})
        diagnosis_basis=case_data.get('诊断依据', {})
        expert_full_info = (
             f"患者个人信息：{json.dumps(patient_info, ensure_ascii=False)}\n"
             f"问诊信息：{json.dumps(consult_info, ensure_ascii=False)}\n"
             f"其余信息：{json.dumps(other_info, ensure_ascii=False)}\n" # Fixed original typo here
             f"诊断结果：{json.dumps(diagnosis_info, ensure_ascii=False)}\n" # Fixed original typo here
             f"诊断依据：\n{json.dumps(diagnosis_basis, ensure_ascii=False)}" # Fixed original typo here
         )


        # --- Format Dialogue History ---
        try:
            # Exclude potential error messages added at the end by simulator
            clean_history = [msg for msg in dialogue_history if not msg['content'].startswith("[ERROR:")]
            dialogue_json_str = json.dumps(clean_history, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Error converting dialogue history to JSON for evaluation: {e}")
            return {"error": "Failed to format dialogue history"}

        # --- Format the Expert Prompt ---
        try:
            # Placeholders expected by multi-turn expert prompt: {json_format}, {expert_full_info}, {dialogue}
            expert_prompt_filled = self.expert_prompt_template.format(
                json_format=self.json_format_str,
                expert_full_info=expert_full_info,
                dialogue=dialogue_json_str
            )
        except KeyError as e:
            logger.error(f"Missing placeholder in multi-turn expert prompt template: {e}")
            return {"error": f"Multi-turn expert prompt formatting error: missing placeholder '{e}'"}
        except Exception as e:
             logger.error(f"Unexpected error formatting multi-turn expert prompt: {e}")
             return {"error": "Unexpected error formatting multi-turn expert prompt"}

        expert_messages = [{"role": "user", "content": expert_prompt_filled}]

        # --- Call Expert LLM ---
        logger.info(f"Sending multi-turn dialogue to expert model '{self.config.expert_model_name}' for evaluation...")
        expert_response_content = None
        parsed_evaluation = {"error": "Evaluation not attempted or failed early"} # Default error state

        # Use the generic response getter (handles retries internally)
        expert_response_content = get_llm_response(
             client=self.expert_client,
             model_name=self.config.expert_model_name,
             messages=expert_messages,
             config=self.config,
             is_test_model=False # Expert is not the model under test
        )

        # --- Parse Evaluation ---
        if expert_response_content:
            parsed_evaluation = parse_expert_evaluation(expert_response_content)
            if "error" not in parsed_evaluation:
                logger.info("Expert evaluation received and parsed successfully (multi-turn).")
            else:
                logger.error(f"Failed to parse multi-turn expert evaluation: {parsed_evaluation.get('error')}")
                # Keep the raw response in the error dict if parsing failed
                parsed_evaluation["raw_response"] = expert_response_content if len(expert_response_content)<2000 else expert_response_content[:2000]+"..."
        else:
            logger.error(f"Failed to get response from expert model '{self.config.expert_model_name}' (multi-turn).")
            parsed_evaluation = {"error": "Expert model failed to respond"}


        return parsed_evaluation